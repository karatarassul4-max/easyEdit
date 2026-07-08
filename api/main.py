import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Anime Clip Matcher API")

# Настройка CORS, чтобы фронтенд мог спокойно делать запросы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для продакшена лучше указать конкретный домен твоего фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Описание структуры входящего запроса
class VectorRequest(BaseModel):
    vector: List[float]
    query: str

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "Anime Clip Matcher"}

@app.post("/match-clip")
def match_clip(payload: VectorRequest):
    try:
        from supabase import create_client
        
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        # Используем переменную окружения, уже настроенную на Vercel
        GROQ_API_KEY = os.environ.get("LLM_API_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise HTTPException(status_code=500, detail="Конфигурация Supabase отсутствует в переменных окружения")
            
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        try:
            # Вытягиваем ТОП-10 кандидатов через RPC-функцию соответствия векторов
            rpc_res = supabase.rpc(
                "match_anime_clips_clip", 
                {
                    "query_embedding": payload.vector, 
                    "match_threshold": 0.0, 
                    "match_count": 10
                }
            ).execute()
        except Exception as sb_err:
            return {"error": "Ошибка при вызове RPC в Supabase", "details": str(sb_err)}
        
        candidates = rpc_res.data if rpc_res.data else []
        
        # Если векторный поиск ничего не вернул, ищем хотя бы одну запись для фолбека
        if not candidates:
            fallback = supabase.table("anime_clips_clip").select("id", "title", "video_url").limit(1).execute()
            if fallback.data:
                return {
                    "id": fallback.data[0]["id"],
                    "title": fallback.data[0]["title"],
                    "video_url": fallback.data[0]["video_url"]
                }
            return {"error": "База данных пуста"}

        # Если нашелся всего один кандидат или ключ API для Groq не задан — реранкинг пропускается
        if len(candidates) == 1 or not GROQ_API_KEY:
            return {
                "id": candidates[0]["id"],
                "title": candidates[0]["title"],
                "video_url": candidates[0]["video_url"]
            }

        # --- ЭТАП МОДЕЛИ-СУДЬИ (GROQ СУДЬЯ) ---
        try:
            from openai import OpenAI
            
            ai_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY
            )
            
            candidates_text = ""
            for item in candidates:
                candidates_text += f"ID: {item['id']} | Описание: {item['title']}\n"
                
            prompt = f"""Ты — эксперт-судья видеоконтента. Твоя задача — выбрать из списка кандидатов ОДНО видео, которое наиболее точно и полно соответствует текстовому запросу пользователя.

Запрос пользователя: "{payload.query}"

Список кандидатов:
{candidates_text}

Инструкция:
1. Внимательно проанализируй запрос и описания кандидатов.
2. Выбери только ОДИН ID, который подходит лучше всего.
3. Верни результат строго в формате JSON: {{"best_id": <выбранный_id>}}"""

            response = ai_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a precise routing assistant that outputs only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            judge_decision = json.loads(response.choices[0].message.content)
            best_id = int(judge_decision.get("best_id"))
            
            # Находим объект по выбранному ID среди кандидатов
            best_match = next((c for c in candidates if c["id"] == best_id), candidates[0])
            
        except Exception as groq_err:
            print(f"Ошибка Groq: {groq_err}")
            # Мягкий откат на первый (наиболее релевантный по вектору) элемент при любом сбое LLM
            best_match = candidates[0]
            
        return {
            "id": best_match["id"],
            "title": best_match["title"],
            "video_url": best_match["video_url"]
        }
        
    except Exception as e:
        return {"error": "Критическая ошибка бэкенда", "details": str(e)}
