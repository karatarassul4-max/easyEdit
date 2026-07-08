import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Anime Clip Matcher API with Debug")

# Настройка CORS для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        GROQ_API_KEY = os.environ.get("LLM_API_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise HTTPException(status_code=500, detail="Конфигурация Supabase отсутствует")
            
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Словарь, который мы вернем на фронтенд для анализа проблем
        debug_info = {
            "groq_activated": False,
            "groq_error": None,
            "raw_judge_decision": None,
            "candidates_offered": []
        }
        
        try:
            # Вытягиваем ТОП-10 кандидатов через RPC-функцию
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
        
        # Если база пуста — мягкий фолбек
        if not candidates:
            fallback = supabase.table("anime_clips_clip").select("id", "title", "video_url").limit(1).execute()
            if fallback.data:
                return {
                    "id": fallback.data[0]["id"],
                    "title": fallback.data[0]["title"],
                    "video_url": fallback.data[0]["video_url"],
                    "debug": {"groq_activated": False, "groq_error": "База пуста, сработал глобальный фолбек", "candidates_offered": []}
                }
            return {"error": "База данных полностью пуста"}

        # Записываем, что именно нам вернул векторный поиск
        debug_info["candidates_offered"] = [
            {"id": c.get("id"), "title": c.get("title")} for c in candidates
        ]

        # Если нашелся всего один кандидат или ключ API не задан — реранкинг пропускается
        if len(candidates) == 1 or not GROQ_API_KEY:
            if not GROQ_API_KEY:
                debug_info["groq_error"] = "Ключ LLM_API_KEY не найден в переменных окружения Vercel"
            return {
                "id": candidates[0]["id"],
                "title": candidates[0]["title"],
                "video_url": candidates[0]["video_url"],
                "debug": debug_info
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
            
            debug_info["raw_judge_decision"] = response.choices[0].message.content
            judge_decision = json.loads(response.choices[0].message.content)
            best_id = int(judge_decision.get("best_id"))
            
            # Находим выбранный объект, иначе берем первый
            best_match = next((c for c in candidates if c["id"] == best_id), candidates[0])
            debug_info["groq_activated"] = True
            
        except Exception as groq_err:
            debug_info["groq_error"] = f"Сбой этапа Groq: {str(groq_err)}"
            best_match = candidates[0]
            
        return {
            "id": best_match["id"],
            "title": best_match["title"],
            "video_url": best_match["video_url"],
            "debug": debug_info
        }
        
    except Exception as e:
        return {"error": "Критическая ошибка бэкенда", "details": str(e)}
