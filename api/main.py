import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI

app = FastAPI(title="Anime Clip Matcher MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация баз данных
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Настройка Groq (используем официальный SDK OpenAI, так как Groq полностью с ним совместим)
GROQ_API_KEY = os.getenv("LLM_API_KEY") # Сюда должен быть вставлен твой gsk_...
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

class SearchRequest(BaseModel):
    text: str

class ClipResponse(BaseModel):
    id: int
    title: str
    video_url: str
    description: str

@app.post("/match-clip", response_model=ClipResponse)
async def match_clip(payload: SearchRequest):
    try:
        # 1. Просим ИИ вытащить 핵심-слова (ключевые слова для поиска)
        prompt = f"""Выдели из этого текста 2-3 главных ключевых слова для поиска в базе данных. 
        Пиши ТОЛЬКО ключевые слова через пробел, без лишнего текста, знаков препинания и объяснений.
        Текст: {payload.text}"""
        
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",  # <--- ПОМЕНЯЛИ МОДЕЛь ТУТ
            temperature=0.1
        )
        
        search_keywords = chat_completion.choices[0].message.content.strip()
        print(f"ИИ превратил запрос в ключевые слова: {search_keywords}")

        # 2. Делаем текстовый поиск в Supabase по ключевым словам
        # 2. Делаем текстовый поиск в Supabase с правильным порядком цепочки методов
        res = supabase.table("anime_clips") \
            .select("*") \
            .limit(1) \
            .text_search("description", search_keywords) \
            .execute()
        
        # Если ничего не нашлось по точным словам, берем просто первый попавшийся клип для теста,
        # чтобы фронтенд не падал
        if not res.data:
            res = supabase.table("anime_clips").select("*").limit(1).execute()
            
        if not res.data:
            raise HTTPException(status_code=404, detail="В базе вообще нет клипов")
            
        best_match = res.data[0]
        
        return ClipResponse(
            id=best_match["id"],
            title=best_match["title"],
            video_url=best_match["video_url"],
            description=best_match["description"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка бэкенда: {str(e)}")
