import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import create_client, Client

app = FastAPI(title="Anime Clip Matcher MVP")

# Настраиваем CORS, чтобы фронтенд мог слать запросы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация клиентов (в продакшене данные берутся из .env)
# Для локального теста можно временно вставить строки, но для Vercel настроим Переменные Окружения
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-supabase-url.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "your-supabase-key")
OPENAI_API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
OPENAI_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1") # Можно заменить на Groq/OpenRouter

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Схемы данных
class SearchRequest(BaseModel):
    text: str

class ClipResponse(BaseModel):
    id: int
    title: str
    video_url: str
    description: str
    similarity: float

async def get_embedding(text: str) -> list[float]:
    """Асинхронное получение вектора (Embedding) для текста пользователя"""
    try:
        response = await ai_client.embeddings.create(
            model="text-embedding-3-small", # Или аналогичная бесплатная модель
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации эмбеддинга: {str(e)}")

@app.post("/match-clip", response_model=ClipResponse)
async def match_clip(payload: SearchRequest):
    # 1. Получаем вектор текста пользователя
    user_embedding = await get_embedding(payload.text)
    
    try:
        # 2. Делаем RPC (Remote Procedure Call) запрос в Supabase для векторного поиска
        # Функция match_clips должна быть предварительно создана в PostgreSQL через SQL-редактор
        response = supabase.rpc(
            "match_clips",
            {
                "query_embedding": user_embedding,
                "match_threshold": 0.3, # Минимальный порог схожести
                "match_count": 1        # Нам нужен только 1 самый лучший клип
            }
        ).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Подходящий клип не найден")
            
        best_match = response.data[0]
        
        return ClipResponse(
            id=best_match["id"],
            title=best_match["title"],
            video_url=best_match["video_url"],
            description=best_match["description"],
            similarity=best_match["similarity"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {str(e)}")
