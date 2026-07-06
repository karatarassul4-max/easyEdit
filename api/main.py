import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Anime Clip Matcher MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Загружаем ту же самую модель на Vercel
# Модель легковесная, Vercel успеет её подгрузить
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class SearchRequest(BaseModel):
    text: str

class ClipResponse(BaseModel):
    id: int
    title: str
    video_url: str
    description: str
    similarity: float

@app.post("/match-clip", response_model=ClipResponse)
async def match_clip(payload: SearchRequest):
    try:
        # Генерируем вектор из текста пользователя (384 измерения)
        user_embedding = model.encode(payload.text).tolist()
        
        # Ищем в Supabase
        response = supabase.rpc(
            "match_clips",
            {
                "query_embedding": user_embedding,
                "match_threshold": 0.1,  # Снизим порог для теста
                "match_count": 1
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
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")
