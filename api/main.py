import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

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
        # Получаем эмбеддинг через бесплатное API Hugging Face
        hf_url = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        hf_res = requests.post(hf_url, json={"inputs": payload.text})
        user_embedding = hf_res.json()
        
        if not isinstance(user_embedding, list):
            raise HTTPException(status_code=500, detail="Ошибка генерации вектора на Hugging Face")

        # Поиск в Supabase
        res = supabase.rpc(
            "match_clips",
            {
                "query_embedding": user_embedding,
                "match_threshold": 0.0, -- Порог в ноль для теста
                "match_count": 1
            }
        ).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Клип не найден")
            
        best_match = res.data[0]
        
        return ClipResponse(
            id=best_match["id"],
            title=best_match["title"],
            video_url=best_match["video_url"],
            description=best_match["description"],
            similarity=best_match["similarity"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
