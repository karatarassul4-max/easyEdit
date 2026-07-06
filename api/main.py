import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI()

# Инициализация ключей из переменных окружения
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class UserRequest(BaseModel):
    text: str

@app.post("/match-clip")
def match_clip(payload: UserRequest):
    try:
        # 1. Генерируем эмбеддинг для поискового запроса пользователя
        # Для моделей семейства E5 критически важно добавлять префикс "query: " к поисковому запросу!
        hf_url = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large"
        hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        hf_payload = {"inputs": f"query: {payload.text}"}
        
        hf_res = requests.post(hf_url, json=hf_payload, headers=hf_headers)
        
        if hf_res.status_code != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Ошибка генерации эмбеддинга Hugging Face: {hf_res.text}"
            )
            
        query_vector = hf_res.json()
        
        # 2. Вызываем RPC-функцию в Supabase для поиска по косинусному сходству
        # match_threshold: 0.3 (минимальный порог похожести)
        # match_count: 1 (нам нужен один лучший результат)
        rpc_res = supabase.rpc(
            "match_clips", 
            {
                "query_embedding": query_vector, 
                "match_threshold": 0.3, 
                "match_count": 1
            }
        ).execute()
        
        # 3. Обрабатываем результат поиска
        if not rpc_res.data:
            # Фолбэк: если ничего не нашлось по порогу, берем самый первый клип из таблицы
            fallback = supabase.table("anime_clips_vector").select("*").limit(1).execute()
            if not fallback.data:
                raise HTTPException(status_code=404, detail="В базе данных вообще нет клипов")
            best_match = fallback.data[0]
        else:
            best_match = rpc_res.data[0]
            
        return {
            "id": best_match["id"],
            "title": best_match["title"],
            "video_url": best_match["video_url"],
            "description": best_match["description"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка бэкенда: {str(e)}")
