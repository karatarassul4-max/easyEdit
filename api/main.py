import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class UserRequest(BaseModel):
    text: str

@app.post("/match-clip")
def match_clip(payload: UserRequest):
    try:
        # 1. Генерируем эмбеддинг
        hf_url = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large"
        hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        hf_payload = {"inputs": f"query: {payload.text}"}
        
        hf_res = requests.post(hf_url, json=hf_payload, headers=hf_headers)
        
        if hf_res.status_code != 200:
            return {"error": f"Hugging Face вернул статус {hf_res.status_code}", "details": hf_res.text}
            
        query_vector = hf_res.json()
        
        # ЗАЩИТА: Проверяем, что HF вернул вектор (список), а не словарь с ошибкой загрузки модели
        if isinstance(query_vector, dict) and "error" in query_vector:
            return {
                "error": "Модель Hugging Face еще загружается, подожди немного",
                "details": query_vector
            }
            
        # 2. Вызываем RPC в Supabase
        try:
            rpc_res = supabase.rpc(
                "match_clips", 
                {
                    "query_embedding": query_vector, 
                    "match_threshold": 0.1, # Немного снизим порог для теста
                    "match_count": 1
                }
            ).execute()
        except Exception as sb_err:
            return {"error": "Ошибка при вызове RPC в Supabase", "details": str(sb_err)}
        
        # 3. Обрабатываем результат
        if not rpc_res.data:
            fallback = supabase.table("anime_clips_vector").select("*").limit(1).execute()
            if not fallback.data:
                return {"error": "В базе данных anime_clips_vector вообще нет записей"}
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
        # Теперь любая непредвиденная ошибка вернется в красивом JSON, а не уронит сервер в 500
        return {"error": "Критическое исключение бэкенда", "details": str(e)}
