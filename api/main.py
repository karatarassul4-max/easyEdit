import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "FastAPI работает, ленивая инициализация активна!"
    }

@app.post("/match-clip")
def match_clip(payload: UserRequest):
    try:
        # Импортируем локально, чтобы сбой импорта не вешал весь сервер при старте
        from supabase import create_client
        
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            return {
                "error": "Критические переменные окружения Supabase отсутствуют в Vercel!",
                "URL_exists": bool(SUPABASE_URL),
                "KEY_exists": bool(SUPABASE_KEY)
            }

        # Инициализируем клиент внутри try-блока
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # 1. Генерируем эмбеддинг
        hf_url = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large"
        hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        hf_payload = {"inputs": f"query: {payload.text}"}
        
        try:
            hf_res = requests.post(hf_url, json=hf_payload, headers=hf_headers)
            if hf_res.status_code != 200:
                return {"error": f"HF вернул код {hf_res.status_code}", "details": hf_res.text}
            query_vector = hf_res.json()
        except Exception as hf_err:
            return {"error": "Сбой сетевого запроса к Hugging Face", "details": str(hf_err)}
            
        if isinstance(query_vector, dict) and "error" in query_vector:
            return {"error": "Модель Hugging Face загружается", "details": query_vector}
            
        # 2. Вызываем RPC в Supabase
        try:
            rpc_res = supabase.rpc(
                "match_clips", 
                {
                    "query_embedding": query_vector, 
                    "match_threshold": 0.0, # Ставим 0.0 для теста, чтобы поймать вообще всё
                    "match_count": 1
                }
            ).execute()
        except Exception as sb_err:
            return {"error": "Ошибка при вызове RPC в Supabase", "details": str(sb_err)}
        
        # 3. Отдаем результат
        if not rpc_res.data:
            fallback = supabase.table("anime_clips_vector").select("*").limit(1).execute()
            if not fallback.data:
                return {"error": "Таблица anime_clips_vector пуста"}
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
        return {"error": "Глобальная ошибка выполнения внутри роута", "details": str(e)}
