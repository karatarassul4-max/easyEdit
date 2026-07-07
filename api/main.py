import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class VectorRequest(BaseModel):
    vector: list # Принимаем готовый 512-мерный вектор из OpenCLIP

@app.get("/")
def read_root():
    return {"status": "online", "message": "Бэкенд готов принимать новые эмбеддинги кадров!"}

@app.post("/match-clip")
def match_clip(payload: VectorRequest):
    try:
        from supabase import create_client
        
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Вызываем именно твою сегодняшнюю RPC-функцию
        try:
            rpc_res = supabase.rpc(
                "match_anime_clips_clip", 
                {
                    "query_embedding": payload.vector, 
                    "match_threshold": 0.0, 
                    "match_count": 1
                }
            ).execute()
        except Exception as sb_err:
            return {"error": "Ошибка при вызове RPC в Supabase", "details": str(sb_err)}
        
        # Фоллбэк, если поиск ничего не выдал (запрашиваем только существующие поля)
        if not rpc_res.data:
            fallback = supabase.table("anime_clips_clip").select("id", "title", "video_url").limit(1).execute()
            best_match = fallback.data[0]
        else:
            best_match = rpc_res.data[0]
            
        # Возвращаем строго то, что есть в таблице
        return {
            "id": best_match["id"],
            "title": best_match["title"],
            "video_url": best_match["video_url"]
        }
        
    except Exception as e:
        return {"error": "Критическая ошибка бэкенда", "details": str(e)}
