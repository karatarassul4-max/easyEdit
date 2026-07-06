import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from openai import AsyncOpenAI

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
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

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
        # Получаем эмбеддинг через внешний API
        response = await ai_client.embeddings.create(
            model="text-embedding-3-small", 
            input=[payload.text]
        )
        user_embedding = response.data[0].embedding
        
        # Поиск в Supabase
        res = supabase.rpc(
            "match_clips",
            {
                "query_embedding": user_embedding,
                "match_threshold": 0.1,
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
