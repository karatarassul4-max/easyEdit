import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

class VectorRequest(BaseModel):
    vector: list  # Возвращаемся к приему готового вектора от клиента

# === ФРОНТЕНД ===
@app.get("/", response_class=HTMLResponse)
def read_root():
    # Забираем токен из окружения бэкенда, чтобы безопасно передать его в JS при загрузке страницы
    # Таким образом, тебе не нужно писать его текстом в коде!
    hf_token = os.environ.get("HF_TOKEN", "")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Anime Semantic Search</title>
        <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
        <style>
            body {{ background-color: #000000; color: #f4f4f5; }}
        </style>
    </head>
    <body class="min-h-screen flex flex-col items-center px-4 py-12 md:py-24">
        <div class="w-full max-w-3xl text-center mb-12">
            <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent mb-4">
                Anime Clip Semantic Search
            </h1>
            <p class="text-zinc-400 text-lg">Ищи сцены из аниме по визуальному смыслу прямо в базе Supabase</p>
        </div>

        <!-- Форма поиска -->
        <div class="w-full max-w-2xl flex flex-col sm:flex-row gap-3 mb-16">
            <input type="text" id="queryInput" 
                   placeholder="Например: a close up of a character face или action scene..." 
                   class="w-full px-4 py-3 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-200 focus:outline-none focus:border-purple-500 placeholder-zinc-500 font-medium transition-all" />
            <button onclick="performSearch()" id="searchBtn"
                    class="px-6 py-3 bg-purple-600 hover:bg-purple-500 text-white font-semibold rounded-xl transition-all active:scale-95 whitespace-nowrap">
                Найти клип
            </button>
        </div>

        <!-- Контейнер для результатов -->
        <div id="resultContainer" class="w-full max-w-2xl hidden">
            <div class="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden p-6 shadow-xl shadow-purple-500/5">
                <h3 id="clipTitle" class="text-xl font-bold text-zinc-100 mb-4 text-center"></h3>
                <div class="relative aspect-video w-full bg-black rounded-lg overflow-hidden">
                    <video id="clipVideo" src="" class="w-full h-full object-cover" controls autoplay loop muted></video>
                </div>
            </div>
        </div>

        <div id="loader" class="text-purple-400 font-medium hidden">Браузер генерирует вектор через Hugging Face и отправляет в Supabase...</div>
        <div id="errorBox" class="text-red-400 font-medium hidden mt-4"></div>

        <script>
            const HF_TOKEN = "{hf_token}";

            async function performSearch() {{
                const query = document.getElementById('queryInput').value.trim();
                const btn = document.getElementById('searchBtn');
                const loader = document.getElementById('loader');
                const resultContainer = document.getElementById('resultContainer');
                const errorBox = document.getElementById('errorBox');
                
                if (!query) return;

                btn.disabled = true;
                loader.classList.remove('hidden');
                resultContainer.classList.add('hidden');
                errorBox.classList.add('hidden');

                try {{
                    // Отправляем запрос на Hugging Face напрямую из браузера, но с правильными заголовками авторизации
                    const hfRes = await fetch(
                        'https://api-inference.huggingface.co/pipeline/feature-extraction/laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
                        {{
                            method: 'POST',
                            headers: {{ 
                                'Authorization': 'Bearer ' + HF_TOKEN,
                                'Content-Type': 'application/json'
                            }},
                            body: JSON.stringify({{ inputs: query }})
                        }}
                    );

                    if (!hfRes.ok) {{
                        const errData = await hfRes.text();
                        throw new Error('Hugging Face API Error: ' + errData);
                    }}
                    
                    let vector = await hfRes.json();

                    // Корректируем размерность, если HF вернул двумерный массив [[...]]
                    if (Array.isArray(vector) && vector.length > 0 && Array.isArray(vector[0])) {{
                        vector = vector[0];
                    }}

                    // Теперь отправляем готовый вектор на наш FastAPI (который Vercel пропустит, так как это внутренний роут)
                    const response = await fetch('/match-clip', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ vector: vector }})
                    }});

                    const result = await response.json();

                    if (result.error) {{
                        throw new Error(result.details || result.error);
                    }}

                    document.getElementById('clipTitle').innerText = result.title;
                    const videoElement = document.getElementById('clipVideo');
                    videoElement.src = result.video_url;
                    
                    resultContainer.classList.remove('hidden');
                }} catch (err) {{
                    errorBox.innerText = 'Произошла ошибка: ' + err.message;
                    errorBox.classList.remove('hidden');
                }} finally {{
                    btn.disabled = false;
                    loader.classList.add('hidden');
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# === БЭКЕНД ===
@app.post("/match-clip")
def match_clip(payload: VectorRequest):
    try:
        from supabase import create_client
        
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
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
        
        if not rpc_res.data:
            fallback = supabase.table("anime_clips_clip").select("id", "title", "video_url").limit(1).execute()
            best_match = fallback.data[0]
        else:
            best_match = rpc_res.data[0]
            
        return {
            "id": best_match["id"],
            "title": best_match["title"],
            "video_url": best_match["video_url"]
        }
        
    except Exception as e:
        return {"error": "Критическая ошибка бэкенда", "details": str(e)}
