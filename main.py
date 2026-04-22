from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
import os

app = FastAPI()

# ✅ CORS (required for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ROOT
@app.get("/")
def home():
    return {"message": "🔥 RAG API Running"}

# ✅ TEST ROUTE
@app.get("/test")
def test():
    return {"status": "ok"}

# ✅ ASK ROUTE
@app.get("/ask")
async def ask(q: str):

    def generate():
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        if not GROQ_API_KEY:
            yield "ERROR: GROQ_API_KEY not set"
            return

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "user", "content": q}
            ],
            "stream": True,
        }

        try:
            with requests.post(url, headers=headers, json=data, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n"
        except Exception as e:
            yield f"ERROR: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")