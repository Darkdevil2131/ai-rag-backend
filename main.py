from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
import os

app = FastAPI()

# ✅ CORS (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ROOT ROUTE
@app.get("/")
def home():
    return {"message": "🔥 RAG API Running"}

# ✅ API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ ASK ROUTE (THIS IS WHAT YOU NEED)
@app.get("/ask")
def ask(q: str):
    def generate():
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": q}],
            "stream": True,
        }

        with requests.post(url, headers=headers, json=data, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"

    return StreamingResponse(generate(), media_type="text/plain")