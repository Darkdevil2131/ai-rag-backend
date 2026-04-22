from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import json

app = FastAPI()

# ✅ CORS (IMPORTANT)
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

# ✅ ENV
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ ASK ENDPOINT
@app.get("/ask")
async def ask(q: str):
    def generate():
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": q}],
            "stream": True,
        }

        with requests.post(url, headers=headers, json=data, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    decoded = line.decode("utf-8")

                    if decoded.startswith("data: "):
                        chunk = decoded.replace("data: ", "")

                        if chunk == "[DONE]":
                            break

                        try:
                            content = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                            yield content
                        except:
                            continue

    return StreamingResponse(generate(), media_type="text/plain")