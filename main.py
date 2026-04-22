from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {"message": "🔥 RAG API Running"}

# =========================
# DEBUG
# =========================
@app.get("/debug")
def debug():
    return {"message": "DEBUG ROUTE WORKING"}

# =========================
# API KEY
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# STREAMING ROUTE
# =========================
@app.get("/ask")
async def ask(q: str):

    def generate():
        print("🔥 /ask HIT:", q)

        if not GROQ_API_KEY:
            yield "ERROR: Missing GROQ_API_KEY\n"
            return

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "llama3-8b-8192",   # ✅ FIXED MODEL
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
            yield f"ERROR: {str(e)}\n"

    return StreamingResponse(generate(), media_type="text/plain")


# =========================
# NON-STREAM (TEST ROUTE)
# =========================
@app.get("/ask-json")
def ask_json(q: str):

    print("🔥 /ask-json HIT:", q)

    if not GROQ_API_KEY:
        return {"error": "Missing GROQ_API_KEY"}

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama3-8b-8192",   # ✅ FIXED MODEL
        "messages": [
            {"role": "user", "content": q}
        ],
    }

    try:
        r = requests.post(url, headers=headers, json=data)
        return r.json()
    except Exception as e:
        return {"error": str(e)}