from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# =========================
# CORS (IMPORTANT)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROOT ROUTE
# =========================
@app.get("/")
def home():
    return {"message": "🔥 RAG API Running"}

# =========================
# HEALTH CHECK (DEBUG)
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# LOAD API KEY
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# ASK ROUTE (STREAMING)
# =========================
@app.get("/ask")
async def ask(q: str):

    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "Missing GROQ_API_KEY"}
        )

    def generate():
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
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

# =========================
# NON-STREAM DEBUG ROUTE
# =========================
@app.get("/ask-json")
def ask_json(q: str):

    if not GROQ_API_KEY:
        return {"error": "Missing GROQ_API_KEY"}

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": q}],
    }

    try:
        r = requests.post(url, headers=headers, json=data)
        return r.json()
    except Exception as e:
        return {"error": str(e)}