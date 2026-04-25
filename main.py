from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load environment variables
load_dotenv()

app = FastAPI(title="Stable RAG Backend")

# 🔹 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔹 Load chunks safely
print("🔄 Loading chunks...")

chunks = []

try:
    if os.path.exists("chunks.pkl"):
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ Loaded {len(chunks)} chunks")
    else:
        print("❌ chunks.pkl not found")
except Exception as e:
    print("❌ Error loading chunks:", str(e))


# 🔹 ROOT
@app.get("/")
def home():
    return {"status": "running", "message": "Backend Live"}


# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "api_key_loaded": bool(GROQ_API_KEY),
        "chunks_loaded": len(chunks)
    }


# 🔹 SIMPLE RETRIEVAL (SAFE)
def retrieve_context(query, k=5):
    if not chunks:
        return ""

    query = query.lower()
    results = []

    for chunk in chunks:
        if query in chunk.lower():
            results.append(chunk)

    # fallback
    if not results:
        results = chunks[:k]

    return "\n".join(results[:k])


# 🔹 GROQ API CALL
def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)

        if r.status_code != 200:
            return f"API Error: {r.text}"

        result = r.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


# 🔥 MAIN ENDPOINT
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    context = retrieve_context(q)

    prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{q}
"""

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer
    }