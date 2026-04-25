from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle
import numpy as np

# 🔥 Load environment variables
load_dotenv()

app = FastAPI(title="Production RAG API")

# 🔹 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔹 Load chunks at startup (lightweight)
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


# 🔹 Lazy load embeddings (heavy → load only when needed)
def load_embeddings():
    try:
        if os.path.exists("embeddings.pkl"):
            with open("embeddings.pkl", "rb") as f:
                return np.array(pickle.load(f))
    except Exception as e:
        print("❌ Error loading embeddings:", str(e))
    return None


# 🔹 Get embedding from API (NO local model → safe for Render)
def get_embedding(text):
    url = "https://api.groq.com/openai/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "text-embedding-3-small",
        "input": text
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)
        result = r.json()
        return np.array(result["data"][0]["embedding"])
    except Exception as e:
        print("❌ Embedding error:", str(e))
        return None


# 🔹 Semantic Retrieval (cosine similarity)
def retrieve_context(query, k=5):
    embeddings = load_embeddings()

    if embeddings is None or len(chunks) == 0:
        return "", 0.0

    query_vec = get_embedding(query)

    if query_vec is None:
        return "", 0.0

    scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )

    top_k = np.argsort(scores)[-k:][::-1]

    selected_chunks = [chunks[i] for i in top_k]

    confidence = float(scores[top_k[0]])

    return "\n".join(selected_chunks), confidence


# 🔹 Call Groq LLM
def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)

        if r.status_code != 200:
            return f"API Error: {r.text}"

        result = r.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


# 🔹 ROOT
@app.get("/")
def home():
    return {"status": "running", "message": "Production RAG Live"}


# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "api_key_loaded": bool(GROQ_API_KEY),
        "chunks_loaded": len(chunks)
    }


# 🔥 MAIN ENDPOINT
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    context, confidence = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{q}
"""

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer,
        "confidence": confidence
    }