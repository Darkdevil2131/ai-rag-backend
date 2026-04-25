from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle
import numpy as np

# 🔥 LOAD ENV
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

# 🔹 ENV
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔹 LOAD DATA
print("🔄 Loading embeddings...")

chunks = []
embeddings = []

try:
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    embeddings = np.array(embeddings)

    print(f"✅ Loaded {len(chunks)} chunks")

except Exception as e:
    print("❌ Load error:", str(e))


# 🔹 ROOT
@app.get("/")
def home():
    return {"status": "running", "message": "Production RAG Live"}


# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "api_key_loaded": bool(GROQ_API_KEY),
        "chunks_loaded": len(chunks),
        "embeddings_loaded": len(embeddings)
    }


# 🔹 EMBEDDING API
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

    response = requests.post(url, headers=headers, json=data, timeout=20)
    result = response.json()

    return np.array(result["data"][0]["embedding"])


# 🔹 RETRIEVAL (COSINE SIMILARITY)
def retrieve_context(query, k=5):
    if len(embeddings) == 0:
        return "", 0.0

    query_vec = get_embedding(query)

    scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )

    top_k = np.argsort(scores)[-k:][::-1]

    selected_chunks = [chunks[i] for i in top_k]

    confidence = float(scores[top_k[0]])

    return "\n".join(selected_chunks), confidence


# 🔹 GROQ CALL
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

    response = requests.post(url, headers=headers, json=data, timeout=20)
    result = response.json()

    return result["choices"][0]["message"]["content"]


# 🔥 ASK ROUTE
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

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