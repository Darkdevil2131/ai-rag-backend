from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv

# 🔹 RAG LOAD
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# 🔥 LOAD ENV
load_dotenv()

app = FastAPI()

# 🔹 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("API KEY LOADED:", GROQ_API_KEY[:10] if GROQ_API_KEY else "NOT FOUND")

# 🔹 LOAD EMBEDDING MODEL
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 LOAD INDEX + DATA
print("🔄 Loading FAISS index...")

index = faiss.read_index("faiss.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("sources.pkl", "rb") as f:
    sources = pickle.load(f)

print(f"✅ Loaded {len(chunks)} chunks")

# 🔹 ROOT
@app.get("/")
def home():
    return {"message": "🔥 Multi-PDF RAG API Running (Improved)"}

# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "api_key_loaded": True if GROQ_API_KEY else False,
        "chunks_loaded": len(chunks),
        "sources_loaded": len(sources)
    }

# 🔹 RETRIEVAL (IMPROVED)
def retrieve_context(query, k=8):  # 🔥 increased k
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    used_sources = []

    for i in indices[0]:
        results.append(chunks[i])
        used_sources.append(sources[i])

    return "\n".join(results), list(set(used_sources))

# 🔥 FINAL ASK ENDPOINT
@app.get("/ask")
def ask(q: str):

    if not GROQ_API_KEY:
        return {"error": "Missing GROQ_API_KEY"}

    context, used_sources = retrieve_context(q)

    # 🔥 IMPROVED PROMPT
    prompt = f"""
You are an AI assistant.

Answer the question using ONLY the context below.

If the answer is partially available, answer as much as possible.
Only say "Not found in document" if absolutely no relevant info exists.

Context:
{context}

Question:
{q}
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        r = requests.post(url, headers=headers, json=data)
        result = r.json()

        answer = result["choices"][0]["message"]["content"]

        return {
            "answer": answer,
            "sources": used_sources
        }

    except Exception as e:
        return {"error": str(e)}