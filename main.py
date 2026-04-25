from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv

# 🔹 RAG
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔥 LOAD ENV
load_dotenv()

app = FastAPI(title="AI RAG Backend")

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

# 🔹 SAFE GLOBALS
index = None
chunks = []
embed_model = None

# 🔹 LOAD RESOURCES SAFELY
print("🔄 Starting backend...")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    index_path = os.path.join(BASE_DIR, "faiss.index")
    chunks_path = os.path.join(BASE_DIR, "chunks.pkl")

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        print(f"✅ Loaded {len(chunks)} chunks")

    else:
        print("❌ Index files missing")

except Exception as e:
    print("❌ Error loading index:", str(e))


# 🔹 ROOT
@app.get("/")
def home():
    return {"status": "running", "message": "🔥 RAG API Live"}


# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "api_key_loaded": bool(GROQ_API_KEY),
        "chunks_loaded": len(chunks),
        "index_loaded": index is not None
    }


# 🔹 LOAD MODEL LAZY (IMPORTANT FOR RENDER)
def get_model():
    global embed_model
    if embed_model is None:
        print("🔄 Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model


# 🔹 RETRIEVE CONTEXT
def retrieve_context(query, k=5):
    if index is None or not chunks:
        return ""

    model = get_model()

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return "\n".join(results)


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

    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)

        if response.status_code != 200:
            return f"API Error: {response.text}"

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error calling model: {str(e)}"


# 🔥 FINAL ASK ROUTE
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    context = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.
If the answer is not found, say "Not found in document".

Context:
{context}

Question:
{q}
"""

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer,
        "context_length": len(context)
    }