from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load env
load_dotenv()

app = FastAPI(title="Stable RAG Backend")

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

# 🔹 LOAD CHUNKS
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


# 🔥 IMPROVED RETRIEVAL (SCORING BASED)
def retrieve_context(query, k=5):
    if not chunks:
        return ""

    query_words = query.lower().split()
    scored_chunks = []

    for chunk in chunks:
        chunk_lower = chunk.lower()

        # score = number of matching words
        score = sum(1 for word in query_words if word in chunk_lower)

        if score > 0:
            scored_chunks.append((score, chunk))

    # sort best first
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    # take top k
    top_chunks = [chunk for _, chunk in scored_chunks[:k]]

    # fallback if nothing matches
    if not top_chunks:
        top_chunks = chunks[:k]

    return "\n".join(top_chunks)


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
You are an AI assistant.

Answer the question using ONLY the context below.

If the answer is not found, say: "Not found in document."

Be clear and direct.

Context:
{context}

Question:
{q}
"""

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer,
        "context_preview": context[:300]
    }