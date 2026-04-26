from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load environment
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

# 🔥 CORRECT PATH HANDLING (THIS FIXES YOUR ISSUE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")

print("📂 Looking for chunks at:", CHUNKS_PATH)

chunks = []

try:
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ Loaded {len(chunks)} chunks")
    else:
        print("❌ chunks.pkl NOT FOUND at:", CHUNKS_PATH)
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
        "chunks_loaded": len(chunks),
        "chunks_path": CHUNKS_PATH
    }


# 🔥 SMART RETRIEVAL (SCORING + SYNONYMS)
def retrieve_context(query, k=5):
    if not chunks:
        return ""

    query = query.lower()

    synonyms = {
        "leave": ["leave", "time off", "vacation", "sick leave", "fmla"],
        "salary": ["salary", "pay", "wage", "compensation"],
        "benefits": ["benefits", "insurance", "health", "coverage"]
    }

    expanded_words = query.split()

    for word in query.split():
        if word in synonyms:
            expanded_words.extend(synonyms[word])

    scored_chunks = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in expanded_words if word in chunk_lower)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    top_chunks = [chunk for _, chunk in scored_chunks[:k]]

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


# 🔥 MAIN API
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    context = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

Use the context below to answer the question.

If partial information exists, answer as much as possible.
Only say "Not found in document" if nothing relevant exists.

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