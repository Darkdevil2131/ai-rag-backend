from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load environment variables
load_dotenv()

app = FastAPI(title="Production RAG Backend")

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

# 🔹 BASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.pkl")

print("📂 Expected chunks path:", CHUNKS_PATH)

# 🔹 LOAD DATA SAFELY
chunks = []

def load_chunks():
    global chunks
    try:
        if not os.path.exists(CHUNKS_PATH):
            print("❌ chunks.pkl not found")
            chunks = []
            return

        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        if not isinstance(chunks, list):
            print("⚠️ Invalid chunks format")
            chunks = []

        print(f"✅ Loaded {len(chunks)} chunks")

    except Exception as e:
        print("❌ Error loading chunks:", str(e))
        chunks = []

# Load at startup
load_chunks()

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
        "chunks_path": CHUNKS_PATH
    }

# 🔹 SAFE RETRIEVAL
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

    scored = []

    for chunk in chunks:
        try:
            text = chunk.lower()
            score = sum(1 for word in expanded_words if word in text)

            if score > 0:
                scored.append((score, chunk))
        except:
            continue

    scored.sort(reverse=True, key=lambda x: x[0])

    top = [chunk for _, chunk in scored[:k]]

    if not top:
        top = chunks[:k]

    return "\n".join(top)

# 🔹 GROQ CALL (SAFE)
def call_groq(prompt):
    if not GROQ_API_KEY:
        return "Error: Missing API key"

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

        res = r.json()

        return res["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Request failed: {str(e)}"

# 🔥 MAIN ENDPOINT
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    context = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

Answer using the context below.
If partial info exists, answer what you can.
Say "Not found in document" ONLY if nothing relevant exists.

Context:
{context}

Question:
{q}
"""

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer,
        "context_preview": context[:300],
        "chunks_used": len(context.split("\n"))
    }