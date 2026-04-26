from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load env
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

# 🔹 PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.pkl")

print("📂 Loading from:", CHUNKS_PATH)

# 🔹 LOAD CHUNKS
chunks = []

def load_chunks():
    global chunks
    try:
        if not os.path.exists(CHUNKS_PATH):
            print("❌ chunks not found")
            return

        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        print(f"✅ Loaded {len(chunks)} chunks")

    except Exception as e:
        print("❌ Load error:", str(e))

load_chunks()

# 🔹 ROOT
@app.get("/")
def home():
    return {"status": "running"}

# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "chunks_loaded": len(chunks),
        "path": CHUNKS_PATH,
        "api_key": bool(GROQ_API_KEY)
    }

# 🔥 IMPROVED RETRIEVAL (LESS NOISE)
def retrieve_context(query, k=3):
    if not chunks:
        return ""

    query = query.lower()

    synonyms = {
        "leave": ["leave", "time off", "vacation", "sick leave", "fmla"],
        "salary": ["salary", "pay", "wage", "compensation"],
        "benefits": ["benefits", "insurance", "health", "coverage"]
    }

    expanded = query.split()

    for word in query.split():
        if word in synonyms:
            expanded.extend(synonyms[word])

    scored = []

    for chunk in chunks:
        try:
            text = chunk.lower()
            score = sum(1 for word in expanded if word in text)

            # 🔥 stricter filtering
            if score >= 2:
                scored.append((score, chunk))
        except:
            continue

    scored.sort(reverse=True, key=lambda x: x[0])

    selected = [chunk for _, chunk in scored[:k]]

    if not selected:
        selected = chunks[:k]

    return "\n".join(selected)

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
        "temperature": 0.2
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)

        if r.status_code != 200:
            return f"API Error: {r.text}"

        res = r.json()
        return res["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"

# 🔥 MAIN ENDPOINT
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    context = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

Answer clearly and concisely using ONLY the context.

- Do NOT add extra explanation
- Do NOT repeat points
- Keep answer short and precise
- If not found, say: "Not found in document"

Context:
{context}

Question:
{q}
"""

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer,
        "context_used": context[:200]
    }