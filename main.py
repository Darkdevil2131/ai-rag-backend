from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load ENV
load_dotenv()

app = FastAPI(title="RAG Backend Final")

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

# 🔹 PATH SETUP (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.pkl")

print("📂 Loading from:", CHUNKS_PATH)

# 🔹 LOAD DATA
chunks = []

def load_chunks():
    global chunks
    try:
        if not os.path.exists(CHUNKS_PATH):
            print("❌ chunks.pkl NOT FOUND")
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
    return {"status": "running", "message": "Backend Live"}

# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "api_key_loaded": bool(GROQ_API_KEY),
        "chunks_loaded": len(chunks),
        "chunks_path": CHUNKS_PATH
    }

# 🔥 SMART RETRIEVAL (BALANCED)
def retrieve_context(query, k=2):
    if not chunks:
        return ""

    query = query.lower()

    topic_map = {
        "leave": ["leave", "vacation", "sick", "fmla", "absence"],
        "benefits": ["benefits", "insurance", "401", "compensation"],
        "salary": ["salary", "pay", "wage"]
    }

    topic_words = []
    for key in topic_map:
        if key in query:
            topic_words = topic_map[key]
            break

    scored = []

    for chunk in chunks:
        try:
            text = chunk.lower()

            if len(text.strip()) < 50:
                continue

            score = 0

            # topic boost
            if topic_words:
                if any(word in text for word in topic_words):
                    score += 3
                else:
                    continue

            # keyword match
            for word in query.split():
                if word in text:
                    score += 2

            # exact match boost
            if query in text:
                score += 5

            if score > 0:
                scored.append((score, chunk))

        except:
            continue

    scored.sort(reverse=True, key=lambda x: x[0])

    selected = [chunk for _, chunk in scored[:k]]

    return "\n".join(selected)


# 🔹 GROQ CALL
def call_groq(prompt):
    if not GROQ_API_KEY:
        return "Missing API key"

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


# 🔥 MAIN API
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    context = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

RULES:
- Answer ONLY using context
- Do NOT guess
- If partial info exists, answer that
- If nothing found, say: "Not found in document"

Context:
{context}

Question:
{q}
"""

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer,
        "context_preview": context[:200]
    }