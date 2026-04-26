from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load ENV
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

# 🔹 PATH SETUP (Render-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.pkl")

print("📂 Loading chunks from:", CHUNKS_PATH)

# 🔹 LOAD CHUNKS
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
        print("❌ Error loading chunks:", str(e))

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

# 🔥 FINAL RETRIEVAL SYSTEM (BALANCED + DEDUP)
def retrieve_context(query, k=3):
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

            # 🔥 Topic filtering
            if topic_words:
                if any(word in text for word in topic_words):
                    score += 3
                else:
                    continue

            # 🔥 Keyword scoring
            for word in query.split():
                if word in text:
                    score += 2

            # 🔥 Exact phrase boost
            if query in text:
                score += 5

            if score > 0:
                scored.append((score, chunk))

        except:
            continue

    # 🔥 Sort best matches
    scored.sort(reverse=True, key=lambda x: x[0])

    # 🔥 Deduplicate + select top k
    selected = []
    seen = set()

    for score, chunk in scored:
        snippet = chunk[:100]

        if snippet not in seen:
            selected.append(chunk)
            seen.add(snippet)

        if len(selected) >= k:
            break

    return "\n\n".join(selected)


# 🔹 GROQ API CALL
def call_groq(prompt):
    if not GROQ_API_KEY:
        return "Error: Missing GROQ API key"

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

        result = r.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


# 🔥 MAIN ENDPOINT
@app.get("/ask")
def ask(q: str):

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    context = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

RULES:
- Answer ONLY using the provided context
- Combine relevant points into a clear answer
- Do NOT guess or invent information
- Do NOT repeat content
- If partial info exists, answer that
- If nothing relevant, say: "Not found in document"

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