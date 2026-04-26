from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
import pickle

# 🔥 Load environment variables
load_dotenv()

app = FastAPI(title="Reliable RAG Backend")

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

# 🔹 PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.pkl")

print("📂 Loading chunks from:", CHUNKS_PATH)

# 🔹 LOAD CHUNKS
chunks = []

def load_chunks():
    global chunks
    try:
        if not os.path.exists(CHUNKS_PATH):
            print("❌ chunks.pkl not found")
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
    return {"status": "running"}

# 🔹 DEBUG
@app.get("/debug")
def debug():
    return {
        "chunks_loaded": len(chunks),
        "chunks_path": CHUNKS_PATH,
        "api_key": bool(GROQ_API_KEY)
    }

# 🔥 CLEAN RETRIEVAL (FILTERED)
def retrieve_context(query, k=3):
    if not chunks:
        return ""

    query = query.lower()

    synonyms = {
        "leave": ["leave", "time off", "vacation", "sick leave", "fmla", "absence"],
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

            # ignore small/noisy chunks
            if len(text.strip()) < 50:
                continue

            if "electronic communication" in text or "social media" in text:
                continue

            score = 0
            for word in expanded:
                if word in text:
                    score += 2

            if query in text:
                score += 5

            if score >= 3:
                scored.append((score, chunk))

        except:
            continue

    scored.sort(reverse=True, key=lambda x: x[0])

    selected = [chunk for _, chunk in scored[:k]]

    if not selected:
        selected = chunks[:k]

    return "\n".join(selected[:2])  # 🔥 limit context

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
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    context = retrieve_context(q)

    prompt = f"""
You are an AI assistant.

STRICT RULES:
- Answer ONLY from the context
- Do NOT guess or assume
- Do NOT add extra information
- If answer is not clearly present, say: "Not found in document"

Keep answer short and factual.

Context:
{context}

Question:
{q}
"""

    # 🔥 prevent fake numbers
    if any(word in q.lower() for word in ["how many", "number", "days"]):
        prompt += "\nOnly include numbers if explicitly mentioned in context."

    answer = call_groq(prompt)

    return {
        "query": q,
        "answer": answer,
        "context_preview": context[:200]
    }