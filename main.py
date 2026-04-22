from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
import uuid
import json
from pypdf import PdfReader

# ------------------------
# 🚀 INIT
# ------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# 🧠 MODEL
# ------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# 💾 VECTOR DB
# ------------------------
chroma_client = chromadb.Client(
    chromadb.config.Settings(persist_directory="./chroma_db")
)

collection = chroma_client.get_or_create_collection(name="docs")

# ------------------------
# 🧠 MEMORY
# ------------------------
chat_history = []

# ------------------------
# 📄 PDF
# ------------------------
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ------------------------
# 📥 LOAD PDF ON START
# ------------------------
@app.on_event("startup")
def load_pdfs():
    print("📂 Loading PDFs...")

    if collection.count() > 0:
        print("✅ Using existing DB")
        return

    folder = "data"
    if not os.path.exists(folder):
        return

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)

            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)

            embeddings = model.encode(chunks).tolist()
            ids = [str(uuid.uuid4()) for _ in chunks]

            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"source": file}] * len(chunks)
            )

    print("✅ PDFs stored")

# ------------------------
# 🤖 STREAMING RAG + MEMORY
# ------------------------
@app.get("/ask")
def ask(q: str):
    global chat_history

    query_embedding = model.encode([q]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    docs = results["documents"][0]
    context = "\n\n".join(docs)

    history_text = ""
    for msg in chat_history[-6:]:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are an AI assistant.

Use both context and conversation history.

Conversation:
{history_text}

Context:
{context}

User:
{q}
"""

    def generate():
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        full = ""

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                token = data.get("response", "")
                full += token
                yield token

        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": full})

    return StreamingResponse(generate(), media_type="text/plain")

# ------------------------
# 📤 UPLOAD
# ------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)

    embeddings = model.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": file.filename}] * len(chunks)
    )

    os.remove(path)

    return {"message": "Uploaded"}

# ------------------------
# 🔄 RESET
# ------------------------
@app.get("/reset")
def reset():
    global chat_history
    chat_history = []
    return {"message": "Memory cleared"}

# ------------------------
# ROOT
# ------------------------
@app.get("/")
def home():
    return {"message": "🔥 RAG API Running"}