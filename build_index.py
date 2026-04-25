import os
import pickle
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_FOLDER = "data"

def get_embedding(text):
    url = "https://api.groq.com/openai/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "text-embedding-3-small",
        "input": text
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    return result["data"][0]["embedding"]


def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


print("🔄 Building embeddings...")

all_chunks = []
all_embeddings = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".txt"):
        path = os.path.join(DATA_FOLDER, file)

        print(f"📄 Processing {file}")

        text = process_txt(path)
        chunks = chunk_text(text)

        for chunk in chunks:
            emb = get_embedding(chunk)
            all_chunks.append(chunk)
            all_embeddings.append(emb)

# SAVE
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

with open("embeddings.pkl", "wb") as f:
    pickle.dump(all_embeddings, f)

print("✅ DONE — embeddings ready")