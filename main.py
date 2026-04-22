import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# TEST ROUTE (IMPORTANT)
# ========================
@app.get("/")
def home():
    return {"message": "🔥 RAG API Running"}

# ========================
# RESET ROUTE
# ========================
chat_history = []

@app.get("/reset")
def reset():
    global chat_history
    chat_history = []
    return {"message": "Memory cleared"}

# ========================
# UPLOAD ROUTE (SAFE VERSION)
# ========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        return {
            "filename": file.filename,
            "size": len(contents),
            "message": "Uploaded successfully"
        }
    except Exception as e:
        return {"error": str(e)}

# ========================
# START SERVER (CRITICAL FOR RENDER)
# ========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)
