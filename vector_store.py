import faiss
import numpy as np

index = None
stored_chunks = []

def create_index(embeddings, chunks):
    global index, stored_chunks
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    stored_chunks = chunks

def search(query_embedding, k=3):
    D, I = index.search(np.array([query_embedding]), k)
    return [stored_chunks[i] for i in I[0]]