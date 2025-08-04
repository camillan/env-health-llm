# 📁 embeddings/search.py

import faiss
import numpy as np
import pickle
import json
from embeddings.embedder import embed  # ✅ use centralized embed function

# === Config ===
INDEX_DIR = "embeddings/faiss_store"
TOP_K = 3

# === Load index + metadata ===
print("🔍 Loading index and metadata...")
index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search(query: str, k: int = TOP_K):
    query_embedding = embed([query], use_multiprocessing=False)  # ✅ avoid segfaults
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        doc = metadata[idx]
        results.append({
            "title": doc["title"],
            "abstract": doc["abstract"],
            "distance": float(dist)
        })

    return results

# === Optional CLI loop ===
if __name__ == "__main__":
    while True:
        query = input("\n🧠 Ask a question (or 'exit'): ")
        if query.lower() in {"exit", "quit"}:
            break
        results = search(query)
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r['title']} (score: {r['distance']:.4f})\n{r['abstract']}")
