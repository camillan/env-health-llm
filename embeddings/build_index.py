# 📁 embeddings/build_index.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from pathlib import Path

# === Config ===
DATA_PATH = "data/env_health_abstracts.json"
INDEX_DIR = "embeddings/faiss_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === Setup ===
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
model = SentenceTransformer(EMBEDDING_MODEL)

# === Step 1: Load data ===
with open(DATA_PATH, "r") as f:
    documents = json.load(f)

texts = [doc["abstract"] for doc in documents]

# === Step 2: Generate embeddings ===
print("🔍 Embedding documents...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    use_multiprocessing=False  # 🔧 prevents segfaults on Mac
)

# === Step 3: Build FAISS index ===
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance
index.add(embeddings)

# === Step 4: Save FAISS index and metadata ===
faiss.write_index(index, f"{INDEX_DIR}/index.faiss")

# Save metadata (to map results back to original docs)
with open(f"{INDEX_DIR}/metadata.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"✅ Saved FAISS index with {len(documents)} documents to '{INDEX_DIR}/'")
