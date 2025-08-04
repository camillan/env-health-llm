# 📁 embeddings/embedder.py

from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

def embed(texts, use_multiprocessing=False):
    return model.encode(
        texts,
        convert_to_numpy=True,
        use_multiprocessing=use_multiprocessing
    )
