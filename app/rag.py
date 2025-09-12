from __future__ import annotations
import os, uuid, pathlib
from typing import List, Dict
from .config import settings

rag_available: bool = False
rag_unavailable_reason: str | None = None

# Only attempt to import heavy deps if the feature is enabled
if settings.enable_rag:
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import SentenceTransformer

        CHROMA_DIR = pathlib.Path(settings.chroma_dir)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        client = chromadb.Client(ChromaSettings(is_persistent=True, persist_directory=str(CHROMA_DIR)))
        collection = client.get_or_create_collection("company_kb")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        rag_available = True
    except Exception as e:
        rag_available = False
        rag_unavailable_reason = f"RAG deps unavailable: {e}"
else:
    rag_available = False
    rag_unavailable_reason = "RAG disabled via ENABLE_RAG=0"

def _require_available():
    if not rag_available:
        raise RuntimeError(rag_unavailable_reason or "RAG unavailable")

def embed_texts(texts: List[str]) -> List[List[float]]:
    _require_available()
    return embedder.encode(texts, normalize_embeddings=True).tolist()

def ingest_texts(texts: List[str], metadatas: List[Dict] | None = None):
    _require_available()
    ids = [str(uuid.uuid4()) for _ in texts]
    embs = embed_texts(texts)
    collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metadatas)
    client.persist()
    return ids

def search(query: str, top_k: int = 4):
    _require_available()
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
    items = []
    for i in range(len(res["ids"][0])):
        items.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res.get("metadatas", [[{}]])[0][i] or {},
            "score": float(1.0 - (res.get("distances", [[1.0]])[0][i] or 1.0)),
        })
    return items
