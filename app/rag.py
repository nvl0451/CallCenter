from __future__ import annotations
import os, uuid, pathlib
from typing import List, Dict
from .config import settings

rag_available: bool = False
rag_unavailable_reason: str | None = None

# Initialize Chroma only if RAG enabled
if settings.enable_rag:
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        CHROMA_DIR = pathlib.Path(settings.chroma_dir)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.Client(ChromaSettings(is_persistent=True, persist_directory=str(CHROMA_DIR)))
        collection = client.get_or_create_collection("company_kb")
        rag_available = True
    except Exception as e:
        rag_available = False
        rag_unavailable_reason = f"RAG deps unavailable: {e}"
else:
    rag_available = False
    rag_unavailable_reason = "RAG disabled via ENABLE_RAG=0"

# Embedding backend: OpenAI if configured, otherwise SBERT
_embed_backend = None
_embed_note = None
if rag_available:
    try:
        if settings.rag_use_openai_embeddings and settings.openai_api_key:
            from openai import OpenAI  # type: ignore
            _embed_backend = ("openai", OpenAI(api_key=settings.openai_api_key))
            _embed_note = f"openai:{settings.openai_embed_model}"
        else:
            raise ImportError("OpenAI disabled or missing key")
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _embed_backend = ("sbert", SentenceTransformer("all-MiniLM-L6-v2"))
            _embed_note = "sbert:all-MiniLM-L6-v2"
        except Exception as e:
            rag_available = False
            rag_unavailable_reason = f"No embedding backend: {e}"


def _require_available():
    if not rag_available:
        raise RuntimeError(rag_unavailable_reason or "RAG unavailable")


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    _require_available()
    kind, backend = _embed_backend or (None, None)
    if kind == "openai":
        client = backend
        resp = client.embeddings.create(model=settings.openai_embed_model, input=texts)
        return [e.embedding for e in resp.data]
    elif kind == "sbert":
        return backend.encode(texts, normalize_embeddings=True).tolist()
    else:
        raise RuntimeError("No embedding backend configured")


def ingest_texts(texts: List[str], metadatas: List[Dict] | None = None):
    _require_available()
    metadatas = metadatas or [{} for _ in texts]
    all_chunks: List[str] = []
    all_meta: List[Dict] = []
    for doc_text, meta in zip(texts, metadatas):
        chunks = _chunk_text(doc_text, settings.rag_chunk_chars, settings.rag_chunk_overlap)
        for idx, ch in enumerate(chunks):
            m = dict(meta)
            m.update({"chunk": idx, "chunk_size": len(ch), "embed": _embed_note})
            all_chunks.append(ch)
            all_meta.append(m)
    ids = [str(uuid.uuid4()) for _ in all_chunks]
    embs = embed_texts(all_chunks)
    collection.add(ids=ids, embeddings=embs, documents=all_chunks, metadatas=all_meta)
    try:
        client.persist()
    except Exception:
        pass
    return ids


def search(query: str, top_k: int = 4):
    _require_available()
    q_emb = embed_texts([query])[0]
    # Do not request 'ids' in include (not a valid include key in recent Chroma)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    items = []
    for i in range(len(res["ids"][0])):
        d = res.get("distances", [[None]])[0][i]
        # Normalize distance to a similarity-like score in [0,1]
        score: float
        try:
            dv = float(d) if d is not None else None
        except Exception:
            dv = None
        if dv is None:
            score = 0.0
        elif 0.0 <= dv <= 1.0:
            score = 1.0 - dv
        elif 1.0 < dv <= 2.0:
            score = max(0.0, 1.0 - (dv / 2.0))
        else:
            score = max(0.0, 1.0 / (1.0 + dv)) if dv is not None else 0.0
        items.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res.get("metadatas", [[{}]])[0][i] or {},
            "score": float(score),
        })
    return items

def delete_by_doc_id(doc_id: int) -> int:
    _require_available()
    try:
        collection.delete(where={"doc_id": str(int(doc_id))})
        try:
            client.persist()
        except Exception:
            pass
        return 1
    except Exception:
        return 0

def chunk_count(text: str) -> int:
    chunks = _chunk_text(text or "", settings.rag_chunk_chars, settings.rag_chunk_overlap)
    return len(chunks)

def get_embed_note() -> str:
    return _embed_note or ""
