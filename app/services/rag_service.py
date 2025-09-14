from __future__ import annotations

from .backends.rag_backend import (
    rag_available,
    rag_unavailable_reason,
    ingest_texts,
    search,
    delete_by_doc_id,
    reset_index,
    chunk_count,
    get_embed_note,
)

__all__ = [
    "rag_available",
    "rag_unavailable_reason",
    "ingest_texts",
    "search",
    "delete_by_doc_id",
    "reset_index",
    "chunk_count",
    "get_embed_note",
]
