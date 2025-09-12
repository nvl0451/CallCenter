from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

class StartDialogRequest(BaseModel):
    customer_id: Optional[str] = None
    metadata: Optional[dict] = None

class StartDialogResponse(BaseModel):
    session_id: str

class MessageRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)

class MessageResponse(BaseModel):
    type: str
    confidence: float
    reply: str
    cost_estimate_usd: float
    latency_ms: int

class IngestRequest(BaseModel):
    documents: Optional[List[str]] = None  # сырые тексты

class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

class VisionResponse(BaseModel):
    category: str
    confidence: float
    logits: List[float]
    labels: List[str]
