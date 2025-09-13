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
    latencies: Optional[dict] = None

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

# -------- Agent Schemas --------
class AgentMessageRequest(BaseModel):
    session_id: Optional[str] = None
    text: str

class AgentMessageResponse(BaseModel):
    session_id: str
    tools_used: List[str]
    reply: str
    sources: Optional[List[dict]] = None
    latencies: Optional[dict] = None
    llm_fallback: bool = False

# -------- Admin Schemas --------
class AdminClassIn(BaseModel):
    name: str
    synonyms: Optional[List[str]] = None
    stems: Optional[List[str]] = None
    system_prompt: str = ""
    priority: int = 0
    active: int = 1

class AdminClassOut(BaseModel):
    id: int
    name: str
    synonyms_json: str
    stems_json: str
    system_prompt: str
    priority: int
    active: int
    updated_at: float

class AdminVisionIn(BaseModel):
    name: str
    synonyms: Optional[List[str]] = None
    templates: Optional[List[str]] = None
    priority: int = 0
    active: int = 1

class AdminVisionOut(BaseModel):
    id: int
    name: str
    synonyms_json: str
    templates_json: str
    priority: int
    active: int
    updated_at: float

class AdminRagInlineIn(BaseModel):
    title: str
    content_text: str
    index_now: bool = True

class AdminRagDocOut(BaseModel):
    id: int
    title: str
    kind: str
    rel_path: Optional[str] = None
    bytes: int
    mime: Optional[str] = None
    source: Optional[str] = None
    active: int
    created_at: float
    updated_at: float
    indexed_at: Optional[float] = None
    embed_model: Optional[str] = None
    chunks_count: int
    dirty: int
