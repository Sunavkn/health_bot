from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    question: str
    language: Optional[str] = "en"
    context_notes: Optional[str] = None
    domain_hint: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    disclaimer: str
    confidence: str
    warnings: List[str] = []
