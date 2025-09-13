from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    knowledge_base_id: int
    message: str
    max_context_docs: Optional[int] = Field(default=5, gt=0, le=10)

class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    message: str
    relevant_docs: List[dict]
    conversation_id: str
