from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from datetime import datetime

class KnowledgeBaseBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    embedding_model: str = Field(..., min_length=1)
    vector_db: Literal["chromadb", "weaviate"] = Field(default="chromadb")

class KnowledgeBaseCreate(KnowledgeBaseBase):
    pass

class KnowledgeBase(KnowledgeBaseBase):
    id: int
    collection_name: str
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    document_metadata: Optional[Dict] = None

class DocumentCreate(DocumentBase):
    knowledge_base_id: int

class Document(DocumentBase):
    id: int
    knowledge_base_id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class KnowledgeBaseWithDocuments(KnowledgeBase):
    documents: List[Document]
