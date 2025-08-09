from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum

class DocumentSource(str, Enum):
    UPLOAD = "upload"
    WEB = "web"
    DATABASE = "database"

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"

class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: DocumentSource = Field(..., description="Source of the document")
    content_type: Optional[str] = Field(None, description="MIME type of the document")
    size: Optional[int] = Field(None, description="Size of the document in bytes")
    author: Optional[str] = Field(None, description="Author of the document")
    created_at: Optional[datetime] = Field(None, description="When the document was created")
    modified_at: Optional[datetime] = Field(None, description="When the document was last modified")
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
    language: Optional[str] = Field(None, description="Language of the document")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

class DocumentBase(BaseModel):
    """Base model for document operations."""
    title: str = Field(..., description="Title of the document")
    description: Optional[str] = Field(None, description="Description of the document")
    tags: List[str] = Field(default_factory=list, description="Tags for the document")

class DocumentCreate(DocumentBase):
    """Model for creating a new document."""
    pass

class DocumentUpdate(DocumentBase):
    """Model for updating an existing document."""
    title: Optional[str] = Field(None, description="New title for the document")
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class DocumentInDB(DocumentBase):
    """Model for a document stored in the database."""
    id: str = Field(..., description="Unique identifier for the document")
    status: DocumentStatus = Field(..., description="Processing status of the document")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the document was created in the system")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the document was last updated")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        json_schema_extra = {
            "example": {
                "id": "doc_12345",
                "title": "Sample Document",
                "description": "This is a sample document",
                "tags": ["sample", "test"],
                "status": "processed",
                "metadata": {
                    "source": "upload",
                    "content_type": "application/pdf",
                    "size": 1024,
                    "page_count": 5,
                    "language": "en"
                },
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z"
            }
        }

class Document(DocumentInDB):
    """Model for a document with content."""
    content: Optional[str] = Field(None, description="Extracted text content of the document")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
