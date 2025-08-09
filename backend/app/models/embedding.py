from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np

class ChunkingStrategy(str, Enum):
    """Strategies for splitting documents into chunks."""
    SIMPLE = "simple"  # Fixed-size chunks with overlap
    SENTENCE = "sentence"  # Split on sentence boundaries
    PARAGRAPH = "paragraph"  # Split on paragraph boundaries
    RECURSIVE = "recursive"  # Recursive character-based splitting

class TextChunk(BaseModel):
    """A chunk of text with associated metadata."""
    text: str = Field(..., description="The text content of the chunk")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the source document")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the chunk"
    )

    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
        }

class DocumentChunk(TextChunk):
    """A chunk of a document with embedding."""
    embedding: Optional[List[float]] = Field(
        None,
        description="Vector embedding of the chunk text"
    )
    
    @validator('embedding', pre=True)
    def validate_embedding(cls, v):
        if v is not None and not isinstance(v, (list, np.ndarray)):
            raise ValueError("Embedding must be a list or numpy array")
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return v

class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = Field(
        ChunkingStrategy.SIMPLE,
        description="Strategy for splitting documents into chunks"
    )
    chunk_size: int = Field(
        1000,
        gt=0,
        le=10000,
        description="Maximum number of characters per chunk"
    )
    chunk_overlap: int = Field(
        200,
        ge=0,
        description="Number of characters to overlap between chunks"
    )
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", " ", ""],
        description="Separators to use for splitting text into chunks"
    )

class EmbeddingConfig(BaseModel):
    """Configuration for generating embeddings."""
    model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the embedding model to use"
    )
    batch_size: int = Field(
        32,
        gt=0,
        le=256,
        description="Number of chunks to process in a single batch"
    )
    device: str = Field(
        "cpu",
        description="Device to run the embedding model on (cpu, cuda, mps)"
    )
    normalize_embeddings: bool = Field(
        True,
        description="Whether to normalize the embeddings to unit length"
    )

class QueryResult(BaseModel):
    """Result of a similarity search query."""
    chunk: DocumentChunk
    score: float = Field(..., description="Similarity score (higher is more similar)")

class SearchResponse(BaseModel):
    """Response from a similarity search."""
    query: str
    results: List[QueryResult]
    total_results: int
    model: str
    search_duration: float  # in seconds
