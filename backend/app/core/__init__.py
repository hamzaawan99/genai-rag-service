"""Core functionality for the RAG service."""

from .vector_store.base import VectorStore
from .vector_store.faiss_store import FAISSStore

__all__ = ["VectorStore", "FAISSStore"]
