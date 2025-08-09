from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from app.models.embedding import DocumentChunk, SearchResponse

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_documents(
        self,
        chunks: List[DocumentChunk],
        **kwargs
    ) -> List[str]:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of document chunk IDs that were added
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResponse:
        """Search for similar document chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional filters to apply to the search
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            SearchResponse containing matching chunks and scores
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        chunk_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """Delete document chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            document_ids: List of document IDs whose chunks should be deleted
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary containing statistics
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all data from the vector store.
        
        Returns:
            True if the operation was successful
        """
        pass
    
    @staticmethod
    def _validate_embeddings(embeddings: List[List[float]]) -> np.ndarray:
        """Validate and convert embeddings to numpy array.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Numpy array of shape (n_embeddings, embedding_dim)
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
            
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        if len(embeddings_np.shape) != 2:
            raise ValueError(f"Expected 2D array, got {embeddings_np.shape}")
            
        return embeddings_np
