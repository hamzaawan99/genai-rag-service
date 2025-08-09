from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import numpy as np
from loguru import logger

from app.models.embedding import EmbeddingConfig, DocumentChunk, TextChunk
from app.config.settings import settings

class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding service."""
        self.config = config or EmbeddingConfig()
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector as a list of floats
        """
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    async def embed_documents(self, chunks: List[TextChunk]) -> List[DocumentChunk]:
        """Embed a list of text chunks.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List of document chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract texts for batching
        texts = [chunk.text for chunk in chunks]
        
        # Get embeddings
        embeddings = await self.get_embeddings(texts)
        
        # Create document chunks with embeddings
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            doc_chunk = DocumentChunk(
                **chunk.dict(),
                embedding=embedding
            )
            results.append(doc_chunk)
        
        return results

class SentenceTransformerService(EmbeddingService):
    """Sentence Transformers embedding service."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the Sentence Transformers service."""
        super().__init__(config)
        self._model = None
    
    @property
    def model(self):
        """Lazy load the Sentence Transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
        return self._model
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            # Encode the texts
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_tensor=False
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the OpenAI embedding service."""
        super().__init__(config)
        self._client = None
        
        # Override model name if using Azure
        if settings.AZURE_OPENAI_API_KEY:
            self.config.model_name = settings.AZURE_OPENAI_DEPLOYMENT or "text-embedding-ada-002"
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            if settings.AZURE_OPENAI_API_KEY:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
                )
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            # Prepare batches
            batch_size = self.config.batch_size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Call the API
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.config.model_name
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise

def get_embedding_service(config: Optional[EmbeddingConfig] = None) -> EmbeddingService:
    """Factory function to get the appropriate embedding service.
    
    Args:
        config: Optional configuration for the embedding service
        
    Returns:
        An instance of an embedding service
    """
    config = config or EmbeddingConfig()
    
    # Check if we should use OpenAI
    use_openai = (
        settings.OPENAI_API_KEY or 
        (settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT)
    )
    
    if use_openai and config.model_name != "sentence-transformers/all-MiniLM-L6-v2":
        return OpenAIEmbeddingService(config)
    
    # Default to Sentence Transformers
    return SentenceTransformerService(config)
