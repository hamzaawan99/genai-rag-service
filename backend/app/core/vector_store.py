from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

class VectorDBClient(ABC):
    """Abstract base class for vector database operations."""
    
    @abstractmethod
    def create_collection(self, name: str) -> None:
        """Create a new collection/class."""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection/class."""
        pass
    
    @abstractmethod
    def add_document(
        self,
        collection_name: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict] = None,
        vector: Optional[List[float]] = None
    ) -> str:
        """Add a document to the vector database."""
        pass
    
    @abstractmethod
    def delete_document(self, collection_name: str, document_id: str) -> None:
        """Delete a document from the vector database."""
        pass
    
    @abstractmethod
    def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass

class ChromaDBClient(VectorDBClient):
    def __init__(self):
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path="./data/chromadb",
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
    
    def create_collection(self, name: str) -> None:
        """Create a new collection in ChromaDB."""
        self.client.create_collection(name=name)
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection from ChromaDB."""
        self.client.delete_collection(name=name)
    
    def add_document(
        self,
        collection_name: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict] = None,
        vector: Optional[List[float]] = None
    ) -> str:
        """Add a document to ChromaDB."""
        collection = self.client.get_collection(name=collection_name)
        
        # If vector is provided, use it, otherwise ChromaDB will generate embeddings
        if vector:
            collection.add(
                ids=[document_id],
                documents=[content],
                metadatas=[metadata] if metadata else None,
                embeddings=[vector]
            )
        else:
            collection.add(
                ids=[document_id],
                documents=[content],
                metadatas=[metadata] if metadata else None
            )
        
        return document_id
    
    def delete_document(self, collection_name: str, document_id: str) -> None:
        """Delete a document from ChromaDB."""
        collection = self.client.get_collection(name=collection_name)
        collection.delete(ids=[document_id])
    
    def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        collection = self.client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit
        )
        
        # Format results to match the expected structure
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else None,
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results

class VectorDBFactory:
    """Factory class to create vector database clients."""
    
    @staticmethod
    def create_client(db_type: str) -> VectorDBClient:
        if db_type == "chromadb":
            return ChromaDBClient()
        elif db_type == "weaviate":
            raise NotImplementedError("Weaviate support is not available yet")
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
