import os
import weaviate
from weaviate.classes.init import Auth
from typing import List, Dict, Any, Optional
import uuid

class WeaviateClient:
    def __init__(self):
        from app.core.config import settings
        
        self.client = weaviate.connect_to_local(
            host=settings.WEAVIATE_HOST,
            port=int(settings.WEAVIATE_PORT),
            grpc_port=int(settings.WEAVIATE_GRPC_PORT),
            auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY)
        )
    
    def create_class(self, class_name: str) -> None:
        """Create a new class in Weaviate."""
        class_obj = {
            "class": class_name,
            "vectorizer": "none",  # We'll set vectors manually
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                },
                {
                    "name": "document_metadata",
                    "dataType": ["text"],
                },
                {
                    "name": "document_id",
                    "dataType": ["int"],
                },
            ]
        }
        
        self.client.schema.create_class(class_obj)
    
    def delete_class(self, class_name: str) -> None:
        """Delete a class from Weaviate."""
        self.client.schema.delete_class(class_name)
    
    def add_document(
        self, 
        class_name: str, 
        content: str, 
        vector: List[float], 
        metadata: Optional[str] = None,
        document_id: Optional[int] = None
    ) -> str:
        """Add a document to Weaviate with its vector embedding."""
        data_object = {
            "content": content,
            "document_metadata": metadata,
            "document_id": document_id
        }
        
        # Generate a UUID for the object
        uuid_str = str(uuid.uuid4())
        
        # Add the object with its vector
        self.client.data_object.create(
            data_object=data_object,
            class_name=class_name,
            vector=vector,
            uuid=uuid_str
        )
        
        return uuid_str
    
    def delete_document(self, class_name: str, uuid_str: str) -> None:
        """Delete a document from Weaviate."""
        self.client.data_object.delete(
            class_name=class_name,
            uuid=uuid_str
        )
    
    def search_similar(
        self, 
        class_name: str, 
        vector: List[float], 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        result = (
            self.client.query
            .get(class_name, ["content", "document_metadata", "document_id"])
            .with_near_vector({
                "vector": vector,
                "certainty": 0.7
            })
            .with_limit(limit)
            .do()
        )
        
        return result.get("data", {}).get("Get", {}).get(class_name, [])
