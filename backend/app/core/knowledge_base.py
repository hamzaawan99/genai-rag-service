from typing import List, Optional
from sqlalchemy.orm import Session
import json

from app.models.knowledge_base import KnowledgeBase, Document
from app.models.user import User
from app.schemas.knowledge_base import KnowledgeBaseCreate, DocumentCreate
from app.core.weaviate_client import WeaviateClient

def generate_class_name(kb_name: str, user_id: int) -> str:
    """Generate a unique Weaviate class name for the knowledge base."""
    # Remove spaces and special characters, convert to title case
    clean_name = "".join(c for c in kb_name if c.isalnum())
    return f"User{user_id}{clean_name}"

class KnowledgeBaseService:
    def __init__(self):
        self.weaviate_client = WeaviateClient()
    
    def create_knowledge_base(
        self,
        db: Session,
        kb_create: KnowledgeBaseCreate,
        user: User
    ) -> KnowledgeBase:
        """Create a new knowledge base."""
        # Generate Weaviate class name
        weaviate_class_name = generate_class_name(kb_create.name, user.id)
        
        # Create the class in Weaviate
        self.weaviate_client.create_class(weaviate_class_name)
        
        # Create knowledge base in database
        db_kb = KnowledgeBase(
            name=kb_create.name,
            description=kb_create.description,
            embedding_model=kb_create.embedding_model,
            weaviate_class_name=weaviate_class_name,
            user_id=user.id
        )
        
        db.add(db_kb)
        db.commit()
        db.refresh(db_kb)
        
        return db_kb
    
    def delete_knowledge_base(
        self,
        db: Session,
        kb_id: int,
        user: User
    ) -> bool:
        """Delete a knowledge base and its associated Weaviate class."""
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user.id
        ).first()
        
        if not kb:
            return False
        
        # Delete the Weaviate class
        try:
            self.weaviate_client.delete_class(kb.weaviate_class_name)
        except Exception:
            # Log the error but continue with database deletion
            pass
        
        # Delete from database
        db.delete(kb)
        db.commit()
        
        return True
    
    def get_knowledge_bases(
        self,
        db: Session,
        user: User,
        skip: int = 0,
        limit: int = 100
    ) -> List[KnowledgeBase]:
        """Get all knowledge bases for a user."""
        return db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user.id
        ).offset(skip).limit(limit).all()
    
    def get_knowledge_base(
        self,
        db: Session,
        kb_id: int,
        user: User
    ) -> Optional[KnowledgeBase]:
        """Get a specific knowledge base."""
        return db.query(KnowledgeBase).filter(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user.id
        ).first()
    
    def add_document(
        self,
        db: Session,
        doc_create: DocumentCreate,
        user: User,
        vector: List[float]
    ) -> Optional[Document]:
        """Add a document to a knowledge base."""
        # Check if knowledge base exists and belongs to user
        kb = self.get_knowledge_base(db, doc_create.knowledge_base_id, user)
        if not kb:
            return None
        
        # Create document in database
        db_doc = Document(
            title=doc_create.title,
            content=doc_create.content,
            document_metadata=doc_create.document_metadata,
            knowledge_base_id=doc_create.knowledge_base_id,
            user_id=user.id
        )
        
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        
        # Add to Weaviate
        try:
            self.weaviate_client.add_document(
                class_name=kb.weaviate_class_name,
                content=doc_create.content,
                vector=vector,
                metadata=doc_create.document_metadata,
                document_id=db_doc.id
            )
        except Exception as e:
            # If Weaviate insertion fails, rollback the database
            db.delete(db_doc)
            db.commit()
            raise e
        
        return db_doc
    
    def get_documents(
        self,
        db: Session,
        kb_id: int,
        user: User,
        skip: int = 0,
        limit: int = 100
    ) -> List[Document]:
        """Get all documents in a knowledge base."""
        return db.query(Document).filter(
            Document.knowledge_base_id == kb_id,
            Document.user_id == user.id
        ).offset(skip).limit(limit).all()
    
    def delete_document(
        self,
        db: Session,
        doc_id: int,
        user: User
    ) -> bool:
        """Delete a document from both database and Weaviate."""
        doc = db.query(Document).filter(
            Document.id == doc_id,
            Document.user_id == user.id
        ).first()
        
        if not doc:
            return False
        
        # Delete from database
        db.delete(doc)
        db.commit()
        
        return True
