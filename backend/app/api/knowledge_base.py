from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.knowledge_base import KnowledgeBaseService
from app.schemas.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseCreate,
    Document,
    DocumentCreate,
    KnowledgeBaseWithDocuments
)
from app.models.user import User
from app.api.auth import get_current_user

router = APIRouter()
kb_service = KnowledgeBaseService()

@router.post("/", response_model=KnowledgeBase)
async def create_knowledge_base(
    kb_create: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new knowledge base."""
    try:
        return kb_service.create_knowledge_base(db, kb_create, current_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/", response_model=List[KnowledgeBase])
async def list_knowledge_bases(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all knowledge bases for the current user."""
    return kb_service.get_knowledge_bases(db, current_user, skip, limit)

@router.get("/{kb_id}", response_model=KnowledgeBaseWithDocuments)
async def get_knowledge_base(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific knowledge base with its documents."""
    kb = kb_service.get_knowledge_base(db, kb_id, current_user)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
    return kb

@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a knowledge base."""
    if not kb_service.delete_knowledge_base(db, kb_id, current_user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
    return {"message": "Knowledge base deleted successfully"}

@router.post("/{kb_id}/documents", response_model=Document)
async def add_document(
    kb_id: int,
    doc_create: DocumentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a document to a knowledge base."""
    # TODO: Get vector embeddings from the embedding model
    # For now, using dummy embeddings
    dummy_vector = [0.0] * 1536  # Assuming 1536-dimensional embeddings
    
    try:
        doc = kb_service.add_document(db, doc_create, current_user, dummy_vector)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge base not found"
            )
        return doc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{kb_id}/documents", response_model=List[Document])
async def list_documents(
    kb_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all documents in a knowledge base."""
    # Verify knowledge base exists and belongs to user
    kb = kb_service.get_knowledge_base(db, kb_id, current_user)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
    
    return kb_service.get_documents(db, kb_id, current_user, skip, limit)

@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a document from a knowledge base."""
    # Verify knowledge base exists and belongs to user
    kb = kb_service.get_knowledge_base(db, kb_id, current_user)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
    
    if not kb_service.delete_document(db, doc_id, current_user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return {"message": "Document deleted successfully"}
