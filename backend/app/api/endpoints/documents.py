from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
from loguru import logger

from app.models.document import Document, DocumentInDB, DocumentStatus
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import get_embedding_service
from app.core.vector_store.faiss_store import FAISSStore
from app.config.settings import settings

router = APIRouter()

# Initialize services
vector_store = FAISSStore()
embedding_service = get_embedding_service()
document_processor = DocumentProcessor()

@router.post("/upload", response_model=DocumentInDB, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[dict] = None,
    user_id: Optional[str] = None
):
    """
    Upload a document for processing and indexing.
    
    - **file**: The document file to upload (PDF, DOCX, TXT, etc.)
    - **metadata**: Optional metadata for the document
    - **user_id**: Optional user ID for access control
    
    Returns the created document with processing status.
    """
    try:
        # Process the uploaded file
        document, chunks = await document_processor.process_upload(
            file=file.file,
            filename=file.filename,
            metadata=metadata,
            user_id=user_id
        )
        
        # Generate embeddings for the chunks
        document_chunks = await embedding_service.embed_documents(chunks)
        
        # Add to vector store
        chunk_ids = await vector_store.add_documents(document_chunks)
        
        # Update document with chunk information
        document.metadata.custom_metadata["chunk_ids"] = chunk_ids
        document.status = DocumentStatus.PROCESSED
        
        # TODO: Save document metadata to database
        
        return document
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentInDB)
async def get_document(document_id: str):
    """
    Get a document by ID.
    
    - **document_id**: The ID of the document to retrieve
    """
    # TODO: Implement document retrieval from database
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document retrieval not implemented yet"
    )

@router.get("/", response_model=List[DocumentInDB])
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    status: Optional[DocumentStatus] = None
):
    """
    List all documents with optional filtering.
    
    - **skip**: Number of documents to skip
    - **limit**: Maximum number of documents to return
    - **status**: Filter by document status
    """
    # TODO: Implement document listing from database with pagination and filtering
    return []

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    """
    Delete a document and its associated chunks.
    
    - **document_id**: The ID of the document to delete
    """
    try:
        # Delete from vector store
        success = await vector_store.delete(document_ids=[document_id])
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
        # TODO: Delete document from database
        
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content=None
        )
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )
