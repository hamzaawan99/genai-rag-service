import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, BinaryIO
from loguru import logger
import magic
from pypdf import PdfReader
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from datetime import datetime
import uuid

from app.models.document import Document, DocumentStatus, DocumentSource, DocumentMetadata
from app.models.embedding import TextChunk, DocumentChunk, ChunkingConfig, ChunkingStrategy
from app.config.settings import settings

class DocumentProcessor:
    """Service for processing uploaded documents."""
    
    SUPPORTED_MIME_TYPES = {
        'application/pdf': 'pdf',
        'text/plain': 'txt',
        'text/markdown': 'md',
        'text/csv': 'csv',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    }
    
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        """Initialize the document processor.
        
        Args:
            chunking_config: Configuration for chunking documents
        """
        self.chunking_config = chunking_config or ChunkingConfig()
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_upload(
        self,
        file: BinaryIO,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Tuple[Document, List[TextChunk]]:
        """Process an uploaded file.
        
        Args:
            file: File-like object containing the uploaded file
            filename: Original filename
            metadata: Additional metadata for the document
            user_id: ID of the user who uploaded the file
            
        Returns:
            A tuple of (document, chunks) where document is the processed document
            and chunks are the extracted text chunks
        """
        # Read file content
        file_content = file.read()
        
        # Get MIME type
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(file_content)
        
        # Validate MIME type
        if mime_type not in self.SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported file type: {mime_type}")
        
        # Generate a unique filename
        file_ext = self.SUPPORTED_MIME_TYPES[mime_type]
        file_id = str(uuid.uuid4())
        file_path = self.upload_dir / f"{file_id}.{file_ext}"
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Get file stats
        file_size = os.path.getsize(file_path)
        
        # Create document metadata
        doc_metadata = DocumentMetadata(
            source=DocumentSource.UPLOAD,
            content_type=mime_type,
            size=file_size,
            custom_metadata=metadata or {}
        )
        
        # Create document
        document = Document(
            id=file_id,
            title=filename,
            description=f"Uploaded {filename}",
            status=DocumentStatus.PROCESSING,
            metadata=doc_metadata,
            tags=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Extract text and chunk it
        chunks = await self.extract_and_chunk(document, file_path)
        
        return document, chunks
    
    async def extract_and_chunk(
        self,
        document: Document,
        file_path: Path
    ) -> List[TextChunk]:
        """Extract text from a document and split it into chunks.
        
        Args:
            document: The document to process
            file_path: Path to the document file
            
        Returns:
            List of text chunks
        """
        try:
            # Update document status
            document.status = DocumentStatus.PROCESSING
            
            # Extract text based on file type
            if str(file_path).lower().endswith('.pdf'):
                text = await self._extract_text_from_pdf(file_path)
            else:
                text = await self._extract_text_generic(file_path)
            
            # Clean up the text
            text = self._clean_text(text)
            
            # Split into chunks
            chunks = self._chunk_text(text, document.id)
            
            # Update document metadata
            document.metadata.page_count = self._get_page_count(file_path)
            document.status = DocumentStatus.PROCESSED
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            document.status = DocumentStatus.FAILED
            raise
    
    async def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    async def _extract_text_generic(self, file_path: Path) -> str:
        """Extract text from a generic file using Unstructured."""
        try:
            # Use Unstructured to extract text
            elements = partition(str(file_path))
            
            # Combine all text elements
            text_parts = [str(el) for el in elements]
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _chunk_text(self, text: str, document_id: str) -> List[TextChunk]:
        """Split text into chunks."""
        if not text.strip():
            return []
        
        chunks = []
        
        if self.chunking_config.strategy == ChunkingStrategy.SIMPLE:
            # Simple fixed-size chunking
            chunk_size = self.chunking_config.chunk_size
            overlap = self.chunking_config.chunk_overlap
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_id=f"{document_id}_{len(chunks)}",
                    document_id=document_id,
                    chunk_index=len(chunks),
                    metadata={
                        "chunking_strategy": "simple",
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap
                    }
                )
                chunks.append(chunk)
                
        elif self.chunking_config.strategy == ChunkingStrategy.RECURSIVE:
            # Recursive chunking using Unstructured
            elements = [{"text": text}]
            chunked_elements = chunk_by_title(
                elements,
                max_characters=self.chunking_config.chunk_size,
                overlap=self.chunking_config.chunk_overlap,
                overlap_all=True
            )
            
            for i, chunk_el in enumerate(chunked_elements):
                chunk = TextChunk(
                    text=chunk_el.text,
                    chunk_id=f"{document_id}_{i}",
                    document_id=document_id,
                    chunk_index=i,
                    metadata={
                        "chunking_strategy": "recursive",
                        "element_type": chunk_el.metadata.to_dict() if hasattr(chunk_el, 'metadata') else {}
                    }
                )
                chunks.append(chunk)
                
        # Add other chunking strategies as needed...
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean up the extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters (except newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def _get_page_count(self, file_path: Path) -> Optional[int]:
        """Get the number of pages in a document."""
        try:
            if str(file_path).lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    return len(reader.pages)
        except Exception as e:
            logger.warning(f"Could not determine page count for {file_path}: {e}")
        
        return None
