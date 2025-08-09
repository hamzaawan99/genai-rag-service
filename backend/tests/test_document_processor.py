"""Tests for the document processor service."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest
from fastapi import UploadFile

from app.models.document import DocumentStatus
from app.services.document_processor import DocumentProcessor
from app.models.embedding import DocumentChunk, ChunkingConfig


@pytest.fixture
temp_upload_dir():
    """Create a temporary directory for uploads."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pdf_content():
    """Return sample PDF content for testing."""
    # This is a minimal PDF file
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 700 Td (Hello, World!) Tj ET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000114 00000 n \n0000000212 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n303\n%%EOF"
    )


@pytest.fixture
def sample_text_content():
    """Return sample text content for testing."""
    return "This is a sample text file.\nIt contains multiple lines.\n"


class TestDocumentProcessor:
    """Tests for the DocumentProcessor class."""
    
    @pytest.fixture(autouse=True)
    def setup(self, temp_upload_dir):
        """Set up test environment."""
        self.upload_dir = temp_upload_dir
        self.processor = DocumentProcessor(upload_dir=str(temp_upload_dir))
        self.chunking_config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )
    
    def create_upload_file(self, content: bytes, filename: str) -> UploadFile:
        """Helper to create an UploadFile for testing."""
        return UploadFile(
            filename=filename,
            file=MagicMock(read=MagicMock(return_value=content))
        )
    
    def test_process_pdf_document(self, sample_pdf_content):
        """Test processing a PDF document."""
        # Arrange
        upload_file = self.create_upload_file(sample_pdf_content, "test.pdf")
        
        # Mock the PyPDF loader
        mock_document = MagicMock()
        mock_document.page_content = "Page 1 content\n\nPage 2 content"
        mock_document.metadata = {"source": "test.pdf", "page": 0}
        
        # Act
        with patch("app.services.document_processor.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = [mock_document]
            
            result = self.processor.process_document(
                file=upload_file,
                chunking_config=self.chunking_config,
                metadata={"source": "test"}
            )
        
        # Assert
        assert result.status == DocumentStatus.PROCESSED
        assert result.filename == "test.pdf"
        assert result.metadata["file_type"] == "application/pdf"
        assert result.metadata["pages"] == 1
        assert len(result.chunks) > 0
        
        # Check that the file was saved
        saved_file = self.upload_dir / result.document_id / "test.pdf"
        assert saved_file.exists()
    
    def test_process_text_document(self, sample_text_content):
        """Test processing a text document."""
        # Arrange
        upload_file = self.create_upload_file(
            sample_text_content.encode("utf-8"), 
            "test.txt"
        )
        
        # Act
        result = self.processor.process_document(
            file=upload_file,
            chunking_config=self.chunking_config,
            metadata={"source": "test"}
        )
        
        # Assert
        assert result.status == DocumentStatus.PROCESSED
        assert result.filename == "test.txt"
        assert result.metadata["file_type"] == "text/plain"
        assert len(result.chunks) > 0
    
    def test_chunk_text(self):
        """Test the text chunking functionality."""
        # Arrange
        text = """This is a sample text. It contains multiple sentences. 
        We want to test if the chunking works correctly."""
        
        # Act
        chunks = self.processor._chunk_text(
            text=text,
            chunking_config=self.chunking_config,
            metadata={"source": "test"}
        )
        
        # Assert
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.metadata["source"] == "test" for chunk in chunks)
    
    def test_extract_text_from_file_pdf(self, sample_pdf_content, tmp_path):
        """Test text extraction from PDF files."""
        # Arrange
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(sample_pdf_content)
        
        # Mock the PyPDF loader
        mock_document = MagicMock()
        mock_document.page_content = "Extracted text from PDF"
        mock_document.metadata = {"source": str(pdf_path), "page": 0}
        
        # Act
        with patch("app.services.document_processor.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = [mock_document]
            
            text, metadata = self.processor._extract_text_from_file(
                file_path=pdf_path,
                file_type="application/pdf"
            )
        
        # Assert
        assert text == "Extracted text from PDF"
        assert metadata["pages"] == 1
        assert metadata["file_type"] == "application/pdf"
    
    def test_extract_text_from_txt(self, sample_text_content, tmp_path):
        """Test text extraction from text files."""
        # Arrange
        txt_path = tmp_path / "test.txt"
        txt_path.write_text(sample_text_content)
        
        # Act
        text, metadata = self.processor._extract_text_from_file(
            file_path=txt_path,
            file_type="text/plain"
        )
        
        # Assert
        assert text == sample_text_content
        assert metadata["file_type"] == "text/plain"
    
    def test_unsupported_file_type(self):
        """Test processing an unsupported file type."""
        # Arrange
        upload_file = self.create_upload_file(b"test content", "test.unsupported")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.processor.process_document(
                file=upload_file,
                chunking_config=self.chunking_config,
                metadata={"source": "test"}
            )
    
    def test_save_upload_file(self, sample_text_content):
        """Test saving an uploaded file."""
        # Arrange
        upload_file = self.create_upload_file(
            sample_text_content.encode("utf-8"), 
            "test_save.txt"
        )
        document_id = "test-doc-123"
        
        # Act
        saved_path = self.processor._save_upload_file(
            file=upload_file,
            document_id=document_id
        )
        
        # Assert
        assert saved_path.exists()
        assert saved_path.name == "test_save.txt"
        assert saved_path.parent.name == document_id
        assert saved_path.read_text() == sample_text_content
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Arrange
        dirty_text = "  This  is  a  test.  \n\nMultiple  spaces  and  newlines.  "
        expected = "This is a test.\n\nMultiple spaces and newlines."
        
        # Act
        cleaned = self.processor._clean_text(dirty_text)
        
        # Assert
        assert cleaned == expected
