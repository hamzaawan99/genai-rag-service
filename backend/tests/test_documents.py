"""Tests for document-related endpoints and functionality."""
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.models.document import DocumentStatus


def test_upload_document(
    test_app: TestClient, 
    sample_pdf_file: tuple[bytes, str],
    test_upload_dir: Path
) -> None:
    """Test uploading a document."""
    # Arrange
    pdf_content, filename = sample_pdf_file
    files = {"file": (filename, pdf_content, "application/pdf")}
    
    # Act
    with patch("app.services.document_processor.DocumentProcessor.process_document") as mock_process:
        mock_process.return_value = {
            "document_id": "test-doc-123",
            "filename": filename,
            "status": DocumentStatus.PROCESSED,
            "chunk_count": 5,
            "metadata": {"pages": 1, "file_type": "application/pdf"},
        }
        
        response = test_app.post(
            "/api/v1/documents/upload",
            files=files,
            data={"chunk_size": 1000, "chunk_overlap": 200}
        )
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["document_id"] == "test-doc-123"
    assert data["filename"] == filename
    assert data["status"] == DocumentStatus.PROCESSED
    assert data["chunk_count"] == 5
    
    # Verify the file was saved
    saved_file = test_upload_dir / "test-doc-123" / filename
    assert saved_file.exists()


def test_upload_unsupported_file_type(test_app: TestClient) -> None:
    """Test uploading an unsupported file type."""
    # Arrange
    files = {"file": ("test.unsupported", b"test content", "application/octet-stream")}
    
    # Act
    response = test_app.post(
        "/api/v1/documents/upload",
        files=files,
        data={"chunk_size": 1000, "chunk_overlap": 200}
    )
    
    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported file type" in response.json()["detail"]


def test_get_document(test_app: TestClient) -> None:
    """Test retrieving a document by ID."""
    # Arrange
    doc_id = "test-doc-123"
    
    # Mock the document service
    mock_doc = {
        "document_id": doc_id,
        "filename": "test.pdf",
        "status": DocumentStatus.PROCESSED,
        "chunk_count": 5,
        "metadata": {"pages": 1, "file_type": "application/pdf"},
        "created_at": "2023-01-01T00:00:00",
        "updated_at": "2023-01-01T00:00:00",
    }
    
    # Act
    with patch("app.api.endpoints.documents.document_service.get_document") as mock_get:
        mock_get.return_value = mock_doc
        response = test_app.get(f"/api/v1/documents/{doc_id}")
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["document_id"] == doc_id
    assert data["status"] == DocumentStatus.PROCESSED


def test_list_documents(test_app: TestClient) -> None:
    """Test listing all documents."""
    # Arrange
    mock_docs = [
        {
            "document_id": "doc-1",
            "filename": "doc1.pdf",
            "status": DocumentStatus.PROCESSED,
            "chunk_count": 3,
            "created_at": "2023-01-01T00:00:00",
        },
        {
            "document_id": "doc-2",
            "filename": "doc2.pdf",
            "status": DocumentStatus.PROCESSING,
            "chunk_count": 0,
            "created_at": "2023-01-02T00:00:00",
        },
    ]
    
    # Act
    with patch("app.api.endpoints.documents.document_service.list_documents") as mock_list:
        mock_list.return_value = mock_docs
        response = test_app.get("/api/v1/documents/")
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert data[0]["document_id"] == "doc-1"
    assert data[1]["document_id"] == "doc-2"


def test_delete_document(test_app: TestClient) -> None:
    """Test deleting a document."""
    # Arrange
    doc_id = "test-doc-123"
    
    # Act
    with patch("app.api.endpoints.documents.document_service.delete_document") as mock_delete:
        mock_delete.return_value = True
        response = test_app.delete(f"/api/v1/documents/{doc_id}")
    
    # Assert
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert response.content == b""


def test_download_document(
    test_app: TestClient, 
    sample_pdf_file: tuple[bytes, str],
    test_upload_dir: Path
) -> None:
    """Test downloading a document file."""
    # Arrange
    doc_id = "test-doc-123"
    pdf_content, filename = sample_pdf_file
    
    # Create a test file
    doc_dir = test_upload_dir / doc_id
    doc_dir.mkdir(exist_ok=True)
    file_path = doc_dir / filename
    file_path.write_bytes(pdf_content)
    
    # Mock the document service
    mock_doc = {
        "document_id": doc_id,
        "filename": filename,
        "status": DocumentStatus.PROCESSED,
        "file_path": str(file_path),
    }
    
    # Act
    with patch("app.api.endpoints.documents.document_service.get_document") as mock_get:
        mock_get.return_value = mock_doc
        response = test_app.get(f"/api/v1/documents/{doc_id}/download")
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
    assert response.content == pdf_content
