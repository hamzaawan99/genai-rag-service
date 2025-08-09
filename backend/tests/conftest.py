"""Pytest configuration and fixtures for the test suite."""
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Any

import pytest
from fastapi.testclient import TestClient

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app
from config.settings import settings


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create and return a temporary directory for test data."""
    test_dir = Path(tempfile.mkdtemp(prefix="rag_test_"))
    yield test_dir
    # Cleanup after tests are done
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def test_upload_dir(test_data_dir: Path) -> Path:
    """Create and return a temporary upload directory for tests."""
    upload_dir = test_data_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


@pytest.fixture(scope="session")
def test_faiss_index_dir(test_data_dir: Path) -> Path:
    """Create and return a temporary FAISS index directory for tests."""
    faiss_dir = test_data_dir / "faiss_index"
    faiss_dir.mkdir(exist_ok=True)
    return faiss_dir


@pytest.fixture(scope="module")
def test_app() -> Generator[TestClient, Any, None]:
    """Create a test client for the FastAPI app."""
    # Override settings for testing
    original_upload_dir = settings.UPLOAD_DIR
    original_faiss_path = settings.FAISS_INDEX_PATH
    
    # Create a temporary directory for test data
    test_dir = Path(tempfile.mkdtemp(prefix="rag_test_"))
    test_upload_dir = test_dir / "uploads"
    test_upload_dir.mkdir()
    test_faiss_dir = test_dir / "faiss_index"
    test_faiss_dir.mkdir()
    
    # Update settings
    settings.UPLOAD_DIR = str(test_upload_dir)
    settings.FAISS_INDEX_PATH = str(test_faiss_dir)
    settings.ENVIRONMENT = "test"
    
    # Create test client
    with TestClient(app) as client:
        yield client
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    
    # Restore original settings
    settings.UPLOAD_DIR = original_upload_dir
    settings.FAISS_INDEX_PATH = original_faiss_path


@pytest.fixture
def sample_pdf_file() -> tuple[bytes, str]:
    """Return a sample PDF file for testing uploads."""
    # This is a minimal PDF file
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 700 Td (Hello, World!) Tj ET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000114 00000 n \n0000000212 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n303\n%%EOF"
    )
    return pdf_content, "test.pdf"


@pytest.fixture
def sample_text_file() -> tuple[bytes, str]:
    """Return a sample text file for testing uploads."""
    text_content = b"This is a sample text file for testing.\nIt contains multiple lines.\n"
    return text_content, "test.txt"
