"""Tests for the FAISS vector store implementation."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.core.vector_store.faiss_store import FAISSVectorStore
from app.models.embedding import TextChunk, DocumentChunk
from config.settings import settings


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for FAISS index files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_chunks() -> List[DocumentChunk]:
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i//2}",  # Two chunks per document
            text=f"Sample text {i}",
            metadata={"page": i % 3, "source": f"doc-{i//2}.pdf"},
            embedding=[float(x) for x in np.random.rand(384).tolist()],
        )
        for i in range(6)
    ]


class TestFAISSVectorStore:
    """Tests for the FAISSVectorStore class."""
    
    def test_initialize_new_index(self, temp_index_dir: Path):
        """Test initializing a new FAISS index."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        dimension = 384
        
        # Act
        store = FAISSVectorStore(index_path=str(index_path), dimension=dimension)
        
        # Assert
        assert store.index is not None
        assert store.dimension == dimension
        assert store.index.ntotal == 0  # No vectors added yet
        
        # Check that the index file was created
        assert (temp_index_dir / "test_index.index").exists()
        assert (temp_index_dir / "test_index.metadata.json").exists()
    
    def test_add_documents(self, temp_index_dir: Path, sample_chunks: List[DocumentChunk]):
        """Test adding document chunks to the index."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        store = FAISSVectorStore(index_path=str(index_path), dimension=384)
        
        # Act
        store.add_documents(sample_chunks)
        
        # Assert
        assert store.index.ntotal == len(sample_chunks)
        assert len(store.metadata_store) == len(sample_chunks)
        
        # Check that all chunks were added correctly
        for chunk in sample_chunks:
            assert chunk.chunk_id in store.metadata_store
            assert store.metadata_store[chunk.chunk_id] == chunk.dict()
    
    def test_similarity_search(self, temp_index_dir: Path, sample_chunks: List[DocumentChunk]):
        """Test searching for similar vectors."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        store = FAISSVectorStore(index_path=str(index_path), dimension=384)
        store.add_documents(sample_chunks)
        
        # Create a query embedding (using the first chunk's embedding as the query)
        query_embedding = sample_chunks[0].embedding
        
        # Act
        results = store.similarity_search(
            query_embedding=query_embedding,
            k=2,
            filter_conditions={"document_id": "doc-0"}
        )
        
        # Assert
        assert len(results) == 2  # We have 2 chunks for doc-0
        assert all(isinstance(result, TextChunk) for result in results)
        assert all(result.document_id == "doc-0" for result in results)
    
    def test_get_document_chunks(self, temp_index_dir: Path, sample_chunks: List[DocumentChunk]):
        """Test retrieving all chunks for a document."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        store = FAISSVectorStore(index_path=str(index_path), dimension=384)
        store.add_documents(sample_chunks)
        
        # Act
        doc_chunks = store.get_document_chunks("doc-1")
        
        # Assert
        assert len(doc_chunks) == 2  # We have 2 chunks for doc-1
        assert all(chunk.document_id == "doc-1" for chunk in doc_chunks)
    
    def test_delete_document(self, temp_index_dir: Path, sample_chunks: List[DocumentChunk]):
        """Test deleting a document and its chunks from the index."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        store = FAISSVectorStore(index_path=str(index_path), dimension=384)
        store.add_documents(sample_chunks)
        
        # Get the chunk IDs for doc-0
        doc0_chunk_ids = [chunk.chunk_id for chunk in sample_chunks if chunk.document_id == "doc-0"]
        
        # Act
        deleted = store.delete_document("doc-0")
        
        # Assert
        assert deleted is True
        assert store.index.ntotal == len(sample_chunks) - len(doc0_chunk_ids)
        assert all(chunk_id not in store.metadata_store for chunk_id in doc0_chunk_ids)
    
    def test_save_and_load(self, temp_index_dir: Path, sample_chunks: List[DocumentChunk]):
        """Test saving and loading the index from disk."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        
        # Create and populate a store
        store1 = FAISSVectorStore(index_path=str(index_path), dimension=384)
        store1.add_documents(sample_chunks)
        
        # Save the store
        store1.save()
        
        # Create a new store that loads from disk
        store2 = FAISSVectorStore(index_path=str(index_path), dimension=384)
        
        # Assert
        assert store2.index.ntotal == len(sample_chunks)
        assert len(store2.metadata_store) == len(sample_chunks)
    
    def test_clear(self, temp_index_dir: Path, sample_chunks: List[DocumentChunk]):
        """Test clearing the index."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        store = FAISSVectorStore(index_path=str(index_path), dimension=384)
        store.add_documents(sample_chunks)
        
        # Act
        store.clear()
        
        # Assert
        assert store.index.ntotal == 0
        assert len(store.metadata_store) == 0
    
    def test_get_stats(self, temp_index_dir: Path, sample_chunks: List[DocumentChunk]):
        """Test getting statistics about the index."""
        # Arrange
        index_path = temp_index_dir / "test_index"
        store = FAISSVectorStore(index_path=str(index_path), dimension=384)
        store.add_documents(sample_chunks)
        
        # Act
        stats = store.get_stats()
        
        # Assert
        assert stats["total_chunks"] == len(sample_chunks)
        assert stats["dimension"] == 384
        assert stats["documents"] == 3  # 3 unique document_ids in sample_chunks
