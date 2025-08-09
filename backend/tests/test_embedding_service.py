"""Tests for the embedding service."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.services.embedding_service import (
    EmbeddingService,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    get_embedding_service,
)
from config.settings import settings


def test_embedding_service_abstract() -> None:
    """Test that EmbeddingService is an abstract base class."""
    with pytest.raises(TypeError):
        EmbeddingService()  # type: ignore


def test_get_embedding_service_sentence_transformer() -> None:
    """Test getting a SentenceTransformer embedding service."""
    with patch("app.services.embedding_service.SentenceTransformer") as mock_st:
        # Mock the SentenceTransformer model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model
        
        # Test with default model
        service = get_embedding_service()
        assert isinstance(service, SentenceTransformerEmbedding)
        assert service.model_name == settings.EMBEDDING_MODEL
        
        # Test with custom model
        custom_model = "all-mpnet-base-v2"
        service = get_embedding_service(model_name=custom_model)
        assert service.model_name == custom_model


def test_get_embedding_service_openai() -> None:
    """Test getting an OpenAI embedding service."""
    with patch("app.services.embedding_service.openai") as mock_openai:
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )
        mock_openai.OpenAI.return_value = mock_client
        
        # Test with OpenAI model
        service = get_embedding_service(model_name="text-embedding-ada-002")
        assert isinstance(service, OpenAIEmbedding)
        assert service.model_name == "text-embedding-ada-002"


def test_get_embedding_service_invalid() -> None:
    """Test getting an embedding service with an invalid model name."""
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        get_embedding_service(model_name="invalid-model")


class TestSentenceTransformerEmbedding:
    """Tests for the SentenceTransformer embedding service."""
    
    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock SentenceTransformer model."""
        with patch("app.services.embedding_service.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock_st.return_value = mock_model
            yield mock_model
    
    def test_embed_text(self, mock_model: MagicMock) -> None:
        """Test embedding a single text."""
        service = SentenceTransformerEmbedding()
        embedding = service.embed_text("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert all(isinstance(x, float) for x in embedding)
        mock_model.encode.assert_called_once_with("test text", show_progress_bar=False)
    
    def test_embed_texts(self, mock_model: MagicMock) -> None:
        """Test embedding multiple texts."""
        service = SentenceTransformerEmbedding()
        texts = ["text 1", "text 2"]
        embeddings = service.embed_texts(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        mock_model.encode.assert_called_once_with(texts, show_progress_bar=True)
    
    def test_get_embedding_dimension(self, mock_model: MagicMock) -> None:
        """Test getting the embedding dimension."""
        mock_model.get_sentence_embedding_dimension.return_value = 384
        service = SentenceTransformerEmbedding()
        
        assert service.get_embedding_dimension() == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()


class TestOpenAIEmbedding:
    """Tests for the OpenAI embedding service."""
    
    @pytest.fixture
    def mock_openai(self) -> MagicMock:
        """Create a mock OpenAI client."""
        with patch("app.services.embedding_service.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = MagicMock(
                data=[
                    MagicMock(embedding=[0.1, 0.2, 0.3]),
                    MagicMock(embedding=[0.4, 0.5, 0.6]),
                ]
            )
            mock_openai.OpenAI.return_value = mock_client
            yield mock_client
    
    def test_embed_text(self, mock_openai: MagicMock) -> None:
        """Test embedding a single text."""
        service = OpenAIEmbedding(model_name="text-embedding-ada-002")
        embedding = service.embed_text("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert all(isinstance(x, float) for x in embedding)
        
        # Check that the OpenAI client was called correctly
        mock_openai.OpenAI.return_value.embeddings.create.assert_called_once()
        call_args = mock_openai.OpenAI.return_value.embeddings.create.call_args
        assert call_args[1]["input"] == "test text"
        assert call_args[1]["model"] == "text-embedding-ada-002"
    
    def test_embed_texts(self, mock_openai: MagicMock) -> None:
        """Test embedding multiple texts."""
        service = OpenAIEmbedding(model_name="text-embedding-ada-002")
        texts = ["text 1", "text 2"]
        embeddings = service.embed_texts(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        
        # Check that the OpenAI client was called correctly
        mock_openai.OpenAI.return_value.embeddings.create.assert_called_once()
        call_args = mock_openai.OpenAI.return_value.embeddings.create.call_args
        assert call_args[1]["input"] == texts
        assert call_args[1]["model"] == "text-embedding-ada-002"
    
    def test_get_embedding_dimension(self) -> None:
        """Test getting the embedding dimension."""
        # Test with ada-002 model
        service = OpenAIEmbedding(model_name="text-embedding-ada-002")
        assert service.get_embedding_dimension() == 1536
        
        # Test with unknown model (should return default)
        service = OpenAIEmbedding(model_name="unknown-model")
        assert service.get_embedding_dimension() == 1536  # Default for unknown models
