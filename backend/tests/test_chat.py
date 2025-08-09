"""Tests for chat-related endpoints and functionality."""
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import status
from fastapi.testclient import TestClient

def test_list_models(test_app: TestClient) -> None:
    """Test listing available chat models."""
    # Arrange
    mock_models = [
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        {"id": "gpt-4", "name": "GPT-4"},
    ]
    
    # Act
    with patch("app.api.endpoints.chat.llm_service.list_models") as mock_list:
        mock_list.return_value = mock_models
        response = test_app.get("/api/v1/chat/models")
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == "gpt-3.5-turbo"
    assert data[1]["id"] == "gpt-4"


def test_chat_completion(test_app: TestClient) -> None:
    """Test generating a chat completion."""
    # Arrange
    request_data = {
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100,
        "use_rag": True,
    }
    
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there, how may I assist you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        },
        "context_documents": [
            {
                "document_id": "doc-123",
                "content": "Sample document content",
                "score": 0.85
            }
        ]
    }
    
    # Act
    with patch("app.api.endpoints.chat.chat_service.generate_chat_completion") as mock_chat:
        mock_chat.return_value = mock_response
        response = test_app.post("/api/v1/chat/completions", json=request_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == "chatcmpl-123"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "context_documents" in data


@pytest.mark.asyncio
async def test_stream_chat_completion(test_app: TestClient) -> None:
    """Test streaming chat completions."""
    # Arrange
    request_data = {
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "model": "gpt-3.5-turbo",
        "stream": True,
        "use_rag": True,
    }
    
    # Mock the async generator
    mock_chunks = [
        {"choices": [{"delta": {"role": "assistant"}, "index": 0}]},
        {"choices": [{"delta": {"content": "Hello"}, "index": 0}]},
        {"choices": [{"delta": {"content": " there"}, "index": 0}]},
        {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]},
    ]
    
    async def mock_generator():
        for chunk in mock_chunks:
            yield chunk
    
    # Act
    with patch(
        "app.api.endpoints.chat.chat_service.stream_chat_completion",
        return_value=mock_generator()
    ):
        response = test_app.post(
            "/api/v1/chat/completions",
            json=request_data,
            headers={"Accept": "text/event-stream"}
        )
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "text/event-stream" in response.headers["content-type"]
    
    # Parse the Server-Sent Events
    lines = response.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    
    assert len(data_lines) == 4  # 3 content chunks + [DONE]
    assert "role" in data_lines[0]
    assert "Hello" in data_lines[1]
    assert "there" in data_lines[2]
    assert data_lines[-1] == "data: [DONE]"


def test_chat_completion_missing_messages(test_app: TestClient) -> None:
    """Test chat completion with missing messages."""
    # Arrange
    request_data = {
        "model": "gpt-3.5-turbo",
    }
    
    # Act
    response = test_app.post("/api/v1/chat/completions", json=request_data)
    
    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "field required" in response.text.lower()


def test_chat_completion_invalid_model(test_app: TestClient) -> None:
    """Test chat completion with an invalid model."""
    # Arrange
    request_data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "invalid-model",
    }
    
    # Act
    with patch(
        "app.api.endpoints.chat.llm_service.get_model", 
        side_effect=ValueError("Model not found")
    ):
        response = test_app.post("/api/v1/chat/completions", json=request_data)
    
    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "model not found" in response.json()["detail"].lower()
