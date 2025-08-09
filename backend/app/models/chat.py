from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, validator

class ModelInfo(BaseModel):
    """Information about a language model."""
    id: str = Field(..., description="The model identifier")
    object: str = Field("model", description="The object type, always 'model'")
    created: int = Field(..., description="When the model was created (Unix timestamp)")
    owned_by: str = Field(..., description="The organization that owns the model")
    permission: List[Dict[str, Any]] = Field(default_factory=list, description="List of permissions for the model")
    root: str = Field(..., description="The root model identifier")
    parent: Optional[str] = Field(None, description="The parent model identifier, if any")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
                "permission": [],
                "root": "gpt-3.5-turbo",
                "parent": None
            }
        }

class ChatRole(str, Enum):
    """Role of the message sender."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: ChatRole = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="The name of the message sender")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was sent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the message")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?",
                "name": "John",
                "timestamp": "2023-01-01T00:00:00Z",
                "metadata": {}
            }
        }

class ChatRequest(BaseModel):
    """Request model for chat completion."""
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: float = Field(0.7, ge=0, le=2, description="Controls randomness (0-2)")
    max_tokens: int = Field(1000, gt=0, description="Maximum number of tokens to generate")
    top_p: float = Field(1.0, ge=0, le=1, description="Nucleus sampling parameter")
    context_window: int = Field(5, ge=1, le=20, description="Number of relevant chunks to include")
    document_id: Optional[str] = Field(None, description="ID of the document to use for context")
    include_context: bool = Field(False, description="Whether to include the context in the response")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("At least one message is required")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
                "context_window": 5,
                "document_id": "doc_12345",
                "include_context": False
            }
        }

class ChatResponseChoice(BaseModel):
    """A choice in the chat response."""
    message: ChatMessage
    finish_reason: str
    index: int

class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    """Response model for chat completion."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Type of the response")
    created: int = Field(..., description="When the completion was created")
    model: str = Field(..., description="The model used for the completion")
    choices: List[ChatResponseChoice] = Field(..., description="List of completion choices")
    usage: TokenUsage = Field(..., description="Token usage statistics")
    context: Optional[str] = Field(None, description="The context used for the completion")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-12345",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris.",
                            "timestamp": "2023-01-01T00:00:00Z"
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                },
                "context": "Paris is the capital of France..."
            }
        }

class ChatSession(BaseModel):
    """A chat session with a history of messages."""
    id: str = Field(default_factory=lambda: f"chat_{uuid.uuid4().hex}", description="Unique identifier for the chat session")
    user_id: Optional[str] = Field(None, description="ID of the user who owns the chat session")
    title: str = Field("New Chat", description="Title of the chat session")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the chat session was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the chat session was last updated")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of messages in the chat session")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the chat session")
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the chat session."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_messages(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get messages from the chat session.
        
        Args:
            limit: Maximum number of messages to return. If None, returns all messages.
            
        Returns:
            List of messages, optionally limited to the most recent ones.
        """
        if limit is not None and limit > 0:
            return self.messages[-limit:]
        return self.messages
    
    def clear_messages(self) -> None:
        """Clear all messages from the chat session."""
        self.messages = []
        self.updated_at = datetime.utcnow()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        json_schema_extra = {
            "example": {
                "id": "chat_12345",
                "user_id": "user_123",
                "title": "My Chat",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:05:00Z",
                "messages": [
                    {"role": "user", "content": "Hello!", "timestamp": "2023-01-01T00:00:00Z"},
                    {"role": "assistant", "content": "Hi there! How can I help you today?", "timestamp": "2023-01-01T00:00:01Z"}
                ],
                "metadata": {}
            }
        }

class ChatSessionCreate(BaseModel):
    """Model for creating a new chat session."""
    title: str = Field("New Chat", description="Title of the chat session")
    user_id: Optional[str] = Field(None, description="ID of the user who owns the chat session")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the chat session")

class ChatSessionUpdate(BaseModel):
    """Model for updating a chat session."""
    title: Optional[str] = Field(None, description="New title for the chat session")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata for the chat session")
