from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import time
import uuid
from loguru import logger

from app.models.chat import ChatMessage, ChatRole
from app.config.settings import settings

class LLMService(ABC):
    """Abstract base class for LLM services."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the LLM service.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.client = self._initialize_client(**kwargs)
    
    @abstractmethod
    def _initialize_client(self, **kwargs):
        """Initialize the client for the LLM service."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop: List of strings to stop generation at
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated text
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generated text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            stop: List of strings to stop generation at
            **kwargs: Additional model-specific parameters
            
        Yields:
            Chunks of generated text
        """
        pass
    
    def format_prompt(
        self,
        messages: List[ChatMessage],
        context: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Format messages and context into a prompt for the LLM.
        
        Args:
            messages: List of chat messages
            context: Optional context to include in the prompt
            document_id: Optional document ID for context
            
        Returns:
            Formatted prompt string
        """
        # Start with system message if context is provided
        prompt_parts = []
        
        if context:
            system_message = (
                "You are a helpful AI assistant. "
                "Answer the user's questions based on the context provided below. "
                "If you don't know the answer, say you don't know.\n\n"
                f"Context:\n{context}\n\n"
            )
            prompt_parts.append({"role": "system", "content": system_message})
        
        # Add conversation history
        for msg in messages:
            role = "user" if msg.role == ChatRole.USER else "assistant"
            prompt_parts.append({"role": role, "content": msg.content})
        
        return prompt_parts

class OpenAIService(LLMService):
    """LLM service for OpenAI models."""
    
    def _initialize_client(self, **kwargs):
        """Initialize the OpenAI client."""
        if settings.AZURE_OPENAI_API_KEY:
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
        else:
            from openai import OpenAI
            return OpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI's API."""
        try:
            # For Azure, we need to specify the deployment name
            if settings.AZURE_OPENAI_API_KEY:
                kwargs["deployment_id"] = settings.AZURE_OPENAI_DEPLOYMENT or self.model_name
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI generation: {e}")
            raise
    
    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generated text using OpenAI's API."""
        try:
            # For Azure, we need to specify the deployment name
            if settings.AZURE_OPENAI_API_KEY:
                kwargs["deployment_id"] = settings.AZURE_OPENAI_DEPLOYMENT or self.model_name
            
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in OpenAI streaming: {e}")
            raise

# Factory function to get the appropriate LLM service
def get_llm_service(
    model_name: Optional[str] = None,
    **kwargs
) -> LLMService:
    """Get an instance of the appropriate LLM service.
    
    Args:
        model_name: Name of the model to use. If None, uses default from settings.
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of an LLM service
    """
    model_name = model_name or settings.DEFAULT_LLM_MODEL or "gpt-3.5-turbo"
    
    # For now, we only support OpenAI models
    # In the future, we can add support for other providers like Anthropic, Cohere, etc.
    return OpenAIService(model_name=model_name, **kwargs)
