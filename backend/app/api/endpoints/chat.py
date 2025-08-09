from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional, AsyncGenerator, Dict, Any
import json
import time
from loguru import logger

from app.models.chat import ChatMessage, ChatRequest, ChatResponse, ChatRole
from app.services.llm_service import get_llm_service, LLMService
from app.core.vector_store.faiss_store import FAISSStore
from app.services.embedding_service import get_embedding_service
from app.config.settings import settings

router = APIRouter()

# Initialize services
vector_store = FAISSStore()
embedding_service = get_embedding_service()
llm_service = get_llm_service()

@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    llm: LLMService = Depends(get_llm_service)
):
    """
    Generate a chat completion with RAG capabilities.
    
    - **messages**: List of messages in the conversation
    - **stream**: Whether to stream the response
    - **temperature**: Controls randomness (0-2)
    - **max_tokens**: Maximum number of tokens to generate
    - **top_p**: Nucleus sampling parameter
    - **context_window**: Number of relevant chunks to include
    - **index_name**: Name of the vector store index to use
    
    Returns a chat completion response with the generated message.
    """
    try:
        # Get the last user message
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided"
            )
            
        last_message = request.messages[-1]
        if last_message.role != ChatRole.USER:
            raise HTTPException(
                status_code=status.HETTP_400_BAD_REQUEST,
                detail="Last message must be from user"
            )
        
        # Get relevant context from the vector store
        query_embedding = await embedding_service.get_embedding(last_message.content)
        search_results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=request.context_window or 5,
            filter_conditions={"document_id": request.document_id} if request.document_id else None,
            query_text=last_message.content
        )
        
        # Prepare the context for the LLM
        context = "\n\n".join(
            f"[Document {i+1}]\n{result.chunk.text}"
            for i, result in enumerate(search_results.results)
        )
        
        # Prepare the prompt with context and chat history
        prompt = llm_service.format_prompt(
            messages=request.messages,
            context=context,
            document_id=request.document_id
        )
        
        if request.stream:
            # Stream the response
            return StreamingResponse(
                stream_chat_completion(
                    llm=llm,
                    prompt=prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    context=context
                ),
                media_type="text/event-stream"
            )
        else:
            # Generate a single response
            response = await llm.generate(
                prompt=prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=None
            )
            
            return ChatResponse(
                id=f"chatcmpl-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=llm.model_name,
                choices=[
                    {
                        "message": {
                            "role": ChatRole.ASSISTANT,
                            "content": response
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split())
                },
                context=context if request.include_context else None
            )
            
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

async def stream_chat_completion(
    llm: LLMService,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 1.0,
    context: str = ""
) -> AsyncGenerator[str, None]:
    """Stream chat completion response."""
    try:
        # Send initial response with context if needed
        if context:
            yield f"data: {json.dumps({'context': context})}\n\n"
        
        # Stream the response from the LLM
        async for chunk in llm.stream(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        
        # Send a done signal
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming chat completion: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
@router.get("/models")
async def list_models():
    """List available chat models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "owned_by": "openai"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "owned_by": "openai"
            }
            # Add more models as needed
        ]
    }
