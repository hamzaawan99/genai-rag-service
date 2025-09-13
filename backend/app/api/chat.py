from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.chat import ChatService
from app.schemas.chat import ChatRequest, ChatResponse
from app.models.user import User
from app.api.auth import get_current_user

router = APIRouter()
chat_service = ChatService()

@router.post("", response_model=ChatResponse)
async def chat_with_knowledge_base(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        response = await chat_service.chat(
            db=db,
            chat_request=chat_request,
            user=current_user
        )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
