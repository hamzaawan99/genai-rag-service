from typing import List, Optional, Dict, Any
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.llm import AzureOpenAIClient
from app.core.vector_store import VectorDBFactory
from app.models.chat import ChatSession, ChatMessage
from app.models.user import User
from app.models.knowledge_base import KnowledgeBase
from app.schemas.chat import ChatRequest

class ChatService:
    def __init__(self):
        self.llm_client = AzureOpenAIClient()
        
    async def chat(
        self,
        db: Session,
        chat_request: ChatRequest,
        user: User
    ) -> Dict[str, Any]:
        # Get knowledge base
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.id == chat_request.knowledge_base_id,
            KnowledgeBase.user_id == user.id
        ).first()
        if not kb:
            raise ValueError("Knowledge base not found")
            
        # Get or create chat session
        chat_session = self._get_or_create_chat_session(db, kb, user)
        
        # Get conversation history (last 10 messages)
        history = self._get_chat_history(db, chat_session.session_id)
        
        # Get relevant documents
        relevant_docs = self._search_relevant_documents(
            kb,
            chat_request.message,
            chat_request.max_context_docs
        )
        
        # Prepare messages for LLM
        messages = self._prepare_messages(history, chat_request.message, relevant_docs)
        
        # Get response from LLM
        response = await self.llm_client.get_chat_completion(
            messages=messages,
            system_message=self._get_system_prompt()
        )
        
        # Save messages to history
        self._save_messages(db, chat_session.session_id, chat_request.message, response)
        
        return {
            "message": response,
            "relevant_docs": relevant_docs,
            "conversation_id": chat_session.session_id
        }
    
    def _get_or_create_chat_session(
        self,
        db: Session,
        kb: KnowledgeBase,
        user: User
    ) -> ChatSession:
        # Try to find an existing session
        chat_session = db.query(ChatSession).filter(
            ChatSession.knowledge_base_id == kb.id,
            ChatSession.user_id == user.id
        ).order_by(desc(ChatSession.created_at)).first()
        
        if not chat_session:
            # Create new session if none exists
            chat_session = ChatSession(
                session_id=uuid.uuid4(),
                knowledge_base_id=kb.id,
                user_id=user.id
            )
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
        
        return chat_session
    
    def _get_chat_history(
        self,
        db: Session,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(desc(ChatMessage.created_at)).limit(limit).all()
        
        return [{"role": msg.role, "content": msg.content} for msg in reversed(messages)]
    
    def _search_relevant_documents(
        self,
        kb: KnowledgeBase,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        # Get embeddings for the query
        query_embedding = self.llm_client.get_embeddings(query)
        
        # Search in vector store
        vector_client = VectorDBFactory.create_client(kb.vector_db)
        results = vector_client.search_similar(
            collection_name=kb.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return results
    
    def _prepare_messages(
        self,
        history: List[Dict[str, str]],
        current_message: str,
        relevant_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        # Add context from relevant documents
        context = "Here are some relevant documents that might help answer the query:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Document {i}:\n{doc['content']}\n\n"
        
        messages = []
        
        # Add history
        messages.extend(history)
        
        # Add context and current message
        messages.append({"role": "user", "content": f"{context}\nQuery: {current_message}"})
        
        return messages
    
    def _save_messages(
        self,
        db: Session,
        session_id: str,
        user_message: str,
        assistant_message: str
    ) -> None:
        # Save user message
        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=user_message
        )
        db.add(user_msg)
        
        # Save assistant message
        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=assistant_message
        )
        db.add(assistant_msg)
        
        db.commit()
    
    def _get_system_prompt(self) -> str:
        return """You are a helpful AI assistant that answers questions based on the provided document context. 
        When responding:
        1. Use the relevant documents provided to formulate accurate answers
        2. If the documents don't contain enough information to answer the question, say so
        3. Be concise but informative
        4. Cite specific documents when referencing information"""
