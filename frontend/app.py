import os
import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from pathlib import Path

# Page config
st.set_page_config(
    page_title="RAG Service",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/documents/upload"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/completions"
MODELS_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/models"

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

# Helper functions
def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available models from the API."""
    try:
        response = requests.get(MODELS_ENDPOINT)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def upload_document(file, metadata: Optional[Dict] = None) -> Optional[Dict]:
    """Upload a document to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"metadata": json.dumps(metadata or {})}
        
        response = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            data=data
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return None

def chat_completion(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    stream: bool = False,
    context_window: int = 5,
    document_id: Optional[str] = None
) -> Dict:
    """Send a chat completion request to the API."""
    try:
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "context_window": context_window,
            "document_id": document_id,
            "include_context": True
        }
        
        if stream:
            response = requests.post(
                CHAT_ENDPOINT,
                json=data,
                stream=True
            )
            response.raise_for_status()
            return response
        else:
            response = requests.post(
                CHAT_ENDPOINT,
                json=data
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error in chat completion: {e}")
        raise

# Sidebar
with st.sidebar:
    st.title("RAG Service")
    
    # Model selection
    st.subheader("Model Settings")
    models = get_available_models()
    model_names = [m["id"] for m in models]
    selected_model = st.selectbox(
        "Select Model",
        model_names,
        index=model_names.index("gpt-3.5-turbo") if "gpt-3.5-turbo" in model_names else 0
    )
    
    # Model parameters
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)
    context_window = st.slider("Context Window", 1, 20, 5)
    
    # Document selection
    st.subheader("Document Context")
    selected_document = st.selectbox(
        "Use Document for Context",
        ["None"] + [doc.get("title", doc.get("id", "Unknown")) for doc in st.session_state.uploaded_documents],
        index=0
    )
    
    # Document upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md", "docx", "pptx", "xlsx"]
    )
    
    if uploaded_file is not None:
        if st.button("Upload"):
            with st.spinner("Uploading document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.session_state.uploaded_documents.append(result)
                    st.success(f"Document '{result.get('title', '')}' uploaded successfully!")
                else:
                    st.error("Failed to upload document.")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Main chat interface
st.title("Chat with RAG")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show context if available
        if message.get("context"):
            with st.expander("View context used"):
                st.text(message["context"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get document ID if a document is selected
    document_id = None
    if selected_document != "None":
        for doc in st.session_state.uploaded_documents:
            if doc.get("title", doc.get("id", "")) == selected_document:
                document_id = doc.get("id")
                break
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare messages for the API
        messages_for_api = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
            if msg["role"] in ["user", "assistant"]
        ]
        
        try:
            # Stream the response
            response = chat_completion(
                messages=messages_for_api,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                context_window=context_window,
                document_id=document_id
            )
            
            # Stream the response
            for line in response.iter_lines():
                if line:
                    chunk = line.decode('utf-8')
                    if chunk.startswith('data: '):
                        chunk = chunk[6:]  # Remove 'data: ' prefix
                        if chunk.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(chunk)
                            if 'content' in data:
                                full_response += data['content']
                                message_placeholder.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            continue
            
            # Final update without the cursor
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "context": ""  # Context would be available in non-streaming mode
            })
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add some CSS for better styling
st.markdown("""
    <style>
        .stTextInput input {
            padding: 10px !important;
        }
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        .stSpinner > div > div {
            border-color: #4CAF50 !important;
        }
    </style>
""", unsafe_allow_html=True)
