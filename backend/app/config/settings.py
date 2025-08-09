from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "RAG Service"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Service"
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # OpenAI settings (optional)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
    
    # Azure OpenAI settings (optional)
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = None
    
    # Vector Database settings
    VECTOR_DB: str = "faiss"  # or "weaviate"
    
    # Weaviate settings (if VECTOR_DB is "weaviate")
    WEAVIATE_URL: str = "http://weaviate:8080"
    WEAVIATE_API_KEY: Optional[str] = None
    
    # FAISS settings (if VECTOR_DB is "faiss")
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    
    # File storage settings
    UPLOAD_DIR: str = "./data/uploads"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Create settings instance
settings = Settings()
