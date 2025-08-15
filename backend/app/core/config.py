from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import PostgresDsn

class Settings(BaseSettings):
    PROJECT_NAME: str = "GenAI RAG Service"
    API_V1_STR: str = "/api/v1"

    WEAVIATE_API_USER: str
    WEAVIATE_API_KEY: str

    REDIS_PORT: str
    REDIS_PASSWORD: str
    
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DATABASE: str
    
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None
    
    # JWT settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Construct database URI
        if not self.SQLALCHEMY_DATABASE_URI:
            self.SQLALCHEMY_DATABASE_URI = PostgresDsn.build(
                scheme="postgresql",
                username=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD,
                host=self.POSTGRES_HOST,
                port=int(self.POSTGRES_PORT),
                path=f"{self.POSTGRES_DATABASE}"
            )

settings = Settings()
