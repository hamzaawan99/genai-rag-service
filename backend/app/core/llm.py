from typing import List, Optional
import openai
from openai import AsyncAzureOpenAI
from app.core.config import settings

class AzureOpenAIClient:
    def __init__(self):
        openai.api_key = settings.AZURE_OPENAI_API_KEY
        openai.api_base = settings.AZURE_OPENAI_ENDPOINT
        openai.api_type = "azure"
        openai.api_version = settings.AZURE_OPENAI_VERSION

        self.aoai_client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )

    async def get_chat_completion(
        self,
        messages: List[dict],
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        # response = await openai.ChatCompletion.acreate(
        #     deployment_id=settings.AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT,
        #     messages=messages,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        # )
        response = await self.aoai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def get_embeddings(self, text: str) -> List[float]:
        response = self.aoai_client.embeddings.create(
            model=settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
            input=text,
        )
        return response.data[0].embedding
