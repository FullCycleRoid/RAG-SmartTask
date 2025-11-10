"""
OpenAI Embedding Service
"""

import asyncio
from typing import List

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logger import logger

settings = get_settings()


class OpenAIEmbeddingService:
    """
    Сервис эмбеддингов на базе OpenAI API
    """

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: str = None,
    ):
        """
        Инициализация сервиса

        Args:
            model: Название модели OpenAI
            api_key: OpenAI API ключ
        """
        self.model = model
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.embedding_dimension = settings.EMBEDDING_DIM

        # Инициализация OpenAI клиента
        self.client = AsyncOpenAI(api_key=self.api_key)

        logger.info(
            f"✅ OpenAI Embedding Service initialized: {model} (dim={self.embedding_dimension})"
        )

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Генерировать эмбеддинг для текста

        Args:
            text: Текст для векторизации

        Returns:
            List[float]: Векторное представление
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return [0.0] * self.embedding_dimension

        try:
            # Ограничиваем длину текста для эмбеддингов (OpenAI лимит ~8192 токена)
            max_length = 8000  # Безопасный лимит символов
            if len(text) > max_length:
                logger.warning(
                    f"Text too long for embedding, truncating to {max_length} characters"
                )
                text = text[:max_length]

            response = await self.client.embeddings.create(model=self.model, input=text)

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            # Возвращаем нулевой вектор в случае ошибки
            return [0.0] * self.embedding_dimension

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Генерировать эмбеддинги для батча текстов

        Args:
            texts: Список текстов для векторизации

        Returns:
            List[List[float]]: Список векторных представлений
        """
        if not texts:
            return []

        try:
            # Фильтруем пустые тексты
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return [[] for _ in texts]

            # Ограничиваем длину текстов
            processed_texts = []
            for text in valid_texts:
                if len(text) > 8000:
                    processed_texts.append(text[:8000])
                else:
                    processed_texts.append(text)

            response = await self.client.embeddings.create(
                model=self.model, input=processed_texts
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings batch: {e}")
            return [[] for _ in texts]

    def get_model_info(self) -> dict:
        """Получить информацию о модели"""
        return {
            "provider": "openai",
            "model_name": self.model,
            "embedding_dimension": self.embedding_dimension,
            "requires_api_key": True,
            "local": False,
        }


def create_openai_embedding_service(
    model: str = None, api_key: str = None
) -> OpenAIEmbeddingService:
    """
    Фабрика для создания embedding service

    Args:
        model: Модель OpenAI (по умолчанию из настроек)
        api_key: OpenAI API ключ (по умолчанию из настроек)

    Returns:
        OpenAIEmbeddingService: Инициализированный сервис
    """
    model = model or settings.OPENAI_EMBEDDING_MODEL
    logger.info(f"Creating OpenAI embedding service: {model}")

    return OpenAIEmbeddingService(model=model, api_key=api_key)
