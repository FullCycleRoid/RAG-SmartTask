"""
LangChain-based Embedding Service
Локальные эмбединги без необходимости API ключей
"""

from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from app.core.logger import logger


class LangChainEmbeddingService:
    """
    Сервис эмбедингов на базе LangChain с локальными моделями
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: str = "./data/models",
    ):
        """
        Инициализация сервиса

        Args:
            model_name: Название модели HuggingFace
            device: Устройство для вычислений ('cpu' или 'cuda')
            cache_folder: Папка для кэширования моделей
        """
        self.model_name = model_name
        self.device = device
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initializing LangChain Embedding Service with model: {model_name}"
        )

        try:
            # Инициализация HuggingFace Embeddings через LangChain
            self.embeddings: Embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device, "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )

            # Определяем размерность эмбедингов
            self._determine_embedding_dimension()

            logger.info(
                f"✅ LangChain Embeddings initialized: {model_name} (dim={self.embedding_dimension})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _determine_embedding_dimension(self):
        """Определить размерность эмбедингов"""
        dimension_map = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "paraphrase-MiniLM-L6-v2": 384,
            "intfloat/multilingual-e5-large": 1024,
        }

        if self.model_name in dimension_map:
            self.embedding_dimension = dimension_map[self.model_name]
        else:
            # Если модель неизвестна, определяем размерность через тестовый вектор
            try:
                test_embedding = self.embeddings.embed_query("test")
                self.embedding_dimension = len(test_embedding)
                logger.info(f"Detected embedding dimension: {self.embedding_dimension}")
            except Exception as e:
                logger.error(f"Error detecting embedding dimension: {e}")
                self.embedding_dimension = 384  # fallback

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Генерировать эмбеддинг для текста

        Args:
            text: Текст для векторизации

        Returns:
            List[float]: Векторное представление
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension

        try:
            embedding = self.embeddings.embed_query(text)
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.embedding_dimension

    def get_model_info(self) -> dict:
        """Получить информацию о модели"""
        return {
            "provider": "langchain-huggingface",
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "cache_folder": str(self.cache_folder),
            "requires_api_key": False,
            "local": True,
        }


EMBEDDING_MODELS = {
    # Легкая модель для быстрой работы
    "light": {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Быстрая и легкая модель для большинства задач",
    },
    # Средняя модель для лучшего качества
    "medium": {
        "model_name": "all-mpnet-base-v2",
        "dimension": 768,
        "description": "Сбалансированное качество и скорость",
    },
    #
    # # Мультиязычная модель для русского и английского
    "multilingual": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Поддержка множества языков, включая русский",
    },
    #
    # # Большая русскоязычная модель
    "russian": {
        "model_name": "intfloat/multilingual-e5-large",
        "dimension": 1024,
        "description": "Отличная поддержка русского языка",
    },
}


def create_langchain_embedding_service(
    model_type: str = "light", device: str = "cpu"
) -> LangChainEmbeddingService:
    """
    Фабрика для создания embedding service

    Args:
        model_type: Тип модели ('light', 'medium', 'multilingual', 'russian')
        device: Устройство ('cpu' или 'cuda')

    Returns:
        LangChainEmbeddingService: Инициализированный сервис
    """
    if model_type not in EMBEDDING_MODELS:
        logger.warning(f"Unknown model type: {model_type}, using 'light'")
        model_type = "light"

    model_config = EMBEDDING_MODELS[model_type]
    logger.info(f"Creating embedding service: {model_config['description']}")

    return LangChainEmbeddingService(
        model_name=model_config["model_name"], device=device
    )
