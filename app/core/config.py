"""
Конфигурация приложения с поддержкой LangChain
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""
    # ====================================
    # API Keys
    # ====================================
    ANTHROPIC_API_KEY: str

    # ====================================
    # Database
    # ====================================
    DATABASE_URL: str

    # ====================================
    # Redis
    # ====================================
    REDIS_URL: str
    CACHE_TTL: int = 3600

    # ====================================
    # Application
    # ====================================
    APP_NAME: str = "SmartTask FAQ Service"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ====================================
    # LLM Settings
    # ====================================
    CLAUDE_MODEL: str = "claude-sonnet-4-5-20250929"
    MAX_RESPONSE_TOKENS: int = 1024

    # Алиас для обратной совместимости
    @property
    def LLM_MODEL(self) -> str:
        return self.CLAUDE_MODEL

    @property
    def LLM_MAX_TOKENS(self) -> int:
        return self.MAX_RESPONSE_TOKENS

    LLM_TEMPERATURE: float = 0.7

    # ====================================
    # LangChain Embeddings Settings
    # ====================================
    # Модель эмбедингов: light, medium, multilingual, russian
    EMBEDDING_MODEL: str = "russian"

    # Размерность эмбедингов (автоопределяется из модели, но можно переопределить)
    EMBEDDING_DIM: int = 1024

    # Старое название для обратной совместимости
    @property
    def VECTOR_DIMENSION(self) -> int:
        return self.EMBEDDING_DIM

    # ====================================
    # LangSmith Settings (Optional)
    # ====================================
    ENABLE_LANGSMITH: bool = False
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "smarttask-faq"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ====================================
    # Document Processing
    # ====================================
    DOCUMENTS_DIR: str = "./data/documents"
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 150
    PDF_LOADER_TYPE: str = "pypdf"

    # ====================================
    # Vector Store
    # ====================================
    VECTOR_SIMILARITY_THRESHOLD: float = 0.7

    # ====================================
    # Advanced LangChain Features (Optional)
    # ====================================
    ENABLE_LANGGRAPH: bool = False
    LANGCHAIN_CACHE: bool = True
    LANGCHAIN_CALLBACKS_BACKGROUND: bool = False

    # ====================================
    # Load Documents Memory Optimization
    # ====================================
    MAX_MEMORY_MB: int = 2048  # Максимальное использование памяти в MB
    PROCESSING_BATCH_SIZE: int = 3  # Размер батча для обработки чанков
    ENABLE_MEMORY_MONITOR: bool = True
    FORCE_GC_INTERVAL: int = 5  # Принудительная сборка мусора каждые N батчей

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    def get_embedding_dimension(self) -> int:
        """
        Получить размерность эмбедингов на основе модели

        Returns:
            int: Размерность вектора
        """
        dimension_map = {
            "light": 384,  # all-MiniLM-L6-v2
            "medium": 768,  # all-mpnet-base-v2
            "multilingual": 384,  # paraphrase-multilingual-MiniLM-L12-v2
            "russian": 1024  # intfloat/multilingual-e5-large
        }

        return dimension_map.get(self.EMBEDDING_MODEL, self.EMBEDDING_DIM)

    def is_langsmith_enabled(self) -> bool:
        """Проверка, включен ли LangSmith"""
        return self.ENABLE_LANGSMITH and self.LANGCHAIN_API_KEY is not None

    def get_langchain_config(self) -> dict:
        """Получить конфигурацию LangChain"""
        return {
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dimension": self.get_embedding_dimension(),
            "langsmith_enabled": self.is_langsmith_enabled(),
            "langsmith_project": self.LANGCHAIN_PROJECT if self.is_langsmith_enabled() else None,
            "pdf_loader": self.PDF_LOADER_TYPE,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP
        }


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки приложения (с кэшированием)"""
    return Settings()


settings = get_settings()