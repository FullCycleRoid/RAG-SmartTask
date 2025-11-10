"""
Pydantic схемы для валидации запросов и ответов
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Запрос с вопросом"""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="Вопрос пользователя"
    )
    session_id: Optional[str] = Field(
        None, max_length=255, description="ID сессии для отслеживания"
    )


class QueryResponse(BaseModel):
    """Ответ на вопрос"""

    question: str = Field(..., description="Исходный вопрос")
    answer: str = Field(..., description="Ответ на вопрос")
    sources: List["Source"] = Field(
        default_factory=list, description="Источники информации"
    )
    tokens_used: int = Field(0, description="Количество использованных токенов")
    response_time: float = Field(0.0, description="Время ответа в секундах")
    cached: bool = Field(False, description="Флаг использования кэша")


class QuestionRequest(BaseModel):
    """Запрос с вопросом"""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="Вопрос пользователя"
    )
    session_id: Optional[str] = Field(
        None, max_length=255, description="ID сессии для отслеживания"
    )


class Source(BaseModel):
    """Источник информации"""

    document: str
    content: str
    relevance: float


class QuestionResponse(BaseModel):
    """Ответ на вопрос"""

    answer: str
    sources: List[Source]
    tokens_used: int
    response_time: float
    cached: bool = False


class QueryHistory(BaseModel):
    """Элемент истории запросов"""

    id: int
    question: str
    answer: str
    tokens_used: int
    response_time: float
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    """Ответ после загрузки документа"""

    filename: str
    chunks_created: int
    message: str


QueryResponse.model_rebuild()
