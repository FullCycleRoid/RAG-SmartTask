"""
Репозитории для работы с БД
"""

from app.repositories.base import BaseRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.query_repository import QueryRepository

__all__ = [
    "BaseRepository",
    "QueryRepository",
    "DocumentChunkRepository",
]
