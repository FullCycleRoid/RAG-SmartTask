"""
Сервис векторного хранилища (pgvector) - использует репозиторий
"""

from typing import List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import logger
from app.models.documents import DocumentChunk
from app.repositories.document_chunk_repository import DocumentChunkRepository


class VectorStore:
    """Сервис для работы с векторным хранилищем через репозиторий"""

    def __init__(self, db: AsyncSession):
        """
        Инициализация сервиса

        Args:
            db: Асинхронная сессия БД
        """
        self.repository = DocumentChunkRepository(db)

    async def add_chunk(
        self, document_name: str, content: str, embedding: List[float], chunk_index: int
    ) -> DocumentChunk:
        """
        Добавить фрагмент в векторное хранилище

        Args:
            document_name: Название документа
            content: Текстовое содержимое
            embedding: Векторное представление
            chunk_index: Индекс фрагмента

        Returns:
            DocumentChunk: Созданный фрагмент
        """
        try:
            chunk = await self.repository.create_chunk(
                document_name=document_name,
                content=content,
                embedding=embedding,
                chunk_index=chunk_index,
            )
            return chunk

        except Exception as e:
            logger.error(f"Error adding chunk via VectorStore: {e}")
            raise

    async def search_similar(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Поиск похожих фрагментов по векторному представлению

        Args:
            query_embedding: Векторное представление запроса
            limit: Максимальное количество результатов

        Returns:
            List[Tuple[DocumentChunk, float]]: Список (фрагмент, similarity_score)
        """
        try:
            chunks_with_scores = await self.repository.search_similar_chunks(
                embedding=query_embedding, limit=limit
            )
            return chunks_with_scores

        except Exception as e:
            logger.error(f"Vector store error searching similar chunks: {e}")
            return []

    async def delete_document_chunks(self, document_name: str) -> int:
        """
        Удалить все фрагменты документа

        Args:
            document_name: Название документа

        Returns:
            int: Количество удаленных фрагментов
        """
        try:
            count = await self.repository.delete_by_document_name(document_name)
            return count

        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise

    async def get_document_count(self) -> int:
        """
        Получить количество уникальных документов

        Returns:
            int: Количество документов
        """
        try:
            return await self.repository.get_document_count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    async def get_all_documents(self) -> List[str]:
        """
        Получить список всех документов

        Returns:
            List[str]: Список названий документов
        """
        try:
            return await self.repository.get_unique_document_names()
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
