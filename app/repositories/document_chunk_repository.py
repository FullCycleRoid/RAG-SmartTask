"""
Репозиторий для работы с фрагментами документов
"""

from typing import List, Tuple

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import logger
from app.models.documents import DocumentChunk
from app.repositories.base import BaseRepository


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    """Репозиторий для работы с DocumentChunk"""

    def __init__(self, db: AsyncSession):
        """
        Инициализация репозитория

        Args:
            db: Асинхронная сессия БД
        """
        super().__init__(DocumentChunk, db)
        self.similarity_threshold = settings.VECTOR_SIMILARITY_THRESHOLD

    async def create_chunk(
            self,
            document_name: str,
            content: str,
            embedding: List[float],
            chunk_index: int,
    ) -> DocumentChunk:
        """
        Создать новый фрагмент документа

        Args:
            document_name: Название документа
            content: Текстовое содержимое фрагмента
            embedding: Векторное представление
            chunk_index: Индекс фрагмента в документе

        Returns:
            DocumentChunk: Созданный фрагмент
        """
        try:
            # Проверяем размер контента
            if len(content) > settings.CHUNK_SIZE + 100:  # Допуск 100 символов
                logger.warning(f"Chunk {chunk_index} from {document_name} is large: {len(content)} chars")

            chunk = await self.create(
                document_name=document_name,
                content=content,
                embedding=embedding,
                chunk_index=chunk_index,
            )
            logger.debug(f"Created chunk {chunk_index} from {document_name} ({len(content)} chars)")
            return chunk

        except Exception as e:
            logger.error(f"Error creating chunk: {e}")
            raise

    async def get_by_document_name(self, document_name: str) -> List[DocumentChunk]:
        """
        Получить все фрагменты документа

        Args:
            document_name: Название документа

        Returns:
            List[DocumentChunk]: Список фрагментов
        """
        try:
            query = (
                select(DocumentChunk)
                .where(DocumentChunk.document_name == document_name)
                .order_by(DocumentChunk.chunk_index)
            )

            result = await self.db.execute(query)
            chunks = list(result.scalars().all())

            logger.debug(f"Retrieved {len(chunks)} chunks from {document_name}")
            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks by document: {e}")
            return []

    async def search_similar_chunks(self, embedding: list, limit: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Поиск похожих фрагментов по векторному представлению

        Args:
            embedding: Векторное представление для поиска
            limit: Максимальное количество результатов

        Returns:
            List[Tuple[DocumentChunk, float]]: Список (фрагмент, similarity_score)
        """
        try:
            embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

            stmt = text("""
                SELECT 
                    id,
                    document_name,
                    content,
                    chunk_index,
                    created_at,
                    1 - (embedding <=> :embedding) as similarity
                FROM document_chunks
                WHERE 1 - (embedding <=> :embedding) > :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """)

            results = await self.db.execute(
                stmt,
                {
                    'embedding': embedding_str,
                    'threshold': self.similarity_threshold,
                    'limit': limit
                }
            )

            # Преобразуем результаты в кортежи (DocumentChunk, similarity)
            similar_chunks = []
            for row in results:
                chunk = DocumentChunk(
                    id=row[0],
                    document_name=row[1],
                    content=row[2],
                    chunk_index=row[3],
                    created_at=row[4]
                )
                similar_chunks.append((chunk, float(row[5])))
            return similar_chunks

        except Exception as e:
            logger.error(f"Repository error searching similar chunks: {e}")
            return []

    async def delete_by_document_name(self, document_name: str) -> int:
        """
        Удалить все фрагменты документа

        Args:
            document_name: Название документа

        Returns:
            int: Количество удаленных фрагментов
        """
        try:
            chunks = await self.get_by_document_name(document_name)
            count = 0

            for chunk in chunks:
                await self.delete(chunk.id)
                count += 1

            logger.info(f"Deleted {count} chunks from {document_name}")
            return count

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise

    async def get_unique_document_names(self) -> List[str]:
        """
        Получить список уникальных имен документов

        Returns:
            List[str]: Список имен документов
        """
        try:
            query = select(DocumentChunk.document_name).distinct()
            result = await self.db.execute(query)
            documents = [row[0] for row in result.fetchall()]
            return documents

        except Exception as e:
            logger.error(f"Error getting document names: {e}")
            return []

    async def get_chunks_count_by_document(self, document_name: str) -> int:
        """
        Получить количество фрагментов для документа

        Args:
            document_name: Название документа

        Returns:
            int: Количество фрагментов
        """
        try:
            chunks = await self.get_by_document_name(document_name)
            return len(chunks)
        except Exception as e:
            logger.error(f"Error counting chunks: {e}")
            return 0

    async def get_document_count(self) -> int:
        """
        Получить количество уникальных документов

        Returns:
            int: Количество документов
        """
        try:
            names = await self.get_unique_document_names()
            return len(names)
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0