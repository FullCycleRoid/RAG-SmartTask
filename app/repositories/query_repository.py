"""
Репозиторий для работы с запросами пользователей (Query)
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import logger
from app.models.query import Query
from app.repositories.base import BaseRepository


class QueryRepository(BaseRepository[Query]):
    """Репозиторий для работы с Query"""

    def __init__(self, db: AsyncSession):
        """
        Инициализация репозитория

        Args:
            db: Асинхронная сессия БД
        """
        super().__init__(Query, db)

    async def create_query(
        self,
        question: str,
        answer: str,
        tokens_used: int = 0,
        response_time: float = 0.0,
        sources: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Query:
        """
        Создать новый запрос

        Args:
            question: Вопрос пользователя
            answer: Ответ на вопрос
            tokens_used: Количество использованных токенов
            response_time: Время ответа в секундах
            sources: JSON строка с источниками
            session_id: ID сессии пользователя

        Returns:
            Query: Созданный запрос
        """
        try:
            query = await self.create(
                question=question,
                answer=answer,
                tokens_used=tokens_used,
                response_time=response_time,
                sources=sources,
                session_id=session_id,
            )
            logger.debug(f"Created query with id: {query.id}")
            return query

        except Exception as e:
            logger.error(f"Error creating query: {e}")
            raise

    async def get_recent_queries(
        self, limit: int = 10, session_id: Optional[str] = None
    ) -> List[Query]:
        """
        Получить последние запросы

        Args:
            limit: Максимальное количество запросов
            session_id: Фильтр по session_id (опционально)

        Returns:
            List[Query]: Список последних запросов
        """
        try:
            query = select(Query)

            if session_id:
                query = query.where(Query.session_id == session_id)

            query = query.order_by(desc(Query.created_at)).limit(limit)

            result = await self.db.execute(query)
            queries = list(result.scalars().all())

            logger.debug(f"Retrieved {len(queries)} recent queries")
            return queries

        except Exception as e:
            logger.error(f"Error getting recent queries: {e}")
            return []

    async def get_queries_by_session(self, session_id: str) -> List[Query]:
        """
        Получить все запросы для конкретной сессии

        Args:
            session_id: ID сессии

        Returns:
            List[Query]: Список запросов сессии
        """
        try:
            return await self.find_many(session_id=session_id)

        except Exception as e:
            logger.error(f"Error getting queries by session: {e}")
            return []

    async def get_queries_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Query]:
        """
        Получить запросы за период времени

        Args:
            start_date: Начальная дата
            end_date: Конечная дата

        Returns:
            List[Query]: Список запросов за период
        """
        try:
            query = (
                select(Query)
                .where(Query.created_at >= start_date)
                .where(Query.created_at <= end_date)
                .order_by(desc(Query.created_at))
            )

            result = await self.db.execute(query)
            queries = list(result.scalars().all())

            logger.debug(f"Retrieved {len(queries)} queries for date range")
            return queries

        except Exception as e:
            logger.error(f"Error getting queries by date range: {e}")
            return []

    async def get_total_tokens_used(self, session_id: Optional[str] = None) -> int:
        """
        Получить общее количество использованных токенов

        Args:
            session_id: Фильтр по session_id (опционально)

        Returns:
            int: Общее количество токенов
        """
        try:
            queries = (
                await self.get_queries_by_session(session_id)
                if session_id
                else await self.get_all()
            )
            return sum(q.tokens_used for q in queries)

        except Exception as e:
            logger.error(f"Error calculating total tokens: {e}")
            return 0

    async def get_average_response_time(
        self, session_id: Optional[str] = None
    ) -> float:
        """
        Получить среднее время ответа

        Args:
            session_id: Фильтр по session_id (опционально)

        Returns:
            float: Среднее время ответа в секундах
        """
        try:
            queries = (
                await self.get_queries_by_session(session_id)
                if session_id
                else await self.get_all()
            )

            if not queries:
                return 0.0

            total_time = sum(q.response_time for q in queries)
            return total_time / len(queries)

        except Exception as e:
            logger.error(f"Error calculating average response time: {e}")
            return 0.0

    async def delete_old_queries(self, days: int = 30) -> int:
        """
        Удалить старые запросы

        Args:
            days: Удалить запросы старше указанного количества дней

        Returns:
            int: Количество удаленных запросов
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            query = select(Query).where(Query.created_at < cutoff_date)
            result = await self.db.execute(query)
            old_queries = list(result.scalars().all())

            for query in old_queries:
                await self.db.delete(query)

            await self.db.flush()

            logger.info(f"Deleted {len(old_queries)} old queries")
            return len(old_queries)

        except Exception as e:
            logger.error(f"Error deleting old queries: {e}")
            return 0
