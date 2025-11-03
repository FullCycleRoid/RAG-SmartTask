"""
Тесты репозитория для запросов пользователей
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.query_repository import QueryRepository


class TestQueryRepository:
    """Тесты для QueryRepository"""

    @pytest.fixture
    async def repository(self, db_session: AsyncSession):
        """Создать репозиторий для тестирования"""
        return QueryRepository(db_session)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_and_retrieve_query(
        self,
        repository: QueryRepository,
        db_session: AsyncSession
    ):
        """Тест создания и получения запроса"""
        
        query = await repository.create_query(
            question="Как создать задачу?",
            answer="Нажмите кнопку создания задачи",
            tokens_used=150,
            response_time=1.5,
            session_id="test-session"
        )
        await db_session.commit()

        
        assert query.id is not None
        assert query.question == "Как создать задачу?"
        assert query.answer == "Нажмите кнопку создания задачи"
        assert query.tokens_used == 150
        assert query.response_time == 1.5
        assert query.session_id == "test-session"
        assert query.created_at is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_recent_queries(
        self,
        repository: QueryRepository,
        db_session: AsyncSession
    ):
        """Тест получения последних запросов"""
        
        for i in range(10):
            await repository.create_query(
                question=f"Вопрос {i}",
                answer=f"Ответ {i}"
            )
        await db_session.commit()

        
        recent = await repository.get_recent_queries(limit=5)

        
        assert len(recent) == 5
        # Проверяем порядок (последние первыми)
        assert recent[0].question == "Вопрос 9"
        assert recent[4].question == "Вопрос 5"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_queries_by_session(
        self,
        repository: QueryRepository,
        db_session: AsyncSession
    ):
        """Тест получения запросов по сессии"""
        
        sessions_data = [
            ("session-1", "Q1", "A1"),
            ("session-1", "Q2", "A2"),
            ("session-2", "Q3", "A3")
        ]

        for session_id, question, answer in sessions_data:
            await repository.create_query(
                question=question,
                answer=answer,
                session_id=session_id
            )
        await db_session.commit()

        
        session_queries = await repository.get_queries_by_session("session-1")

        
        assert len(session_queries) == 2
        assert all(q.session_id == "session-1" for q in session_queries)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_queries_statistics(
        self,
        repository: QueryRepository,
        db_session: AsyncSession
    ):
        """Тест статистики запросов"""
        
        test_queries = [
            ("Q1", "A1", 100, 1.0),
            ("Q2", "A2", 200, 2.0),
            ("Q3", "A3", 150, 1.5)
        ]

        for question, answer, tokens, time in test_queries:
            await repository.create_query(
                question=question,
                answer=answer,
                tokens_used=tokens,
                response_time=time
            )
        await db_session.commit()

        total_tokens = await repository.get_total_tokens_used()
        avg_time = await repository.get_average_response_time()

        assert total_tokens == 450
        assert avg_time == pytest.approx(1.5)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_delete_old_queries(
        self,
        repository: QueryRepository,
        db_session: AsyncSession
    ):
        """Тест удаления старых запросов"""
        for i in range(5):
            await repository.create_query(
                question=f"Старый вопрос {i}",
                answer=f"Старый ответ {i}"
            )
        await db_session.commit()

        deleted_count = await repository.delete_old_queries(days=0)
        await db_session.commit()

        
        assert deleted_count == 5

        # Проверяем, что запросы удалены
        remaining = await repository.get_all()
        assert len(remaining) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_queries_by_date_range(
        self,
        repository: QueryRepository,
        db_session: AsyncSession
    ):
        """Тест получения запросов по диапазону дат"""
        
        await repository.create_query(question="Q1", answer="A1")
        await db_session.commit()

        
        end_date = datetime.utcnow() + timedelta(days=1)
        start_date = datetime.utcnow() - timedelta(days=1)

        queries = await repository.get_queries_by_date_range(start_date, end_date)

        
        assert len(queries) == 1
        assert queries[0].question == "Q1"