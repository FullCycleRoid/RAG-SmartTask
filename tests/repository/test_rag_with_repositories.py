"""
Интеграционные тесты для RAGPipeline с репозиториями
"""

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag import RAGPipeline


class TestRAGPipelineIntegration:
    """Интеграционные тесты для RAG Pipeline"""

    @pytest.fixture
    async def rag_pipeline(self, db_session: AsyncSession):
        """Создать RAG pipeline"""
        return RAGPipeline(db_session)

    @pytest.mark.asyncio
    @patch("app.services.rag.llm_service")
    @patch("app.services.rag.cache_manager")
    async def test_process_question_full_flow(
        self,
        mock_cache,
        mock_llm,
        rag_pipeline: RAGPipeline,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест полного потока обработки вопроса"""
        
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        mock_llm.generate_embedding = AsyncMock(return_value=sample_embedding)
        mock_llm.generate_answer = AsyncMock(return_value=("Test answer", 100))

        # Создаем тестовый документ
        from app.repositories.document_chunk_repository import \
            DocumentChunkRepository

        chunk_repo = DocumentChunkRepository(db_session)
        await chunk_repo.create_chunk(
            document_name="test.pdf",
            content="Test content for search",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        result = await rag_pipeline.process_question(
            question="Test question", session_id="test-session"
        )

        
        assert result["question"] == "Test question"
        assert result["answer"] == "Test answer"
        assert result["tokens_used"] == 100
        assert result["response_time"] > 0
        assert isinstance(result["sources"], list)

        # Проверяем, что запрос сохранен в БД
        from app.repositories.query_repository import QueryRepository

        query_repo = QueryRepository(db_session)
        queries = await query_repo.get_queries_by_session("test-session")
        assert len(queries) == 1
        assert queries[0].question == "Test question"
        assert queries[0].answer == "Test answer"

    @pytest.mark.asyncio
    @patch("app.services.rag.cache_manager")
    async def test_process_question_cache_hit(
        self, mock_cache, rag_pipeline: RAGPipeline
    ):
        """Тест получения ответа из кэша"""
        
        cached_response = {
            "question": "Cached question",
            "answer": "Cached answer",
            "tokens_used": 0,
            "response_time": 0.0,
            "sources": [],
        }
        mock_cache.get = AsyncMock(return_value=cached_response)

        
        result = await rag_pipeline.process_question("Cached question")

        
        assert result == cached_response
        mock_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_history(
        self, rag_pipeline: RAGPipeline, db_session: AsyncSession
    ):
        """Тест получения истории запросов"""
        
        from app.repositories.query_repository import QueryRepository

        query_repo = QueryRepository(db_session)
        for i in range(5):
            await query_repo.create_query(
                question=f"Question {i}",
                answer=f"Answer {i}",
                session_id="test-session",
            )
        await db_session.commit()

        
        history = await rag_pipeline.get_history(limit=3, session_id="test-session")

        
        assert len(history) == 3
        assert history[0]["question"] == "Question 4"  # Последний первым
        assert history[1]["question"] == "Question 3"
        assert history[2]["question"] == "Question 2"

    @pytest.mark.asyncio
    async def test_get_statistics(
        self,
        rag_pipeline: RAGPipeline,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест получения статистики"""
        
        from app.repositories.document_chunk_repository import \
            DocumentChunkRepository
        from app.repositories.query_repository import QueryRepository

        query_repo = QueryRepository(db_session)
        chunk_repo = DocumentChunkRepository(db_session)

        # Создаем запросы
        await query_repo.create_query(
            question="Q1", answer="A1", tokens_used=100, response_time=1.0
        )
        await query_repo.create_query(
            question="Q2", answer="A2", tokens_used=200, response_time=2.0
        )

        # Создаем документы
        await chunk_repo.create_chunk(
            document_name="doc1.pdf",
            content="Content",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await chunk_repo.create_chunk(
            document_name="doc2.pdf",
            content="Content",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        stats = await rag_pipeline.get_statistics()

        
        assert stats["total_queries"] == 2
        assert stats["total_tokens_used"] == 300
        assert stats["average_response_time"] == 1.5
        assert stats["total_documents"] == 2

    @pytest.mark.asyncio
    @patch("app.services.rag.cache_manager")
    async def test_clear_cache(self, mock_cache, rag_pipeline: RAGPipeline):
        """Тест очистки кэша"""
        
        mock_cache.clear = AsyncMock()

        
        await rag_pipeline.clear_cache()

        
        mock_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.rag.llm_service")
    @patch("app.services.rag.cache_manager")
    async def test_process_question_no_similar_chunks(
        self, mock_cache, mock_llm, rag_pipeline: RAGPipeline, db_session: AsyncSession
    ):
        """Тест обработки вопроса без похожих документов"""
        
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        mock_llm.generate_embedding = AsyncMock(return_value=[0.5] * 384)
        mock_llm.generate_answer = AsyncMock(return_value=("No context answer", 50))

        
        result = await rag_pipeline.process_question("Question without docs")

        
        assert result["answer"] == "No context answer"
        assert result["tokens_used"] == 50
        assert len(result["sources"]) == 0  # Нет похожих документов

    @pytest.mark.asyncio
    @patch("app.services.rag.llm_service")
    @patch("app.services.rag.cache_manager")
    async def test_process_question_with_session_id(
        self,
        mock_cache,
        mock_llm,
        rag_pipeline: RAGPipeline,
        db_session: AsyncSession,
    ):
        """Тест сохранения session_id"""
        
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        mock_llm.generate_embedding = AsyncMock(return_value=[0.1] * 384)
        mock_llm.generate_answer = AsyncMock(return_value=("Answer", 100))

        
        await rag_pipeline.process_question(question="Test", session_id="user-123")

        
        from app.repositories.query_repository import QueryRepository

        query_repo = QueryRepository(db_session)
        queries = await query_repo.get_queries_by_session("user-123")
        assert len(queries) == 1
        assert queries[0].session_id == "user-123"

    @pytest.mark.asyncio
    async def test_get_history(self, rag_pipeline: RAGPipeline, db_session: AsyncSession):
        """Тест получения истории для пустой сессии"""
        
        history = await rag_pipeline.get_history(
            limit=10, session_id="non-existing-session"
        )

        
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_statistics(
            self,
            rag_pipeline: RAGPipeline,
            db_session: AsyncSession,
            sample_embedding: list,
    ):
        """Тест получения статистики"""
        stats = await rag_pipeline.get_statistics()

        assert stats["total_queries"] == 2
        assert stats["total_tokens"] == 300
        assert stats["average_response_time"] == 1.5
        assert stats["total_documents"] == 2


    @pytest.mark.asyncio
    @patch("app.services.rag.llm_service")
    @patch("app.services.rag.cache_manager")
    async def test_process_question_error_handling(
        self, mock_cache, mock_llm, rag_pipeline: RAGPipeline
    ):
        """Тест обработки ошибок"""
        
        mock_cache.get = AsyncMock(return_value=None)
        mock_llm.generate_embedding = AsyncMock(side_effect=Exception("LLM Error"))

        with pytest.raises(Exception) as exc_info:
            await rag_pipeline.process_question("Test question")

        assert "LLM Error" in str(exc_info.value)
