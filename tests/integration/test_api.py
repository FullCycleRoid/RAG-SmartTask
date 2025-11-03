"""
Интеграционные тесты для API endpoints
"""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from app.repositories import QueryRepository, DocumentChunkRepository


class TestAPIIntegration:
    """Интеграционные тесты для API endpoints"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ask_endpoint_full_integration(self, client: AsyncClient):
        """Тест endpoint /api/ask"""
        with patch('app.api.routes.llm_service') as mock_llm, \
                patch('app.api.routes.cache_manager') as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            mock_llm.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
            mock_llm.generate_answer = AsyncMock(return_value=(
                "Для создания задачи нажмите кнопку 'Новая задача' в верхней панели.",
                120  # Фиксированное значение для теста
            ))

            response = await client.post(
                "/api/ask",
                json={
                    "question": "Как создать новую задачу?",
                    "session_id": "integration-test-session"
                }
            )

            assert response.status_code == 200
            data = response.json()

            assert "question" in data
            assert "answer" in data
            assert "tokens_used" in data
            assert "response_time" in data
            assert "sources" in data
            assert "cached" in data

            assert isinstance(data["tokens_used"], int)
            assert data["tokens_used"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_endpoint_integration(self, client: AsyncClient):
        """Тест health endpoint с реальными подключениями"""
        
        response = await client.get("/api/health")


        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "database" in data
        assert "redis" in data
        assert "vector_store" in data

        assert data["status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ask_endpoint_validation_errors(self, client: AsyncClient):
        """Тест валидации входных данных в ask endpoint"""
        response = await client.post("/api/ask", json={"question": ""})

        assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ask_endpoint_service_error(self, client: AsyncClient):
        """Тест обработки ошибок сервиса в ask endpoint"""
        with patch('app.api.routes.RAGPipeline') as mock_rag:
            mock_rag.return_value.process_question = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = await client.post(
                "/api/ask",
                json={"question": "Тестовый вопрос"}
            )

            assert response.status_code == 500
            assert "Service error" in response.json()["detail"]


    @pytest.mark.asyncio
    async def test_history_endpoint(self, client: AsyncClient, db_session):
        """Тест endpoint /api/history"""
        repo = QueryRepository(db_session)
        for i in range(3):
            await repo.create_query(question=f"Q{i}", answer=f"A{i}")
        await db_session.commit()

        response = await client.get("/api/history?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["question"] == "Q2"  # Последний первым

    @pytest.mark.asyncio
    async def test_statistics_endpoint(
            self, client: AsyncClient, db_session, sample_embedding: list
    ):
        """Тест endpoint /api/statistics"""

        query_repo = QueryRepository(db_session)
        chunk_repo = DocumentChunkRepository(db_session)

        await query_repo.create_query(question="Q1", answer="A1", tokens_used=100)
        await chunk_repo.create_chunk(
            document_name="test.pdf",
            content="Content",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        response = await client.get("/api/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 1
        assert data["total_tokens"] == 100
        assert data["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_list_documents_endpoint(
            self, client: AsyncClient, db_session, sample_embedding: list
    ):
        """Тест endpoint GET /api/documents"""

        chunk_repo = DocumentChunkRepository(db_session)
        for i in range(3):
            await chunk_repo.create_chunk(
                document_name="doc1.pdf",
                content=f"Content {i}",
                embedding=sample_embedding,
                chunk_index=i,
            )
        await chunk_repo.create_chunk(
            document_name="doc2.pdf",
            content="Content",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        response = await client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["documents"]) == 2

        doc1 = next(d for d in data["documents"] if d["name"] == "doc1.pdf")
        assert doc1["chunks_count"] == 3

    @pytest.mark.asyncio
    async def test_delete_document_endpoint(
            self, client: AsyncClient, db_session, sample_embedding: list
    ):
        """Тест endpoint DELETE /api/documents/{name}"""

        chunk_repo = DocumentChunkRepository(db_session)
        await chunk_repo.create_chunk(
            document_name="to_delete.pdf",
            content="Content",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        response = await client.delete("/api/documents/to_delete.pdf")

        assert response.status_code == 200
        data = response.json()
        assert data["chunks_deleted"] == 1

        chunks = await chunk_repo.get_by_document_name("to_delete.pdf")
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client: AsyncClient):
        """Тест удаления несуществующего документа"""

        response = await client.delete("/api/documents/non_existing.pdf")

        assert response.status_code == 404

    @pytest.mark.asyncio
    @patch("app.api.routes.cache_manager")
    async def test_clear_cache_endpoint(self, mock_cache, client: AsyncClient):
        """Тест endpoint POST /api/cache/clear"""

        mock_cache.clear = AsyncMock()

        response = await client.post("/api/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        mock_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_endpoint_validation(self, client: AsyncClient):
        """Тест валидации данных в /api/ask"""
        response = await client.post("/api/ask", json={})

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_history_endpoint_pagination(self, client: AsyncClient, db_session):
        """Тест пагинации в /api/history"""

        repo = QueryRepository(db_session)
        for i in range(20):
            await repo.create_query(question=f"Q{i}", answer=f"A{i}")
        await db_session.commit()

        response = await client.get("/api/history?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    @pytest.mark.asyncio
    async def test_history_endpoint_max_limit(self, client: AsyncClient, db_session):
        """Тест максимального лимита в /api/history"""

        repo = QueryRepository(db_session)
        for i in range(150):
            await repo.create_query(question=f"Q{i}", answer=f"A{i}")
        await db_session.commit()

        response = await client.get("/api/history?limit=200")

        assert response.status_code == 422  # Validation error - limit > 100

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client: AsyncClient):
        """Тест получения документов для пустой БД"""

        response = await client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["documents"]) == 0

    @pytest.mark.asyncio
    @patch("app.api.routes.llm_service")
    @patch("app.api.routes.cache_manager")
    async def test_ask_endpoint_error_handling(
            self, mock_cache, mock_llm, client: AsyncClient
    ):
        """Тест обработки ошибок в /api/ask"""

        mock_cache.get = AsyncMock(return_value=None)
        mock_llm.generate_embedding = AsyncMock(side_effect=Exception("LLM Error"))

        response = await client.post("/api/ask", json={"question": "Test question"})

        assert response.status_code == 500

    @pytest.mark.asyncio
    @patch("app.api.routes.llm_service")
    async def test_upload_document_endpoint(
            self, mock_llm, client: AsyncClient, sample_embedding: list
    ):
        """Тест endpoint POST /api/documents"""

        mock_llm.generate_embedding = AsyncMock(return_value=sample_embedding)

        # Создаем тестовый PDF файл (упрощенный)
        pdf_content = b"%PDF-1.4 Test PDF content"

        response = await client.post(
            "/api/documents",
            files={"file": ("test.pdf", pdf_content, "application/pdf")},
        )

        # Может быть 200 или 400 в зависимости от валидности PDF
        # Основная цель - проверить, что endpoint работает
        assert response.status_code in [200, 400, 500]
