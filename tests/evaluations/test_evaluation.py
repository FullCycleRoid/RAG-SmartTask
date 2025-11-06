"""
Unit тесты для RAGEvaluator (ИСПРАВЛЕННАЯ ВЕРСИЯ)
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import llm
from app.services.rag import RAGPipeline
from app.utils.evaluator import RAGEvaluator


class TestRAGEvaluator:
    """Тесты для сервиса оценки RAG системы"""

    @pytest.fixture
    async def evaluator(self, db_session: AsyncSession):
        """Создать экземпляр RAGEvaluator"""
        return RAGEvaluator(db_session)

    @pytest.fixture
    def sample_question(self):
        return "Как создать задачу в SmartTask?"

    @pytest.fixture
    def sample_answer(self):
        return "Для создания задачи нажмите кнопку 'Новая задача'."

    @pytest.fixture
    def sample_context(self):
        return [
            "Для создания новой задачи нажмите 'Создать задачу'.",
            "В форме создания задачи заполните: название, описание.",
        ]

    @pytest.fixture
    def sample_ground_truth(self):
        return "Нажмите 'Новая задача', заполните название."

    @pytest.fixture
    def valid_evaluation_response(self):
        return json.dumps(
            {
                "relevance": 5,
                "accuracy": 4,
                "completeness": 3,
                "coherence": 5,
                "feedback": "Ответ релевантен",
            }
        )

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_single_response_success(
        self,
        mock_llm_service,
        evaluator: RAGEvaluator,
        sample_question,
        sample_answer,
        sample_context,
        valid_evaluation_response,
    ):
        """Тест успешной оценки одного ответа"""
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=valid_evaluation_response
        )

        result = await evaluator.evaluate_single_response(
            question=sample_question, answer=sample_answer, context=sample_context
        )

        # Проверяем что метод был вызван
        mock_llm_service.create_chat_completion.assert_called_once()

        # Проверяем структуру ответа
        assert "relevance" in result
        assert "accuracy" in result
        assert "completeness" in result
        assert "coherence" in result
        assert "feedback" in result
        assert 1 <= result["relevance"] <= 5
        assert 1 <= result["accuracy"] <= 5

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_single_response_with_ground_truth(
        self,
        mock_llm_service,
        evaluator: RAGEvaluator,
        sample_question,
        sample_answer,
        sample_context,
        sample_ground_truth,
        valid_evaluation_response,
    ):
        """Тест оценки с эталонным ответом"""
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=valid_evaluation_response
        )

        result = await evaluator.evaluate_single_response(
            question=sample_question,
            answer=sample_answer,
            context=sample_context,
            ground_truth=sample_ground_truth,
        )

        assert "relevance" in result
        mock_llm_service.create_chat_completion.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_single_response_llm_error(
        self,
        mock_llm_service,
        evaluator: RAGEvaluator,
        sample_question,
        sample_answer,
        sample_context,
    ):
        """Тест обработки ошибки LLM"""
        mock_llm_service.create_chat_completion = AsyncMock(
            side_effect=Exception("LLM service error")
        )

        result = await evaluator.evaluate_single_response(
            question=sample_question, answer=sample_answer, context=sample_context
        )

        # Проверяем что возвращается оценка по умолчанию
        assert result["relevance"] == 3
        assert result["accuracy"] == 3
        assert result["completeness"] == 3
        assert result["coherence"] == 3
        assert "feedback" in result

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_single_response_invalid_json(
        self,
        mock_llm_service,
        evaluator: RAGEvaluator,
        sample_question,
        sample_answer,
        sample_context,
    ):
        """Тест обработки невалидного JSON от LLM"""
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value="Это не JSON, а просто текст"
        )

        result = await evaluator.evaluate_single_response(
            question=sample_question, answer=sample_answer, context=sample_context
        )

        # Проверяем оценку по умолчанию при ошибке парсинга
        assert result["relevance"] == 3
        assert result["accuracy"] == 3
        assert "feedback" in result

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_quality_empty_chunks(
        self, evaluator: RAGEvaluator, sample_question
    ):
        """Тест оценки качества поиска с пустыми результатами"""
        result = await evaluator.evaluate_retrieval_quality(
            question=sample_question, retrieved_chunks=[]
        )

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1_score"] == 0.0
        assert result["avg_similarity"] == 0.0
        assert result["retrieved_count"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_quality_with_chunks(
        self, evaluator: RAGEvaluator, sample_question
    ):
        """Тест оценки качества поиска с результатами"""
        retrieved_chunks = [
            ("Chunk 1 content", 0.8),
            ("Chunk 2 content", 0.6),
            ("Chunk 3 content", 0.9),
        ]

        result = await evaluator.evaluate_retrieval_quality(
            question=sample_question, retrieved_chunks=retrieved_chunks
        )

        assert result["precision"] >= 0.0
        assert 0.0 <= result["avg_similarity"] <= 1.0
        assert result["retrieved_count"] == 3
        expected_avg = (0.8 + 0.6 + 0.9) / 3
        assert result["avg_similarity"] == pytest.approx(expected_avg)

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_quality_high_similarity(
        self, evaluator: RAGEvaluator, sample_question
    ):
        """Тест оценки с высокой релевантностью"""
        retrieved_chunks = [("Chunk 1", 0.95), ("Chunk 2", 0.92), ("Chunk 3", 0.88)]

        result = await evaluator.evaluate_retrieval_quality(
            question=sample_question, retrieved_chunks=retrieved_chunks
        )

        assert result["precision"] >= 0.0
        assert result["avg_similarity"] > 0.8

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_quality_low_similarity(
        self, evaluator: RAGEvaluator, sample_question
    ):
        """Тест оценки с низкой релевантностью"""
        retrieved_chunks = [("Chunk 1", 0.3), ("Chunk 2", 0.4), ("Chunk 3", 0.2)]

        result = await evaluator.evaluate_retrieval_quality(
            question=sample_question, retrieved_chunks=retrieved_chunks
        )

        assert result["precision"] == 0.0
        assert result["avg_similarity"] < 0.5

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_run_comprehensive_evaluation_success(
        self, mock_llm_service, evaluator: RAGEvaluator, db_session: AsyncSession
    ):
        """Тест комплексной оценки системы"""
        mock_rag_pipeline = AsyncMock()
        mock_rag_pipeline.process_question = AsyncMock(
            side_effect=[
                {
                    "answer": "Answer 1",
                    "sources": [{"content": "Source 1"}],
                    "tokens_used": 100,
                    "cached": False,
                },
                {
                    "answer": "Answer 2",
                    "sources": [{"content": "Source 2"}],
                    "tokens_used": 150,
                    "cached": True,
                },
            ]
        )

        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 4,
                    "accuracy": 5,
                    "completeness": 3,
                    "coherence": 4,
                    "feedback": "Good",
                }
            )
        )

        test_questions = [
            {"question": "Question 1", "ground_truth": "Expected answer 1"},
            {"question": "Question 2", "ground_truth": "Expected answer 2"},
        ]

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=test_questions, rag_pipeline=mock_rag_pipeline
        )

        assert results["total_questions"] == 2
        assert results["successful_evaluations"] == 2
        assert "average_metrics" in results
        assert "detailed_results" in results
        assert len(results["detailed_results"]) == 2

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_run_comprehensive_evaluation_with_errors(
        self, mock_llm_service, evaluator: RAGEvaluator, db_session: AsyncSession
    ):
        """Тест комплексной оценки с ошибками"""
        mock_rag_pipeline = AsyncMock()
        mock_rag_pipeline.process_question = AsyncMock(
            side_effect=[
                Exception("RAG pipeline error"),
                {
                    "answer": "Successful answer",
                    "sources": [{"content": "Source"}],
                    "tokens_used": 100,
                    "cached": False,
                },
            ]
        )

        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 4,
                    "accuracy": 5,
                    "completeness": 3,
                    "coherence": 4,
                    "feedback": "Good",
                }
            )
        )

        test_questions = [
            {"question": "Failing question", "ground_truth": "Expected answer"},
            {"question": "Successful question", "ground_truth": "Expected answer"},
        ]

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=test_questions, rag_pipeline=mock_rag_pipeline
        )

        assert results["total_questions"] == 2
        assert results["successful_evaluations"] == 1
        assert len(results["detailed_results"]) == 2

        # Проверяем что есть запись об ошибке
        error_results = [r for r in results["detailed_results"] if "error" in r]
        assert len(error_results) == 1

    @pytest.mark.asyncio
    async def test_run_comprehensive_evaluation_empty_questions(
        self, evaluator: RAGEvaluator, db_session: AsyncSession
    ):
        """Тест комплексной оценки с пустым списком вопросов"""
        mock_rag_pipeline = AsyncMock()

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=[], rag_pipeline=mock_rag_pipeline
        )

        assert results["total_questions"] == 0
        assert results["successful_evaluations"] == 0
        assert "average_metrics" in results

    def test_parse_evaluation_result_valid_json(self, evaluator: RAGEvaluator):
        """Тест парсинга валидного JSON ответа"""
        valid_json = '{"relevance": 5, "accuracy": 4, "completeness": 3, "coherence": 5, "feedback": "Good"}'

        result = evaluator._parse_evaluation_result(valid_json)

        assert result["relevance"] == 5
        assert result["accuracy"] == 4
        assert result["completeness"] == 3
        assert result["coherence"] == 5
        assert result["feedback"] == "Good"

    def test_parse_evaluation_result_invalid_json(self, evaluator: RAGEvaluator):
        """Тест парсинга невалидного JSON"""
        invalid_json = "Not a JSON {relevance: 5}"

        result = evaluator._parse_evaluation_result(invalid_json)

        # Должен вернуться результат по умолчанию
        assert result["relevance"] == 3
        assert result["accuracy"] == 3
        assert "Ошибка при оценке" in result["feedback"]

    def test_parse_evaluation_result_json_with_text(self, evaluator: RAGEvaluator):
        """Тест парсинга JSON в тексте"""
        text_with_json = """
        Вот мой анализ:
        {
            "relevance": 4,
            "accuracy": 3, 
            "completeness": 5,
            "coherence": 4,
            "feedback": "Отличный ответ"
        }
        Это конец анализа.
        """

        result = evaluator._parse_evaluation_result(text_with_json)

        assert result["relevance"] == 4
        assert result["accuracy"] == 3
        assert result["completeness"] == 5
        assert result["coherence"] == 4

    def test_parse_evaluation_result_out_of_range_values(self, evaluator: RAGEvaluator):
        """Тест парсинга с значениями вне диапазона"""
        json_with_out_of_range = json.dumps(
            {
                "relevance": 10,
                "accuracy": -1,
                "completeness": 3,
                "coherence": 7,
                "feedback": "Values out of range",
            }
        )

        result = evaluator._parse_evaluation_result(json_with_out_of_range)

        # Значения должны быть ограничены диапазоном 1-5
        assert result["relevance"] == 5  # Clamped to max
        assert result["accuracy"] == 1  # Clamped to min
        assert result["coherence"] == 5  # Clamped to max

    def test_get_default_evaluation(self, evaluator: RAGEvaluator):
        """Тест получения оценки по умолчанию"""
        result = evaluator._get_default_evaluation()

        assert result["relevance"] == 3
        assert result["accuracy"] == 3
        assert result["completeness"] == 3
        assert result["coherence"] == 3
        assert "Ошибка при оценке" in result["feedback"]

    def test_create_evaluation_prompt_without_ground_truth(
        self, evaluator: RAGEvaluator, sample_question, sample_answer, sample_context
    ):
        """Тест создания промпта без эталонного ответа"""
        prompt = evaluator._create_evaluation_prompt(
            question=sample_question, answer=sample_answer, context=sample_context
        )

        assert sample_question in prompt
        assert sample_answer in prompt
        for ctx in sample_context:
            assert ctx in prompt
        assert "ЭТАЛОННЫЙ ОТВЕТ" not in prompt

    def test_create_evaluation_prompt_with_ground_truth(
        self,
        evaluator: RAGEvaluator,
        sample_question,
        sample_answer,
        sample_context,
        sample_ground_truth,
    ):
        """Тест создания промпта с эталонным ответом"""
        prompt = evaluator._create_evaluation_prompt(
            question=sample_question,
            answer=sample_answer,
            context=sample_context,
            ground_truth=sample_ground_truth,
        )

        assert sample_question in prompt
        assert sample_answer in prompt
        assert sample_ground_truth in prompt
        assert "ЭТАЛОННЫЙ ОТВЕТ" in prompt

    @pytest.mark.asyncio
    async def test_evaluation_prompt_integration(
        self, evaluator: RAGEvaluator, sample_question, sample_answer, sample_context
    ):
        """Интеграционный тест создания промпта"""
        prompt = evaluator._create_evaluation_prompt(
            question=sample_question, answer=sample_answer, context=sample_context
        )

        assert "ВОПРОС:" in prompt
        assert "КОНТЕКСТ" in prompt
        assert "ОТВЕТ СИСТЕМЫ:" in prompt
        assert "RELEVANCE" in prompt


class TestRAGEvaluatorIntegration:
    """Интеграционные тесты для RAGEvaluator"""

    @pytest.fixture
    async def evaluator(self, db_session: AsyncSession):
        return RAGEvaluator(db_session)

    @pytest.fixture
    async def rag_pipeline(self, db_session: AsyncSession):
        return RAGPipeline(db_session)

    @pytest.mark.asyncio
    @patch("app.services.llm.llm_service")
    async def test_end_to_end_evaluation_flow(
        self,
        mock_llm_service,
        evaluator: RAGEvaluator,
        rag_pipeline: RAGPipeline,
        db_session: AsyncSession,
    ):
        """Тест полного цикла оценки"""
        # Используем правильную размерность 1536
        mock_llm_service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
        mock_llm_service.generate_answer = AsyncMock(return_value=("Test answer", 100))
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 5,
                    "accuracy": 4,
                    "completeness": 5,
                    "coherence": 4,
                    "feedback": "Excellent",
                }
            )
        )

        from app.repositories.document_chunk_repository import DocumentChunkRepository

        chunk_repo = DocumentChunkRepository(db_session)

        # Создаем чанк с правильной размерностью
        await chunk_repo.create_chunk(
            document_name="test.pdf",
            content="Test content for evaluation",
            embedding=[0.1] * 1536,  # 1536 измерения
            chunk_index=0,
        )
        await db_session.commit()

        test_questions = [
            {
                "question": "Test question for evaluation",
                "ground_truth": "Expected answer for test question",
            }
        ]

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=test_questions, rag_pipeline=rag_pipeline
        )

        assert results["total_questions"] == 1
        assert results["successful_evaluations"] == 1
        assert len(results["detailed_results"]) == 1

    @pytest.mark.asyncio
    @patch("app.services.llm.llm_service")
    async def test_evaluation_with_real_rag_pipeline(
        self,
        mock_llm_service,
        evaluator: RAGEvaluator,
        rag_pipeline: RAGPipeline,
        db_session: AsyncSession,
    ):
        """Тест оценки с реальным RAG pipeline"""
        mock_llm_service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
        mock_llm_service.generate_answer = AsyncMock(return_value=("Real answer", 120))
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 4,
                    "accuracy": 4,
                    "completeness": 3,
                    "coherence": 5,
                    "feedback": "Good",
                }
            )
        )

        from app.repositories.document_chunk_repository import DocumentChunkRepository

        chunk_repo = DocumentChunkRepository(db_session)

        await chunk_repo.create_chunk(
            document_name="integration_test.pdf",
            content="Content for integration testing",
            embedding=[0.1] * 1536,  # 1536 измерения
            chunk_index=0,
        )
        await db_session.commit()

        test_questions = [
            {
                "question": "Integration test question",
                "ground_truth": "Expected integration answer",
            }
        ]

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=test_questions, rag_pipeline=rag_pipeline
        )

        assert results["successful_evaluations"] == 1

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluation_metrics_calculation(
        self, mock_llm_service, evaluator: RAGEvaluator, db_session: AsyncSession
    ):
        """Тест расчета метрик оценки"""
        mock_rag_pipeline = AsyncMock()
        mock_rag_pipeline.process_question = AsyncMock(
            return_value={
                "answer": "Test answer with more content for better evaluation",
                "sources": [{"content": "Test source with meaningful content"}],
                "tokens_used": 100,
                "cached": False,
            }
        )

        # Используем более реалистичные оценки
        mock_llm_service.create_chat_completion = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "relevance": 5,
                        "accuracy": 5,
                        "completeness": 5,
                        "coherence": 5,
                        "feedback": "Perfect",
                    }
                ),
                json.dumps(
                    {
                        "relevance": 3,
                        "accuracy": 3,
                        "completeness": 3,
                        "coherence": 3,
                        "feedback": "Average",
                    }
                ),
                json.dumps(
                    {
                        "relevance": 4,
                        "accuracy": 4,
                        "completeness": 4,
                        "coherence": 4,
                        "feedback": "Good",
                    }
                ),
            ]
        )

        test_questions = [
            {
                "question": "How to create a task?",
                "ground_truth": "Click the create button",
            },
            {
                "question": "How to delete a task?",
                "ground_truth": "Use the delete option",
            },
            {"question": "How to edit a task?", "ground_truth": "Click edit button"},
        ]

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=test_questions, rag_pipeline=mock_rag_pipeline
        )

        avg_metrics = results["average_metrics"]
        # Проверяем что средние значения в разумных пределах
        assert 3.0 <= avg_metrics["avg_relevance"] <= 5.0
        assert 3.0 <= avg_metrics["avg_accuracy"] <= 5.0
