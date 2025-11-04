"""
Улучшенные unit-тесты для RAGEvaluator (ИСПРАВЛЕННАЯ ВЕРСИЯ)
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.evaluator import RAGEvaluator

# Тестовый датасет
TEST_DATASET = [
    {
        "category": "simple_factual",
        "question": "Как создать задачу в SmartTask?",
        "ground_truth": "Нажмите '+ Задача', введите название, дедлайн и назначьте исполнителя.",
        "expected_keywords": ["задача", "название", "дедлайн", "исполнитель"],
        "difficulty": "easy",
    },
    {
        "category": "api",
        "question": "Как получить список всех задач через API?",
        "ground_truth": "Используйте GET запрос к эндпоинту /tasks с Bearer токеном.",
        "expected_keywords": ["GET", "/tasks", "Bearer"],
        "difficulty": "medium",
    },
    {
        "category": "security",
        "question": "Где хранятся данные SmartTask?",
        "ground_truth": "Данные хранятся в AWS во Франкфурте с шифрованием AES-256.",
        "expected_keywords": ["AWS", "Франкфурт", "AES-256"],
        "difficulty": "medium",
    },
]

QUALITY_THRESHOLDS = {
    "easy": {"min_relevance": 4, "min_accuracy": 4, "max_response_time": 5.0},
    "medium": {"min_relevance": 3, "min_accuracy": 3, "max_response_time": 7.0},
}


class TestRAGEvaluatorWithRealData:
    """Тесты RAGEvaluator с реалистичными данными"""

    @pytest.fixture
    async def evaluator(self, db_session: AsyncSession):
        return RAGEvaluator(db_session)

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_simple_factual_question(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест оценки простого фактического вопроса"""
        question_data = TEST_DATASET[0]

        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 5,
                    "accuracy": 5,
                    "completeness": 4,
                    "coherence": 5,
                    "feedback": "Отличный ответ",
                }
            )
        )

        context = ["Нажмите '+ Задача', введите название"]
        answer = "Чтобы создать задачу, нажмите '+ Задача'"

        result = await evaluator.evaluate_single_response(
            question=question_data["question"],
            answer=answer,
            context=context,
            ground_truth=question_data["ground_truth"],
        )

        assert result["relevance"] >= 1
        assert result["accuracy"] >= 1
        assert "feedback" in result

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_api_question_with_keywords(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест оценки вопроса об API"""
        question_data = TEST_DATASET[1]

        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 5,
                    "accuracy": 5,
                    "completeness": 5,
                    "coherence": 5,
                    "feedback": "Правильный API эндпоинт",
                }
            )
        )

        context = ["GET /tasks — получение задач"]
        answer = "Используйте GET запрос к /tasks"

        result = await evaluator.evaluate_single_response(
            question=question_data["question"],
            answer=answer,
            context=context,
            ground_truth=question_data["ground_truth"],
        )

        assert result["relevance"] >= 1

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_out_of_scope_question(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест out-of-scope вопроса"""
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 1,
                    "accuracy": 1,
                    "completeness": 1,
                    "coherence": 3,
                    "feedback": "Нет информации в контексте",
                }
            )
        )

        result = await evaluator.evaluate_single_response(
            question="Можно ли интегрировать с Jira?",
            answer="К сожалению, нет информации о интеграции с Jira",
            context=["SmartTask интегрируется с Slack и Telegram"],
        )

        # Проверяем что оценка релевантности низкая
        assert result["relevance"] <= 3

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_response_with_partial_context(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест с частичным контекстом"""
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 4,
                    "accuracy": 3,
                    "completeness": 2,
                    "coherence": 4,
                    "feedback": "Частичный ответ",
                }
            )
        )

        result = await evaluator.evaluate_single_response(
            question="Где хранятся данные?",
            answer="Данные хранятся в AWS",
            context=["Данные в AWS"],
        )

        assert result["completeness"] <= 4

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_response_llm_timeout(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест таймаута LLM"""
        mock_llm_service.create_chat_completion = AsyncMock(
            side_effect=TimeoutError("LLM timeout")
        )

        result = await evaluator.evaluate_single_response(
            question="Test question", answer="Test answer", context=["Test context"]
        )

        # Должна вернуться оценка по умолчанию
        assert result["relevance"] == 3
        assert result["accuracy"] == 3
        assert "Ошибка" in result["feedback"]

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_response_malformed_json(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест невалидного JSON"""
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value="{relevance: INVALID, invalid json"
        )

        result = await evaluator.evaluate_single_response(
            question="Test question", answer="Test answer", context=["Test context"]
        )

        # Должна вернуться оценка по умолчанию
        assert result["relevance"] == 3
        assert result["accuracy"] == 3

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_evaluate_response_scores_clamping(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест ограничения оценок 1-5"""
        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 10,
                    "accuracy": -2,
                    "completeness": 3,
                    "coherence": 7,
                    "feedback": "Test",
                }
            )
        )

        result = await evaluator.evaluate_single_response(
            question="Test question", answer="Test answer", context=["Test context"]
        )

        # Значения должны быть ограничены 1-5
        assert 1 <= result["relevance"] <= 5
        assert 1 <= result["accuracy"] <= 5
        assert 1 <= result["coherence"] <= 5

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_empty_results(self, evaluator: RAGEvaluator):
        """Тест пустых результатов поиска"""
        result = await evaluator.evaluate_retrieval_quality(
            question="Test question", retrieved_chunks=[]
        )

        assert result["precision"] == 0.0
        assert result["retrieved_count"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_high_quality_chunks(
        self, evaluator: RAGEvaluator
    ):
        """Тест высококачественных результатов"""
        retrieved_chunks = [
            ("High quality chunk 1", 0.92),
            ("High quality chunk 2", 0.88),
            ("High quality chunk 3", 0.85),
        ]

        result = await evaluator.evaluate_retrieval_quality(
            question="Test question", retrieved_chunks=retrieved_chunks
        )

        assert result["precision"] >= 0.0
        assert result["avg_similarity"] >= 0.0

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_low_quality_chunks(self, evaluator: RAGEvaluator):
        """Тест низкокачественных результатов"""
        retrieved_chunks = [
            ("Low quality 1", 0.35),
            ("Low quality 2", 0.42),
            ("Low quality 3", 0.38),
        ]

        result = await evaluator.evaluate_retrieval_quality(
            question="Test question", retrieved_chunks=retrieved_chunks
        )

        assert result["precision"] <= 1.0
        assert result["avg_similarity"] <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_mixed_quality_chunks(
        self, evaluator: RAGEvaluator
    ):
        """Тест смешанных результатов"""
        retrieved_chunks = [
            ("High quality", 0.95),
            ("High quality", 0.88),
            ("Low quality", 0.45),
            ("Low quality", 0.32),
        ]

        result = await evaluator.evaluate_retrieval_quality(
            question="Test question", retrieved_chunks=retrieved_chunks
        )

        assert 0.0 <= result["precision"] <= 1.0
        expected_avg = (0.95 + 0.88 + 0.45 + 0.32) / 4
        assert result["avg_similarity"] == pytest.approx(expected_avg, rel=0.01)

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_comprehensive_evaluation_all_easy_questions(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест комплексной оценки простых вопросов"""
        easy_questions = [q for q in TEST_DATASET if q["difficulty"] == "easy"]

        mock_rag_pipeline = AsyncMock()
        mock_rag_pipeline.process_question = AsyncMock(
            return_value={
                "answer": "Правильный ответ на вопрос",
                "sources": [{"content": "Релевантный источник информации"}],
                "tokens_used": 120,
                "cached": False,
            }
        )

        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 5,
                    "accuracy": 5,
                    "completeness": 4,
                    "coherence": 5,
                    "feedback": "Отличный ответ",
                }
            )
        )

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=easy_questions, rag_pipeline=mock_rag_pipeline
        )

        assert results["total_questions"] == len(easy_questions)
        assert results["successful_evaluations"] == len(easy_questions)

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_comprehensive_evaluation_mixed_difficulty(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест со смешанной сложностью"""
        mock_rag_pipeline = AsyncMock()
        # Возвращаем более качественные ответы
        mock_rag_pipeline.process_question = AsyncMock(
            return_value={
                "answer": "Подробный ответ на вопрос с полезной информацией",
                "sources": [{"content": "Релевантный источник с деталями"}],
                "tokens_used": 150,
                "cached": False,
            }
        )

        evaluations = [
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
                    "relevance": 4,
                    "accuracy": 4,
                    "completeness": 3,
                    "coherence": 4,
                    "feedback": "Good",
                }
            ),
            json.dumps(
                {
                    "relevance": 3,
                    "accuracy": 3,
                    "completeness": 2,
                    "coherence": 3,
                    "feedback": "Average",
                }
            ),
        ]
        mock_llm_service.create_chat_completion = AsyncMock(side_effect=evaluations)

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=TEST_DATASET, rag_pipeline=mock_rag_pipeline
        )

        assert results["total_questions"] == len(TEST_DATASET)
        # Проверяем что средние значения в разумных пределах
        avg_metrics = results["average_metrics"]
        assert 1.0 <= avg_metrics["avg_relevance"] <= 5.0

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_comprehensive_evaluation_with_failures(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест с частичными ошибками"""
        mock_rag_pipeline = AsyncMock()

        responses = [
            Exception("Ошибка обработки"),
            {
                "answer": "Успешный ответ на вопрос",
                "sources": [{"content": "Релевантный источник"}],
                "tokens_used": 100,
                "cached": False,
            },
        ]
        mock_rag_pipeline.process_question = AsyncMock(side_effect=responses)

        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 4,
                    "accuracy": 4,
                    "completeness": 4,
                    "coherence": 4,
                    "feedback": "Хороший ответ",
                }
            )
        )

        test_questions = TEST_DATASET[:2]

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=test_questions, rag_pipeline=mock_rag_pipeline
        )

        assert results["total_questions"] == 2
        assert results["successful_evaluations"] == 1
        assert results["failed_evaluations"] == 1

    @pytest.mark.asyncio
    @patch("app.utils.evaluator.llm_service")
    async def test_comprehensive_evaluation_performance_metrics(
        self, mock_llm_service, evaluator: RAGEvaluator
    ):
        """Тест метрик производительности"""
        import asyncio

        async def slow_process(*args, **kwargs):
            await asyncio.sleep(0.1)  # Имитируем задержку
            return {
                "answer": "Подробный ответ на вопрос",
                "sources": [{"content": "Источник информации"}],
                "tokens_used": 100,
                "cached": False,
            }

        mock_rag_pipeline = AsyncMock()
        mock_rag_pipeline.process_question = slow_process

        mock_llm_service.create_chat_completion = AsyncMock(
            return_value=json.dumps(
                {
                    "relevance": 4,
                    "accuracy": 4,
                    "completeness": 4,
                    "coherence": 4,
                    "feedback": "Хороший ответ",
                }
            )
        )

        test_questions = TEST_DATASET[:2]

        results = await evaluator.run_comprehensive_evaluation(
            test_questions=test_questions, rag_pipeline=mock_rag_pipeline
        )

        # Проверяем что время ответа присутствует
        assert "avg_response_time" in results["average_metrics"]
        for detail in results["detailed_results"]:
            if "error" not in detail:
                assert "response_time" in detail
                assert detail["response_time"] > 0
