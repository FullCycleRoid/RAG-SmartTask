"""
Сервис для оценки качества RAG системы
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import logger
from app.repositories.query_repository import QueryRepository
from app.services.llm import llm_service


class RAGEvaluator:
    """Сервис для оценки качества RAG системы"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.query_repo = QueryRepository(db)

    async def evaluate_single_response(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict:
        """
        Оценить один ответ по нескольким метрикам

        Args:
            question: Вопрос пользователя
            answer: Сгенерированный ответ
            context: Контекст из поиска
            ground_truth: Эталонный ответ (опционально)

        Returns:
            Dict: Результаты оценки
        """
        evaluation_prompt = self._create_evaluation_prompt(
            question, answer, context, ground_truth
        )

        try:
            messages = [{"role": "user", "content": evaluation_prompt}]
            evaluation_result = await llm_service.create_chat_completion(
                messages=messages,
                system_prompt="Ты - эксперт по оценке качества ответов AI-систем.",
                temperature=0.1,
            )

            return self._parse_evaluation_result(evaluation_result)

        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return self._get_default_evaluation()

    def _create_evaluation_prompt(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None,
    ) -> str:
        """Создать промпт для оценки"""
        context_text = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(context)])

        prompt = f"""
        Оцени ответ AI-системы на следующий вопрос:

        ВОПРОС: {question}

        КОНТЕКСТ (источники информации):
        {context_text}

        ОТВЕТ СИСТЕМЫ:
        {answer}
        """

        if ground_truth:
            prompt += f"""

        ЭТАЛОННЫЙ ОТВЕТ (ground truth):
        {ground_truth}
            """

        prompt += """

        ОЦЕНИ ПО ШКАЛЕ ОТ 1 ДО 5:

        1. RELEVANCE (Релевантность ответа):
           - Насколько ответ соответствует вопросу?
           - Отвечает ли на поставленный вопрос?

        2. ACCURACY (Точность):
           - Насколько факты в ответе соответствуют контексту?
           - Есть ли фактические ошибки?

        3. COMPLETENESS (Полнота):
           - Ответил ли на все аспекты вопроса?
           - Не упущены ли важные детали?

        4. COHERENCE (Связность):
           - Логично ли построен ответ?
           - Легко ли понять?

        Дай ответ в формате JSON:
        {{
            "relevance": 1-5,
            "accuracy": 1-5,
            "completeness": 1-5,
            "coherence": 1-5,
            "feedback": "Текстовый фидбэк"
        }}
        """

        return prompt

    def _parse_evaluation_result(self, evaluation_text: str) -> Dict:
        """Парсить результат оценки от LLM"""
        try:
            start_idx = evaluation_text.find("{")
            end_idx = evaluation_text.rfind("}") + 1

            if start_idx != -1 and end_idx != 0:
                json_str = evaluation_text[start_idx:end_idx]
                result = json.loads(json_str)

                for key in ["relevance", "accuracy", "completeness", "coherence"]:
                    if key in result:
                        result[key] = max(1, min(5, int(result[key])))

                return result

        except Exception as e:
            logger.error(f"Error parsing evaluation result: {e}")

        return self._get_default_evaluation()

    def _get_default_evaluation(self) -> Dict:
        """Возвращает оценку по умолчанию при ошибке"""
        return {
            "relevance": 3,
            "accuracy": 3,
            "completeness": 3,
            "coherence": 3,
            "feedback": "Ошибка при оценке",
        }

    async def evaluate_retrieval_quality(
        self,
        question: str,
        retrieved_chunks: List[Tuple[str, float]],
    ) -> Dict:
        """
        Оценить качество поиска (retrieval)

        Args:
            question: Вопрос
            retrieved_chunks: Найденные фрагменты с релевантностью

        Returns:
            Dict: Метрики поиска
        """
        if not retrieved_chunks:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_similarity": 0.0,
                "retrieved_count": 0,
            }

        similarities = [score for _, score in retrieved_chunks]
        avg_similarity = sum(similarities) / len(similarities)

        precision = self._estimate_precision(question, retrieved_chunks)

        return {
            "precision": precision,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_similarity": avg_similarity,
            "retrieved_count": len(retrieved_chunks),
        }

    def _estimate_precision(
        self, question: str, retrieved_chunks: List[Tuple[str, float]]
    ) -> float:
        """Оценить precision на основе релевантности"""
        high_similarity_threshold = 0.7
        relevant_count = sum(
            1 for _, score in retrieved_chunks if score >= high_similarity_threshold
        )

        return relevant_count / len(retrieved_chunks) if retrieved_chunks else 0.0

    async def run_comprehensive_evaluation(
        self, test_questions: List[Dict], rag_pipeline
    ) -> Dict:
        """
        Запустить комплексную оценку на наборе тестовых вопросов

        Args:
            test_questions: Список тестовых вопросов
            rag_pipeline: RAG пайплайн для тестирования

        Returns:
            Dict: Результаты оценки
        """
        logger.info(
            f"Starting comprehensive evaluation with {len(test_questions)} questions"
        )

        results = []
        total_metrics = {
            "relevance": 0.0,
            "accuracy": 0.0,
            "completeness": 0.0,
            "coherence": 0.0,
            "response_time": 0.0,
        }

        for i, test_case in enumerate(test_questions):
            question = test_case["question"]
            ground_truth = test_case.get("ground_truth")

            try:
                # Замеряем время ответа
                start_time = asyncio.get_event_loop().time()

                # Получаем ответ от RAG системы
                result = await rag_pipeline.process_question(question)

                response_time = asyncio.get_event_loop().time() - start_time

                # Оцениваем ответ
                evaluation = await self.evaluate_single_response(
                    question=question,
                    answer=result["answer"],
                    context=[source["content"] for source in result["sources"]],
                    ground_truth=ground_truth,
                )

                # Добавляем метрики производительности
                evaluation.update(
                    {
                        "question": question,
                        "response_time": response_time,
                        "tokens_used": result["tokens_used"],
                        "cached": result["cached"],
                    }
                )

                results.append(evaluation)

                # Суммируем метрики для средних значений
                for metric in ["relevance", "accuracy", "completeness", "coherence"]:
                    total_metrics[metric] += evaluation.get(metric, 3)
                total_metrics["response_time"] += response_time

                logger.info(
                    f"Evaluated question {i+1}/{len(test_questions)}: {evaluation}"
                )

            except Exception as e:
                logger.error(f"Error evaluating question '{question}': {e}")
                # Добавляем запись об ошибке
                results.append(
                    {
                        "question": question,
                        "error": str(e),
                        "relevance": 1,
                        "accuracy": 1,
                        "completeness": 1,
                        "coherence": 1,
                        "response_time": 0.0,
                    }
                )

        # Вычисляем средние значения
        avg_metrics = {}
        successful_evaluations = len([r for r in results if "error" not in r])

        for metric, total in total_metrics.items():
            avg_metrics[f"avg_{metric}"] = (
                total / successful_evaluations if successful_evaluations > 0 else 0.0
            )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_questions": len(test_questions),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": len(test_questions) - successful_evaluations,
            "average_metrics": avg_metrics,
            "detailed_results": results,
        }
