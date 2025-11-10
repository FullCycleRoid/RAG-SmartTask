"""
LangSmith-based RAG Evaluation System - FIXED VERSION
"""

from datetime import datetime
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.evaluation import aevaluate
from typing_extensions import Annotated, TypedDict

from app.core.config import get_settings
from app.core.logger import logger
from app.services.llm import llm_service
from app.services.rag import RAGPipeline


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class LangSmithEvaluator:
    """LangSmith-based RAG evaluation system"""

    def __init__(self):
        self.settings = get_settings()
        self.client = Client()
        self.dataset_name = "smarttask-faq-eval"

        # Correctness evaluation prompt
        self.correctness_instructions = """You are a teacher grading a quiz about SmartTask FAQ system. 
        You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 
        
        Grade criteria:
        (1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
        (2) Ensure that the student answer does not contain any conflicting statements.
        (3) It is OK if the student answer contains more information than the ground truth answer, 
            as long as it is factually accurate relative to the ground truth answer.

        Correctness:
        - True: Student's answer meets all criteria and is factually correct
        - False: Student's answer contains factual errors or conflicting information

        Explain your reasoning step-by-step. Be strict but fair."""

    async def create_evaluation_dataset(self):
        """Create evaluation dataset in LangSmith"""

        examples = [
            {
                "inputs": {"question": "Как создать задачу в SmartTask?"},
                "outputs": {
                    "answer": "Для создания задачи нажмите кнопку '+ Задача', введите название, описание, установите дедлайн и назначьте исполнителя."
                },
            },
            {
                "inputs": {"question": "Как просмотреть историю задач?"},
                "outputs": {
                    "answer": "Историю задач можно просмотреть в разделе 'Активность' или через фильтр по дате создания."
                },
            },
            {
                "inputs": {"question": "Какие типы задач поддерживает SmartTask?"},
                "outputs": {
                    "answer": "SmartTask поддерживает обычные задачи, повторяющиеся задачи, задачи с чек-листами и групповые задачи."
                },
            },
            {
                "inputs": {"question": "Как настроить уведомления?"},
                "outputs": {
                    "answer": "Уведомления настраиваются в разделе 'Настройки' -> 'Уведомления', где можно выбрать email, push или browser уведомления."
                },
            },
            {
                "inputs": {
                    "question": "Можно ли интегрировать SmartTask с другими системами?"
                },
                "outputs": {
                    "answer": "Да, SmartTask поддерживает интеграцию через REST API и webhooks с популярными системами like Slack, Jira, Google Calendar."
                },
            },
        ]

        # Create dataset if it doesn't exist
        try:
            if not self.client.has_dataset(dataset_name=self.dataset_name):
                dataset = self.client.create_dataset(
                    dataset_name=self.dataset_name,
                    description="SmartTask FAQ evaluation dataset",
                )

                # Add examples to dataset
                for example in examples:
                    self.client.create_example(
                        inputs=example["inputs"],
                        outputs=example["outputs"],
                        dataset_id=dataset.id,
                    )

                logger.info(
                    f"Created dataset: {self.dataset_name} with {len(examples)} examples"
                )
            else:
                logger.info(f"Dataset {self.dataset_name} already exists")
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise

    async def correctness_evaluator(self, run, example) -> dict:
        """Evaluator for RAG answer correctness - FIXED VERSION"""
        try:
            # FIX: Properly access inputs and outputs
            inputs = run.inputs
            outputs = run.outputs
            reference_outputs = example.outputs

            # FIX: Check if keys exist
            question = inputs.get("question", "")
            student_answer = outputs.get("answer", "")
            ground_truth = reference_outputs.get("answer", "")

            if not question or not student_answer or not ground_truth:
                return {
                    "key": "correctness",
                    "score": False,
                    "comment": "Missing question, answer or ground truth",
                }

            answers = f"""\
QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
STUDENT ANSWER: {student_answer}"""

            messages = [
                {"role": "system", "content": self.correctness_instructions},
                {"role": "user", "content": answers},
            ]

            # Use your existing LLM service
            evaluation_response = await llm_service.create_chat_completion(
                messages=messages,
                system_prompt=self.correctness_instructions,
                temperature=0.1,
            )

            # Parse the response - improved logic
            evaluation_lower = evaluation_response.lower()

            if "correct" in evaluation_lower and "true" in evaluation_lower:
                score = True
            elif "correct" in evaluation_lower and "false" in evaluation_lower:
                score = False
            else:
                # Fallback evaluation using keyword matching
                ground_truth_lower = ground_truth.lower()
                student_answer_lower = student_answer.lower()

                key_phrases = [
                    phrase for phrase in ground_truth_lower.split() if len(phrase) > 3
                ]
                if not key_phrases:
                    score = True
                else:
                    matches = sum(
                        1 for phrase in key_phrases if phrase in student_answer_lower
                    )
                    score = matches / len(key_phrases) > 0.6

            return {
                "key": "correctness",
                "score": score,
                "comment": evaluation_response[:500],  # Limit comment length
            }

        except Exception as e:
            logger.error(f"Error in correctness evaluator: {e}")
            return {"key": "correctness", "score": False, "comment": f"Error: {str(e)}"}

    async def relevance_evaluator(self, run, example) -> dict:
        """Simple relevance evaluator - FIXED VERSION"""
        try:
            # FIX: Properly access inputs and outputs
            inputs = run.inputs
            outputs = run.outputs

            question = inputs.get("question", "").lower()
            answer = outputs.get("answer", "").lower()

            if not question or not answer:
                return {
                    "key": "relevance",
                    "score": False,
                    "comment": "Missing question or answer",
                }

            # Check if answer contains question keywords
            question_keywords = [word for word in question.split() if len(word) > 3]
            if not question_keywords:
                score = True
            else:
                matches = sum(1 for keyword in question_keywords if keyword in answer)
                score = matches / len(question_keywords) > 0.5

            return {
                "key": "relevance",
                "score": score,
                "comment": f"Question keywords matched: {score}",
            }

        except Exception as e:
            logger.error(f"Error in relevance evaluator: {e}")
            return {"key": "relevance", "score": False, "comment": f"Error: {str(e)}"}

    async def run_rag_pipeline(self, inputs: dict) -> dict:
        """Run RAG pipeline for evaluation"""
        try:
            from app.core.database import async_session_maker

            async with async_session_maker() as db:
                rag_pipeline = RAGPipeline(db)
                result = await rag_pipeline.process_question(inputs["question"])

                return {
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                    "tokens_used": result.get("tokens_used", 0),
                    "response_time": result.get("response_time", 0),
                    "cached": result.get("cached", False),
                }

        except Exception as e:
            logger.error(f"Error running RAG pipeline: {e}")
            return {"answer": "Error processing question", "sources": []}

    async def run_evaluation(self):
        """Run comprehensive evaluation using LangSmith"""

        # Ensure dataset exists
        await self.create_evaluation_dataset()

        logger.info("Starting LangSmith evaluation...")

        try:
            # Run evaluation using aevaluate for async functions
            experiment_results = await aevaluate(
                self.run_rag_pipeline,
                data=self.dataset_name,
                evaluators=[self.correctness_evaluator, self.relevance_evaluator],
                experiment_prefix="smarttask-rag-eval",
                metadata={
                    "version": "1.0.0",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": get_settings().CLAUDE_MODEL,
                },
                max_concurrency=1,  # Reduced concurrency for stability
            )

            logger.info("Evaluation completed!")
            return experiment_results

        except Exception as e:
            logger.error(f"Error in LangSmith evaluation: {e}")
            raise

    async def get_evaluation_stats(self, experiment_results) -> Dict:
        """Calculate evaluation statistics - FIXED VERSION"""
        try:
            # FIX: Convert to list properly
            results_list = []
            async for result in experiment_results:
                results_list.append(result)

            if not results_list:
                return {
                    "total_examples": 0,
                    "correctness": {"true_count": 0, "false_count": 0, "accuracy": 0},
                    "relevance": {"true_count": 0, "false_count": 0, "accuracy": 0},
                    "timestamp": datetime.utcnow().isoformat(),
                }

            correctness_scores = [
                r
                for r in results_list
                if hasattr(r, "evaluator_name")
                and r.evaluator_name == "correctness_evaluator"
            ]
            relevance_scores = [
                r
                for r in results_list
                if hasattr(r, "evaluator_name")
                and r.evaluator_name == "relevance_evaluator"
            ]

            # Calculate stats
            total_examples = len(correctness_scores)

            correctness_true = sum(
                1 for r in correctness_scores if getattr(r, "score", False) is True
            )
            correctness_false = sum(
                1 for r in correctness_scores if getattr(r, "score", False) is False
            )
            correctness_accuracy = (
                correctness_true / total_examples if total_examples > 0 else 0
            )

            relevance_true = sum(
                1 for r in relevance_scores if getattr(r, "score", False) is True
            )
            relevance_false = sum(
                1 for r in relevance_scores if getattr(r, "score", False) is False
            )
            relevance_accuracy = (
                relevance_true / total_examples if total_examples > 0 else 0
            )

            stats = {
                "total_examples": total_examples,
                "correctness": {
                    "true_count": correctness_true,
                    "false_count": correctness_false,
                    "accuracy": correctness_accuracy,
                },
                "relevance": {
                    "true_count": relevance_true,
                    "false_count": relevance_false,
                    "accuracy": relevance_accuracy,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating evaluation stats: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


# Global evaluator instance
langsmith_evaluator = LangSmithEvaluator()
