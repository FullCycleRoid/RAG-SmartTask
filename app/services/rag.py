"""
RAG Pipeline
"""

import json
import time
import asyncio
from typing import Dict, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import logger
from app.core.database import async_session_maker
from app.repositories.query_repository import QueryRepository
from app.schemas.query import Source
from app.services.cache import cache_manager
from app.services.llm import llm_service
from app.services.vector_store import VectorStore


class RAGPipeline:
    """RAG пайплайн для обработки вопросов"""

    def __init__(self, db: AsyncSession):
        """
        Инициализация RAG пайплайна

        Args:
            db: Асинхронная сессия БД
        """
        self.db = db
        self.vector_store = VectorStore(db)
        self.query_repository = QueryRepository(db)
        self.llm = llm_service
        self.cache = cache_manager

    async def process_question(self, question: str, session_id: str = None) -> Dict:
        """
        ОПТИМИЗИРОВАННАЯ обработка вопроса пользователя

        Args:
            question: Вопрос пользователя
            session_id: ID сессии пользователя

        Returns:
            Dict: Ответ с метаданными
        """
        start_time = time.time()
        timings = {}

        try:
            # 1. Проверяем кэш
            cache_start = time.time()
            cached_answer = await self.cache.get(question)
            timings['cache_check'] = time.time() - cache_start

            if cached_answer:
                logger.info(f"Cache hit for question: {question[:50]}...")
                return {
                    **json.loads(cached_answer),
                    "cached": True,
                }

            # 2. Генерируем эмбеддинг для вопроса
            embedding_start = time.time()
            logger.info(f"Processing question: {question[:50]}...")
            query_embedding = await self.llm.generate_embedding(question)
            timings['embedding'] = time.time() - embedding_start

            # 3. Ищем релевантные фрагменты документов
            search_start = time.time()
            similar_chunks = await self.vector_store.search_similar(
                query_embedding=query_embedding, limit=3
            )
            timings['search'] = time.time() - search_start

            # 4. Формируем контекст для LLM
            llm_context_start = time.time()
            context = []
            sources = []

            for chunk, similarity in similar_chunks:
                context.append(chunk.content)
                sources.append(Source(
                    document=chunk.document_name,
                    content=chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content,
                    relevance=float(similarity),
                ))
            timings['llm_context'] = time.time() - llm_context_start

            # 5. Генерируем ответ через LLM
            llm_generate_start = time.time()
            answer, tokens_used = await self.llm.generate_answer(question, context)
            timings['llm_generate'] = time.time() - llm_generate_start

            # 6. Вычисляем время ответа
            response_time = round(time.time() - start_time, 2)

            # 7. Сохраняем в БД в фоне
            save_start = time.time()
            asyncio.create_task(
                self._save_to_db_and_cache(
                    question=question,
                    answer=answer,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    sources=sources,
                    session_id=session_id,
                    result={
                        "answer": answer,
                        "sources": [s.model_dump() for s in sources],
                        "tokens_used": tokens_used,
                        "response_time": response_time,
                        "cached": False,
                    }
                )
            )
            timings['save'] = time.time() - save_start

            # 8. Формируем результат
            result = {
                "answer": answer,
                "sources": [s.model_dump() for s in sources],
                "tokens_used": tokens_used,
                "response_time": response_time,
                "cached": False,
            }

            total_time = time.time() - start_time
            timings['total'] = total_time

            # Логируем тайминги
            timing_log = " | ".join([f"{k}: {v:.2f}s" for k, v in timings.items()])
            logger.info(f"Optimized timing - {timing_log}")

            return result

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    async def _save_to_db_and_cache(
        self,
        question: str,
        answer: str,
        tokens_used: int,
        response_time: float,
        sources: list,
        session_id: str,
        result: dict
    ):
        """Сохранение в БД и кэш в фоновом режиме"""
        try:
            async with async_session_maker() as db_session:
                query_repository = QueryRepository(db_session)

                await query_repository.create_query(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    sources=json.dumps([s.model_dump() for s in sources]),
                )

                await db_session.commit()

            await self.cache.set(question, json.dumps(result))

        except Exception as e:
            logger.error(f"Background save error: {e}")

    async def get_statistics(self) -> Dict:
        """Получить статистику использования"""
        try:
            total_queries = await self.query_repository.count()
            all_queries = await self.query_repository.get_all()

            total_tokens = sum(q.tokens_used for q in all_queries)
            total_response_time = sum(q.response_time for q in all_queries)
            avg_response_time = total_response_time / len(all_queries) if all_queries else 0

            document_count = await self.vector_store.get_document_count()

            return {
                "total_queries": total_queries,
                "total_tokens": total_tokens,
                "avg_response_time": round(avg_response_time, 2),
                "documents_count": document_count,
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "total_queries": 0,
                "total_tokens": 0,
                "avg_response_time": 0,
                "documents_count": 0,
            }