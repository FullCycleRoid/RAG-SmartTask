"""
Сервис для работы с языковыми моделями (Claude API + OpenAI Embeddings) с оптимизацией скорости
"""

import os
import time
from typing import Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from app.core.config import settings
from app.core.logger import logger
from app.services.embeddings import create_openai_embedding_service


class LLMService:
    """
    Сервис для работы с LLM через LangChain и OpenAI эмбеддингами
    """

    def __init__(self, enable_langsmith: bool = False):
        """
        Инициализация сервиса

        Args:
            enable_langsmith: Включить LangSmith трейсинг
        """
        self.llm = ChatAnthropic(
            model=settings.CLAUDE_MODEL,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=settings.MAX_RESPONSE_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            timeout=30,
            max_retries=2,
        )

        # Инициализация OpenAI эмбеддингов
        self.embedding_service = create_openai_embedding_service()
        self.embedding_dimension = self.embedding_service.embedding_dimension

        # LangSmith настройки (опционально)
        self.enable_langsmith = enable_langsmith
        if enable_langsmith:
            self._setup_langsmith()

        logger.info(
            f"✅ LLM Service initialized: "
            f"Claude={settings.CLAUDE_MODEL}, "
            f"Embeddings=OpenAI {settings.OPENAI_EMBEDDING_MODEL} (dim={self.embedding_dimension}), "
            f"LangSmith={'enabled' if enable_langsmith else 'disabled'}"
        )

    def _setup_langsmith(self):
        """Настройка LangSmith для трейсинга"""
        langsmith_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_project = os.getenv("LANGCHAIN_PROJECT", "smarttask-faq")

        if langsmith_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
            logger.info(f"✅ LangSmith tracing enabled: project={langsmith_project}")
        else:
            logger.warning("⚠️ LangSmith API key not found, tracing disabled")
            self.enable_langsmith = False

    @traceable(name="create_chat_completion")
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Создать ответ через Claude API используя LangChain
        """
        try:
            langchain_messages = []

            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))

            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))

            start_time = time.time()
            response = await self.llm.ainvoke(
                langchain_messages, config={"temperature": temperature}
            )
            elapsed_time = time.time() - start_time

            logger.info(f"LLM response generated in {elapsed_time:.2f}s")
            return response.content

        except Exception as e:
            logger.error(f"Error calling Claude API via LangChain: {e}")
            raise

    async def generate_answer(
        self, question: str, context: List[str]
    ) -> Tuple[str, int]:
        """
        Оптимизированная генерация ответа на вопрос с контекстом

        Args:
            question: Вопрос пользователя
            context: Список релевантных фрагментов текста

        Returns:
            Tuple[str, int]: Ответ и количество использованных токенов
        """
        try:
            optimized_context = []
            for i, text in enumerate(context[:2]):
                truncated_text = text[:300] + "..." if len(text) > 300 else text
                optimized_context.append(f"[{i+1}] {truncated_text}")

            context_text = "\n".join(optimized_context)

            system_prompt = """
            Ты - AI-ассистент SmartTask. Отвечай кратко и по делу. 
            Используй только информацию из контекста. Будь точным и лаконичным. Максимум 3-4 предложения.
            """

            prompt = f"""Контекст:
            {context_text}
            
            Вопрос: {question}
            
            ИНСТРУКЦИИ:
            1. Используй ВСЮ релевантную информацию из контекста
            2. Структурируй ответ логично (используй списки, если нужно)
            3. Включи все важные детали (цифры, названия, шаги)
            4. Если в контексте есть дополнительная полезная информация - упомяни её
            5. Если информации недостаточно - честно скажи об этом
            
            Краткий ответ:"""

            messages = [{"role": "user", "content": prompt}]

            start_time = time.time()
            answer = await self.create_chat_completion(
                messages=messages, system_prompt=system_prompt, temperature=0.3
            )
            llm_time = time.time() - start_time

            if len(answer) > 500:
                answer = answer[:497] + "..."

            tokens_used = (
                len(answer) // 3 + len(prompt) // 3
            )  # Более точный расчет токенов для русского языка

            logger.info(f"Answer generated in {llm_time:.2f}s, tokens: {tokens_used}")
            return answer, tokens_used

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return (
                "Извините, не удалось обработать запрос. Пожалуйста, попробуйте переформулировать вопрос.",
                0,
            )

    @traceable(name="generate_embedding")
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Генерировать эмбеддинг для текста используя OpenAI API
        """
        try:
            return await self.embedding_service.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            return [0.0] * self.embedding_dimension

    def get_service_info(self) -> dict:
        """Получить информацию о сервисе"""
        return {
            "llm": {
                "provider": "langchain-anthropic",
                "model": settings.CLAUDE_MODEL,
                "max_tokens": settings.MAX_RESPONSE_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
            },
            "embeddings": self.embedding_service.get_model_info(),
            "langsmith": {
                "enabled": self.enable_langsmith,
                "project": (
                    os.getenv("LANGCHAIN_PROJECT", "smarttask-faq")
                    if self.enable_langsmith
                    else None
                ),
            },
        }


enable_langsmith = os.getenv("ENABLE_LANGSMITH", "false").lower() == "true"

llm_service = LLMService(enable_langsmith=enable_langsmith)
