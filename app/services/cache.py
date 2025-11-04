"""
Redis кэш для частых вопросов
"""

import hashlib
import json
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logger import logger

settings = get_settings()


class CacheManager:
    """Менеджер кэширования на Redis"""

    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.ttl = settings.CACHE_TTL

    async def connect(self):
        """Подключиться к Redis"""
        try:
            self.redis = await aioredis.from_url(
                settings.REDIS_URL, encoding="utf-8", decode_responses=True
            )
            await self.redis.ping()
            logger.info("✅ Connected to Redis successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis = None

    async def disconnect(self):
        """Отключиться от Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")

    def _get_cache_key(self, question: str) -> str:
        """Генерировать ключ кэша из вопроса"""
        question_lower = question.lower().strip()
        return f"faq:{hashlib.md5(question_lower.encode()).hexdigest()}"

    async def get(self, question: str) -> Optional[dict]:
        """Получить ответ из кэша"""
        if not self.redis:
            return None

        try:
            key = self._get_cache_key(question)
            cached = await self.redis.get(key)

            if cached:
                logger.info(f"Cache HIT for question: {question[:50]}...")
                return json.loads(cached)

            logger.debug(f"Cache MISS for question: {question[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, question: str, answer_data: dict):
        """Сохранить ответ в кэш"""
        if not self.redis:
            return

        try:
            key = self._get_cache_key(question)
            await self.redis.setex(
                key, self.ttl, json.dumps(answer_data, ensure_ascii=False)
            )
            logger.debug(f"Cached answer for question: {question[:50]}...")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def clear(self):
        """Очистить весь кэш"""
        if not self.redis:
            return

        try:
            keys = await self.redis.keys("faq:*")
            if keys:
                await self.redis.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")


cache_manager = CacheManager()
