"""
Unit тесты для кэша
"""

import pytest

from app.services.cache import CacheManager


@pytest.mark.asyncio
async def test_cache_key_generation():
    """Тест генерации ключей кэша"""
    cache = CacheManager()

    # Одинаковые вопросы должны давать одинаковые ключи
    key1 = cache._get_cache_key("Как создать задачу?")
    key2 = cache._get_cache_key("Как создать задачу?")
    assert key1 == key2

    # Регистр не должен влиять
    key3 = cache._get_cache_key("КАК СОЗДАТЬ ЗАДАЧУ?")
    assert key1 == key3

    # Пробелы в начале/конце не должны влиять
    key4 = cache._get_cache_key("  Как создать задачу?  ")
    assert key1 == key4

    # Разные вопросы должны давать разные ключи
    key5 = cache._get_cache_key("Другой вопрос")
    assert key1 != key5


@pytest.mark.asyncio
async def test_cache_operations_without_redis():
    """Тест операций кэша без подключения к Redis"""
    cache = CacheManager()
    # Redis не подключен

    # Get должен вернуть None
    result = await cache.get("тестовый вопрос")
    assert result is None

    # Set не должен падать
    await cache.set("вопрос", {"answer": "ответ"})

    # Clear не должен падать
    await cache.clear()


@pytest.mark.asyncio
async def test_cache_ttl_setting():
    """Тест настройки TTL"""
    cache = CacheManager()
    assert cache.ttl > 0
    assert isinstance(cache.ttl, int)
