"""
Конфигурация pytest и общие фикстуры
"""

import asyncio
from typing import AsyncGenerator

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.core.database import Base, get_db
from app.main import app

# Настройки для тестов
TEST_DATABASE_URL = (
    "postgresql+asyncpg://smarttask:smarttask_password_2024@postgres:5432/smarttask_faq"
)


@pytest.fixture(scope="session")
def event_loop():
    """Создать event loop для всех тестов"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_engine():
    """Создать тестовый движок БД"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    async with engine.begin() as conn:
        # Создаем расширения
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Создать изолированную database сессию для теста"""
    async_session = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Создать тестовый HTTP клиент"""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_question() -> str:
    """Пример вопроса"""
    return "Как создать задачу в SmartTask?"


@pytest.fixture
def sample_answer() -> str:
    """Пример ответа"""
    return "Для создания задачи нажмите кнопку '+ Задача', введите название и назначьте исполнителя."


@pytest.fixture
def sample_embedding() -> list:
    """Пример вектора эмбеддинга"""
    return [0.1] * 1024
