"""
Базовый репозиторий с общими CRUD операциями
"""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Базовый репозиторий для работы с моделями БД"""

    def __init__(self, model: Type[ModelType], db: AsyncSession):
        """
        Инициализация репозитория

        Args:
            model: Класс модели SQLAlchemy
            db: Асинхронная сессия БД
        """
        self.model = model
        self.db = db

    async def create(self, **kwargs) -> ModelType:
        """
        Создать новую запись

        Args:
            **kwargs: Поля для создания записи

        Returns:
            ModelType: Созданная запись
        """
        instance = self.model(**kwargs)
        self.db.add(instance)
        await self.db.flush()
        await self.db.refresh(instance)
        return instance

    async def get_by_id(self, id: Any) -> Optional[ModelType]:
        """
        Получить запись по ID

        Args:
            id: ID записи

        Returns:
            Optional[ModelType]: Найденная запись или None
        """
        query = select(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[ModelType]:
        """
        Получить все записи

        Args:
            limit: Максимальное количество записей
            offset: Смещение для пагинации

        Returns:
            List[ModelType]: Список записей
        """
        query = select(self.model)

        if limit:
            query = query.limit(limit)
        if offset:
            query = query.offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update(self, id: Any, **kwargs) -> Optional[ModelType]:
        """
        Обновить запись по ID

        Args:
            id: ID записи
            **kwargs: Поля для обновления

        Returns:
            Optional[ModelType]: Обновленная запись или None
        """
        instance = await self.get_by_id(id)
        if not instance:
            return None

        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        await self.db.flush()
        await self.db.refresh(instance)
        return instance

    async def delete(self, id: Any) -> bool:
        """
        Удалить запись по ID

        Args:
            id: ID записи

        Returns:
            bool: True если запись удалена, False если не найдена
        """
        instance = await self.get_by_id(id)
        if not instance:
            return False

        await self.db.delete(instance)
        await self.db.flush()
        return True

    async def count(self) -> int:
        """
        Получить количество записей

        Returns:
            int: Количество записей
        """
        query = select(self.model)
        result = await self.db.execute(query)
        return len(result.scalars().all())

    async def exists(self, **filters) -> bool:
        """
        Проверить существование записи по фильтрам

        Args:
            **filters: Фильтры для поиска

        Returns:
            bool: True если запись существует
        """
        query = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.db.execute(query)
        return result.scalar_one_or_none() is not None

    async def find_one(self, **filters) -> Optional[ModelType]:
        """
        Найти одну запись по фильтрам

        Args:
            **filters: Фильтры для поиска

        Returns:
            Optional[ModelType]: Найденная запись или None
        """
        query = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def find_many(self, **filters) -> List[ModelType]:
        """
        Найти несколько записей по фильтрам

        Args:
            **filters: Фильтры для поиска

        Returns:
            List[ModelType]: Список найденных записей
        """
        query = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.db.execute(query)
        return list(result.scalars().all())
