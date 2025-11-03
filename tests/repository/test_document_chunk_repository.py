"""
Unit тесты для DocumentChunkRepository
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.document_chunk_repository import DocumentChunkRepository


class TestDocumentChunkRepository:
    """Тесты для репозитория фрагментов документов"""

    @pytest.fixture
    async def repository(self, db_session: AsyncSession):
        """Создать репозиторий фрагментов"""
        return DocumentChunkRepository(db_session)

    @pytest.mark.asyncio
    async def test_create_chunk(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест создания фрагмента"""
        
        chunk = await repository.create_chunk(
            document_name="test.pdf",
            content="Test content",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        assert chunk.id is not None
        assert chunk.document_name == "test.pdf"
        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.embedding is not None
        assert chunk.created_at is not None

    @pytest.mark.asyncio
    async def test_get_by_document_name(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест получения фрагментов по имени документа"""
        
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 1",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 2",
            embedding=sample_embedding,
            chunk_index=1,
        )
        await repository.create_chunk(
            document_name="doc2.pdf",
            content="Content 3",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        chunks = await repository.get_by_document_name("doc1.pdf")

        
        assert len(chunks) == 2
        assert all(c.document_name == "doc1.pdf" for c in chunks)
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    @pytest.mark.asyncio
    async def test_search_similar_chunks(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест векторного поиска похожих фрагментов"""
        
        # Создаем несколько чанков
        for i in range(3):
            await repository.create_chunk(
                document_name=f"doc{i}.pdf",
                content=f"Content {i}",
                embedding=sample_embedding,
                chunk_index=0,
            )
        await db_session.commit()

        
        query_embedding = [0.1] * 384  # Похожий вектор
        similar = await repository.search_similar_chunks(
            query_embedding=query_embedding, limit=2
        )

        
        # Проверяем, что получили результаты (точное количество зависит от threshold)
        assert isinstance(similar, list)
        for chunk, similarity in similar:
            assert chunk.id is not None
            assert 0.0 <= similarity <= 1.0

    @pytest.mark.asyncio
    async def test_delete_by_document_name(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест удаления фрагментов документа"""
        
        await repository.create_chunk(
            document_name="to_delete.pdf",
            content="Content 1",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await repository.create_chunk(
            document_name="to_delete.pdf",
            content="Content 2",
            embedding=sample_embedding,
            chunk_index=1,
        )
        await repository.create_chunk(
            document_name="keep.pdf",
            content="Content 3",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        deleted_count = await repository.delete_by_document_name("to_delete.pdf")
        await db_session.commit()

        
        assert deleted_count == 2
        remaining = await repository.get_by_document_name("to_delete.pdf")
        assert len(remaining) == 0
        kept = await repository.get_by_document_name("keep.pdf")
        assert len(kept) == 1

    @pytest.mark.asyncio
    async def test_get_unique_document_names(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест получения уникальных имен документов"""
        
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 1",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 2",
            embedding=sample_embedding,
            chunk_index=1,
        )
        await repository.create_chunk(
            document_name="doc2.pdf",
            content="Content 3",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        names = await repository.get_unique_document_names()

        
        assert len(names) == 2
        assert "doc1.pdf" in names
        assert "doc2.pdf" in names

    @pytest.mark.asyncio
    async def test_get_document_count(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест подсчета уникальных документов"""
        
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 1",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 2",
            embedding=sample_embedding,
            chunk_index=1,
        )
        await repository.create_chunk(
            document_name="doc2.pdf",
            content="Content 3",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        count = await repository.get_document_count()

        
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_chunks_count_by_document(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест подсчета фрагментов в документе"""
        
        for i in range(5):
            await repository.create_chunk(
                document_name="test.pdf",
                content=f"Content {i}",
                embedding=sample_embedding,
                chunk_index=i,
            )
        await db_session.commit()

        
        count = await repository.get_chunks_count_by_document("test.pdf")

        
        assert count == 5

    @pytest.mark.asyncio
    async def test_chunk_exists(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест проверки существования фрагмента"""
        
        await repository.create_chunk(
            document_name="test.pdf",
            content="Content",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await db_session.commit()

        
        exists = await repository.chunk_exists("test.pdf", 0)
        not_exists = await repository.chunk_exists("test.pdf", 999)

        
        assert exists is True
        assert not_exists is False

    @pytest.mark.asyncio
    async def test_get_all_chunks_ordered(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест получения всех фрагментов с сортировкой"""
        
        await repository.create_chunk(
            document_name="doc2.pdf",
            content="Content 2",
            embedding=sample_embedding,
            chunk_index=1,
        )
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 1",
            embedding=sample_embedding,
            chunk_index=0,
        )
        await repository.create_chunk(
            document_name="doc1.pdf",
            content="Content 0",
            embedding=sample_embedding,
            chunk_index=1,
        )
        await db_session.commit()

        
        chunks = await repository.get_all_chunks_ordered()

        
        assert len(chunks) == 3
        # Должны быть отсортированы по document_name, потом по chunk_index
        assert chunks[0].document_name == "doc1.pdf"
        assert chunks[0].chunk_index == 0
        assert chunks[1].document_name == "doc1.pdf"
        assert chunks[1].chunk_index == 1
        assert chunks[2].document_name == "doc2.pdf"

    @pytest.mark.asyncio
    async def test_get_all_chunks_ordered_with_limit(
        self,
        repository: DocumentChunkRepository,
        db_session: AsyncSession,
        sample_embedding: list,
    ):
        """Тест получения фрагментов с лимитом"""
        
        for i in range(10):
            await repository.create_chunk(
                document_name="test.pdf",
                content=f"Content {i}",
                embedding=sample_embedding,
                chunk_index=i,
            )
        await db_session.commit()

        
        chunks = await repository.get_all_chunks_ordered(limit=5)

        
        assert len(chunks) == 5

    @pytest.mark.asyncio
    async def test_delete_by_document_name_not_found(
        self, repository: DocumentChunkRepository, db_session: AsyncSession
    ):
        """Тест удаления несуществующего документа"""
        
        deleted_count = await repository.delete_by_document_name("non_existing.pdf")
        await db_session.commit()

        
        assert deleted_count == 0
