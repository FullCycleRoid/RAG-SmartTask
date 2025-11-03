"""
Скрипт загрузки документов в векторную базу данных
Обновленная версия с чанками по 200 символов
"""

import asyncio
import os
import sys
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session_maker
from app.core.logger import logger
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.services.llm import llm_service
from app.utils.document_processor import DocumentProcessor


# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


settings = get_settings()


async def load_document(file_path: str, db: AsyncSession, chunk_repo: DocumentChunkRepository) -> int:
    """
    Загрузить один документ в векторную БД через репозиторий

    Args:
        file_path: Путь к PDF файлу
        db: Сессия БД
        chunk_repo: Репозиторий фрагментов документов

    Returns:
        int: Количество созданных фрагментов
    """
    try:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing document: {file_name}")

        # 1. Обрабатываем PDF
        processor = DocumentProcessor()
        chunks = processor.process_pdf(file_path)

        if not chunks:
            logger.warning(f"No chunks extracted from {file_name}")
            return 0

        logger.info(f"Extracted {len(chunks)} chunks from {file_name} (max {settings.CHUNK_SIZE} chars each)")

        # Проверяем размер чанков
        for i, chunk in enumerate(chunks):
            if len(chunk) > settings.CHUNK_SIZE + 50:
                logger.warning(f"Chunk {i} is too large: {len(chunk)} chars")

        # 2. Удаляем старые фрагменты документа (если есть)
        existing_chunks = await chunk_repo.get_by_document_name(file_name)
        if existing_chunks:
            logger.info(f"Removing {len(existing_chunks)} existing chunks for {file_name}")
            await chunk_repo.delete_by_document_name(file_name)

        # 3. Создаем эмбеддинги и сохраняем через репозиторий
        created_count = 0
        for idx, chunk_content in enumerate(chunks):
            try:
                # Генерируем эмбеддинг
                embedding = await llm_service.generate_embedding(chunk_content)

                # Сохраняем через репозиторий
                await chunk_repo.create_chunk(
                    document_name=file_name,
                    content=chunk_content,
                    embedding=embedding,
                    chunk_index=idx,
                )
                created_count += 1

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing chunk {idx} of {file_name}: {e}")
                continue

        # 4. Коммитим изменения
        await db.commit()

        logger.info(f"Successfully loaded {created_count} chunks from {file_name}")

        logger.info(f"Последние 3 чанка документа {file_name}:")
        for i, chunk_content in enumerate(chunks[-3:]):
            logger.info(f"Чанк {len(chunks) - 3 + i}: {chunk_content[:100]}...")

        return created_count

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        await db.rollback()
        return 0


async def load_all_documents(documents_dir: str = None) -> dict:
    """
    Загрузить все документы из директории

    Args:
        documents_dir: Путь к директории с документами

    Returns:
        dict: Статистика загрузки
    """
    if documents_dir is None:
        documents_dir = settings.DOCUMENTS_DIR

    documents_path = Path(documents_dir)

    if not documents_path.exists():
        logger.error(f"Documents directory not found: {documents_dir}")
        return {"success": False, "error": "Documents directory not found"}

    # Получаем все PDF файлы
    pdf_files = list(documents_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {documents_dir}")
        return {"success": True, "files_processed": 0, "total_chunks": 0}

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Создаем сессию БД
    async with async_session_maker() as db:
        chunk_repo = DocumentChunkRepository(db)

        total_chunks = 0
        processed_files = 0
        failed_files = []

        for pdf_file in pdf_files:
            try:
                chunks_count = await load_document(str(pdf_file), db, chunk_repo)
                if chunks_count > 0:
                    total_chunks += chunks_count
                    processed_files += 1
                    logger.info(f"✅ Successfully processed {pdf_file.name}: {chunks_count} chunks")
                else:
                    failed_files.append(pdf_file.name)
                    logger.error(f"❌ Failed to process {pdf_file.name}: no chunks created")

            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                failed_files.append(pdf_file.name)

    # Итоговая статистика
    result = {
        "success": True,
        "files_found": len(pdf_files),
        "files_processed": processed_files,
        "files_failed": len(failed_files),
        "total_chunks": total_chunks,
        "chunk_size_limit": settings.CHUNK_SIZE,
    }

    if failed_files:
        result["failed_files"] = failed_files

    logger.info("=" * 50)
    logger.info("Loading completed!")
    logger.info(f"Files found: {result['files_found']}")
    logger.info(f"Files processed: {result['files_processed']}")
    logger.info(f"Files failed: {result['files_failed']}")
    logger.info(f"Total chunks created: {result['total_chunks']}")
    logger.info(f"Chunk size limit: {result['chunk_size_limit']} characters")
    logger.info("=" * 50)

    return result


async def get_documents_statistics() -> dict:
    """
    Получить статистику по загруженным документам

    Returns:
        dict: Статистика
    """
    async with async_session_maker() as db:
        chunk_repo = DocumentChunkRepository(db)

        document_names = await chunk_repo.get_unique_document_names()
        total_documents = len(document_names)

        stats = {
            "total_documents": total_documents,
            "documents": [],
        }

        for doc_name in document_names:
            chunks = await chunk_repo.get_by_document_name(doc_name)
            stats["documents"].append({"name": doc_name, "chunks_count": len(chunks)})

    return stats


async def clear_all_documents() -> dict:
    """
    Удалить все документы из векторной БД

    Returns:
        dict: Результат удаления
    """
    async with async_session_maker() as db:
        chunk_repo = DocumentChunkRepository(db)

        document_names = await chunk_repo.get_unique_document_names()
        total_deleted = 0

        for doc_name in document_names:
            deleted = await chunk_repo.delete_by_document_name(doc_name)
            total_deleted += deleted
            logger.info(f"Deleted {deleted} chunks from {doc_name}")

        await db.commit()

    logger.info(f"Total chunks deleted: {total_deleted}")

    return {
        "success": True,
        "documents_deleted": len(document_names),
        "chunks_deleted": total_deleted,
    }


async def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Load documents into vector database")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to documents directory",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show documents statistics",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all documents from database",
    )

    args = parser.parse_args()

    try:
        if args.stats:
            # Показываем статистику
            stats = await get_documents_statistics()
            print("\n" + "=" * 50)
            print("DOCUMENTS STATISTICS")
            print("=" * 50)
            print(f"Total documents: {stats['total_documents']}")
            print(f"Chunk size limit: {settings.CHUNK_SIZE} characters")
            print("\nDocuments:")
            for doc in stats["documents"]:
                print(f"  - {doc['name']}: {doc['chunks_count']} chunks")
            print("=" * 50 + "\n")

        elif args.clear:
            # Очищаем все документы
            print("\nClearing all documents...")
            result = await clear_all_documents()
            print(f"Deleted {result['documents_deleted']} documents")
            print(f"Total chunks deleted: {result['chunks_deleted']}\n")

        else:
            # Загружаем документы
            result = await load_all_documents(args.dir)

            if result["success"]:
                print("\n✅ Documents loaded successfully!")
                print(f"   Files processed: {result['files_processed']}")
                print(f"   Total chunks: {result['total_chunks']}")
                print(f"   Chunk size: max {result['chunk_size_limit']} characters")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())