"""
API маршруты с использованием репозиториев
"""

import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException
from fastapi import Query as QueryParam
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logger import logger
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.common import HealthResponse
from app.schemas.query import (
    DocumentUploadResponse,
    QueryHistory,
    QueryRequest,
    QueryResponse,
)
from app.services.cache import cache_manager
from app.services.llm import llm_service
from app.services.rag import RAGPipeline
from app.services.vector_store import VectorStore
from app.utils.document_processor import DocumentProcessor

settings = get_settings()
router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Обработать вопрос пользователя через RAG pipeline
    """
    try:
        rag = RAGPipeline(db)

        result = await rag.process_question(
            question=request.question, session_id=request.session_id
        )

        result["question"] = request.question
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    """
    Загрузить и обработать документ (PDF, TXT, MD)
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        allowed_extensions = [".pdf", ".txt", ".md"]
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Only {', '.join(allowed_extensions)} files are supported",
            )

        # Сохраняем файл во временную директорию
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Инициализируем репозитории и сервисы
            chunk_repository = DocumentChunkRepository(db)
            vector_store = VectorStore(db)

            processor = DocumentProcessor()
            chunks = processor.process_document(tmp_path)

            if not chunks:
                raise HTTPException(
                    status_code=400, detail="Could not extract text from document"
                )

            # Удаляем старые фрагменты документа через репозиторий
            await vector_store.delete_document_chunks(file.filename)

            # Добавляем новые фрагменты
            for idx, chunk in enumerate(chunks):
                embedding = await llm_service.generate_embedding(chunk)
                await chunk_repository.create_chunk(
                    document_name=file.filename,
                    content=chunk,
                    embedding=embedding,
                    chunk_index=idx,
                )

            await db.commit()

            # Очищаем кэш, так как база знаний изменилась
            await cache_manager.clear()

            logger.info(f"Successfully uploaded document: {file.filename}")

            return DocumentUploadResponse(
                filename=file.filename,
                chunks_created=len(chunks),
                message=f"Document processed successfully. Created {len(chunks)} chunks.",
            )

        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Проверка состояния сервиса
    """
    try:
        # Проверяем БД
        await db.execute(select(1))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"

    # Проверяем Redis
    try:
        if cache_manager.redis:
            await cache_manager.redis.ping()
            redis_status = "healthy"
        else:
            redis_status = "not connected"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "unhealthy"

    # Проверяем векторное хранилище через репозиторий
    try:
        chunk_repository = DocumentChunkRepository(db)
        count = await chunk_repository.get_document_count()
        vector_status = f"healthy ({count} documents)"
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        vector_status = "unhealthy"

    overall_status = (
        "healthy"
        if all(
            [
                db_status == "healthy",
                redis_status == "healthy",
                "healthy" in vector_status,
            ]
        )
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        database=db_status,
        redis=redis_status,
        vector_store=vector_status,
    )


@router.get("/history", response_model=List[QueryHistory])
async def get_history(
    limit: int = QueryParam(10, ge=1, le=100), db: AsyncSession = Depends(get_db)
):
    """
    Получить историю запросов через репозиторий
    """
    try:
        query_repository = QueryRepository(db)
        queries = await query_repository.get_recent_queries(limit=limit)

        return [QueryHistory.model_validate(q) for q in queries]

    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """
    Получить статистику использования
    """
    try:
        rag = RAGPipeline(db)
        stats = await rag.get_statistics()

        return stats

    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(db: AsyncSession = Depends(get_db)):
    """
    Получить список всех загруженных документов
    """
    try:
        chunk_repository = DocumentChunkRepository(db)
        document_names = await chunk_repository.get_unique_document_names()

        documents = []
        for name in document_names:
            chunks_count = await chunk_repository.get_chunks_count_by_document(name)
            documents.append({"name": name, "chunks_count": chunks_count})

        return {"documents": documents, "total": len(documents)}

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_name}")
async def delete_document(document_name: str, db: AsyncSession = Depends(get_db)):
    """
    Удалить документ и все его фрагменты
    """
    try:
        vector_store = VectorStore(db)

        deleted_count = await vector_store.delete_document_chunks(document_name)

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        await db.commit()

        await cache_manager.clear()

        logger.info(f"Deleted document: {document_name}")

        return {
            "message": f"Document '{document_name}' deleted successfully",
            "chunks_deleted": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """
    Очистить кэш ответов
    """
    try:
        await cache_manager.clear()
        return {"message": "Cache cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
