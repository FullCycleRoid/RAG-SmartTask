"""
SQLAlchemy модели для хранения документов
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.core.config import get_settings

settings = get_settings()


class DocumentChunk(Base):
    """Модель фрагмента документа с векторным представлением"""

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_name = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024))
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document={self.document_name}, chunk={self.chunk_index})>"