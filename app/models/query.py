"""
SQLAlchemy модели для хранения запросов
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.core.database import Base


class Query(Base):
    """Модель запроса пользователя"""

    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True, nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    tokens_used = Column(Integer, default=0)
    response_time = Column(Float, default=0.0)  # в секундах
    sources = Column(Text, nullable=True)  # JSON строка с источниками
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Query(id={self.id}, question={self.question[:50]}...)>"
