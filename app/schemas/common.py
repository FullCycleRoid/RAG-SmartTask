from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Ответ health check"""

    status: str
    version: str
    database: str
    redis: str
    vector_store: str
