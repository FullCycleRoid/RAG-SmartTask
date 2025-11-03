"""
Главный файл FastAPI приложения
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logger import logger
from app.services.cache import cache_manager

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Starting up SmartTask FAQ Service...")

    # Подключаемся к Redis
    await cache_manager.connect()

    yield

    # Shutdown
    logger.info("Shutting down SmartTask FAQ Service...")
    await cache_manager.disconnect()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent FAQ service for SmartTask using RAG",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api", tags=["API"])

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Главная страница - веб-интерфейс"""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)