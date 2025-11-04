"""
Настройка логирования
"""

import logging
import sys

from app.core.config import get_settings

settings = get_settings()


def setup_logging():
    """Настроить логирование для приложения"""

    # Формат логов
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Настройка корневого логгера
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Отключаем избыточные логи от библиотек
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = setup_logging()
