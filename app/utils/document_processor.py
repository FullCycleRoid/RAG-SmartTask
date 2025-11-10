"""
Утилиты для обработки документов
Оптимизированная версия с контролем памяти и RecursiveCharacterTextSplitter
"""

import gc
import re
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import get_settings
from app.core.logger import logger

settings = get_settings()


class DocumentProcessor:
    """Процессор для обработки документов с оптимизацией памяти и улучшенным чанкированием"""

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Двойной перенос строки (абзацы)
                "\n",  # Одиночный перенос строки
                ". ",  # Точка с пробелом (предложения)
                "! ",  # Восклицательный знак с пробелом
                "? ",  # Вопросительный знак с пробелом
                "; ",  # Точка с запятой с пробелом
                ": ",  # Двоеточие с пробелом
                ", ",  # Запятая с пробелом
                " ",  # Пробел (слова)
                "",  # Пустая строка (символы)
            ],
        )

    def load_pdf(self, filepath: str) -> str:
        """
        Загрузить текст из PDF файла с оптимизацией памяти

        Args:
            filepath: Путь к PDF файлу

        Returns:
            str: Извлеченный текст
        """
        try:
            reader = PdfReader(filepath)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Очищаем текст от лишних пробелов и переносов
                    cleaned_text = self.clean_text(page_text)
                    text_parts.append(cleaned_text)

                # Очищаем память после каждых 10 страниц
                if page_num > 0 and page_num % 10 == 0:
                    gc.collect()

            logger.info(f"Loaded {len(reader.pages)} pages from {filepath}")

            # Собираем финальный текст
            final_text = "\n\n".join(
                text_parts
            )  # Используем двойные переносы для разделения абзацев
            del text_parts  # Освобождаем память
            gc.collect()

            return final_text

        except Exception as e:
            logger.error(f"Error loading PDF {filepath}: {e}")
            raise

    def load_text_file(self, filepath: str) -> str:
        """
        Загрузить текст из TXT или MD файла

        Args:
            filepath: Путь к текстовому файлу

        Returns:
            str: Извлеченный текст
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            logger.info(f"Loaded text file {filepath}")
            return self.clean_text(text)

        except Exception as e:
            logger.error(f"Error loading text file {filepath}: {e}")
            raise

    def process_text_file(self, filepath: str) -> List[str]:
        """
        Загрузить текстовый файл и разбить на фрагменты с помощью RecursiveCharacterTextSplitter

        Args:
            filepath: Путь к текстовому файлу

        Returns:
            List[str]: Список фрагментов текста
        """
        text = self.load_text_file(filepath)
        return self._chunk_with_text_splitter(text)

    def process_document(self, filepath: str) -> List[str]:
        """
        Универсальный метод для обработки документов разных типов

        Args:
            filepath: Путь к файлу

        Returns:
            List[str]: Список фрагментов текста
        """
        file_extension = Path(filepath).suffix.lower()

        if file_extension == ".pdf":
            return self.load_pdf_memory_efficient(filepath)
        elif file_extension in [".txt", ".md"]:
            return self.process_text_file(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def clean_text(self, text: str) -> str:
        """Улучшенная очистка текста"""
        # Сохраняем больше символов для лучшего разбиения
        text = re.sub(r"[^\w\s\.\,\!\?\-\:\(\)\"\«\»\—\-\+\=\@\#\$\%\&\*]", " ", text)
        # Нормализуем пробелы, но сохраняем структуру
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _chunk_with_text_splitter(self, text: str) -> List[str]:
        """
        Разбить текст на чанки с помощью RecursiveCharacterTextSplitter

        Args:
            text: Исходный текст

        Returns:
            List[str]: Список чанков
        """
        if not text or len(text.strip()) == 0:
            return []

        try:
            chunks = self.text_splitter.split_text(text)
            logger.debug(
                f"Split text into {len(chunks)} chunks using RecursiveCharacterTextSplitter"
            )

            # Логируем информацию о чанках для отладки
            for i, chunk in enumerate(chunks[:3]):  # Первые 3 чанка
                logger.debug(
                    f"Chunk {i+1}: {len(chunk)} chars, preview: {chunk[:100]}..."
                )

            if len(chunks) > 3:
                logger.debug(f"... and {len(chunks) - 3} more chunks")

            return chunks

        except Exception as e:
            logger.error(
                f"Error splitting text with RecursiveCharacterTextSplitter: {e}"
            )
            # Fallback на базовое разбиение
            return self._chunk_text_fallback(text)

    def _chunk_text_fallback(self, text: str) -> List[str]:
        """
        Фолбэк метод для разбиения текста (используется при ошибках text splitter)

        Args:
            text: Исходный текст

        Returns:
            List[str]: Список чанков
        """
        if not text:
            return []

        # Простое разбиение по предложениям с ограничением длины
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def load_pdf_memory_efficient(self, filepath: str) -> List[str]:
        """
        Загрузить PDF и сразу разбить на чанки без хранения полного текста в памяти

        Args:
            filepath: Путь к PDF файлу

        Returns:
            List[str]: Список чанков
        """
        try:
            reader = PdfReader(filepath)
            all_chunks = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Очищаем текст
                    cleaned_text = re.sub(r"\s+", " ", page_text).strip()

                    # Сразу разбиваем на чанки с помощью text splitter
                    page_chunks = self._chunk_with_text_splitter(cleaned_text)
                    all_chunks.extend(page_chunks)

                    # Очищаем ссылки для помощи GC
                    del cleaned_text
                    del page_chunks

                # Сборка мусора после каждых 5 страниц
                if page_num > 0 and page_num % 5 == 0:
                    gc.collect()

            logger.info(f"Loaded and chunked {len(reader.pages)} pages from {filepath}")
            return all_chunks

        except Exception as e:
            logger.error(f"Error in memory efficient PDF loading {filepath}: {e}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """
        Разбить текст на фрагменты (совместимость со старым кодом)

        Args:
            text: Исходный текст

        Returns:
            List[str]: Список фрагментов
        """
        return self._chunk_with_text_splitter(text)

    def process_pdf(self, filepath: str) -> List[str]:
        """
        Загрузить PDF и разбить на фрагменты с оптимизацией памяти

        Args:
            filepath: Путь к PDF файлу

        Returns:
            List[str]: Список фрагментов текста
        """
        return self.load_pdf_memory_efficient(filepath)

    def load_and_chunk_pdf(self, filepath: str) -> List[str]:
        """
        Загрузить PDF и разбить на фрагменты (Алиас для обратной совместимости)

        Args:
            filepath: Путь к PDF файлу

        Returns:
            List[str]: Список фрагментов текста
        """
        return self.process_pdf(filepath)

    def get_document_files(self, directory: str = None) -> List[str]:
        """
        Получить список PDF файлов в директории

        Args:
            directory: Путь к директории (по умолчанию из настроек)

        Returns:
            List[str]: Список путей к PDF файлам
        """
        if directory is None:
            directory = settings.DOCUMENTS_DIR

        directory_path = Path(directory)

        if not directory_path.exists():
            logger.warning(f"Documents directory not found: {directory}")
            return []

        pdf_files = []
        for filename in directory_path.glob("*.pdf"):
            pdf_files.append(str(filename))

        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        return pdf_files

    def get_splitter_info(self) -> dict:
        """
        Получить информацию о text splitter

        Returns:
            dict: Информация о конфигурации
        """
        return {
            "splitter_type": "RecursiveCharacterTextSplitter",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.text_splitter._separators,
        }


document_processor = DocumentProcessor()
