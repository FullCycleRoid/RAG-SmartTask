-- Создание расширения pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Дополнительные расширения при необходимости
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE DATABASE test_db;