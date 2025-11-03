.PHONY: test test-evaluation test-unit test-integration coverage

# Запуск всех тестов
test:
	docker compose exec api pytest tests/ -v

# Запуск только тестов оценки
test-evaluation:
	docker compose exec api pytest tests/services/test_evaluation.py -v

# Запуск unit тестов
test-unit:
	docker compose exec api pytest tests/unit/ tests/services/ -v -k "not integration"

# Запуск интеграционных тестов
test-integration:
	docker compose exec api pytest tests/integration/ tests/services/ -v -k "integration"

# Покрытие кода
coverage:
	docker compose exec api pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Быстрый запуск тестов (только маркер unit)
test-fast:
	docker compose exec api pytest tests/ -v -m "unit"

# Запуск тестов с выводом подробной информации
test-verbose:
	docker compose exec api pytest tests/ -v -s

# Запуск конкретного тестового файла
test-file:
	docker compose exec api pytest tests/services/test_evaluation.py::TestRAGEvaluator::test_evaluate_single_response_success -v -s