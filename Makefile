.PHONY: test test-evaluation test-unit test-integration coverage

# Запуск всех тестов
test:
	docker compose exec api pytest tests/ -v

# Запуск только скрипта оценки
test-langsmith:
	docker compose exec api python -m app.scripts.run_langsmith_evaluation

# Запуск только тестов оценки
test-evaluation:
	docker compose exec api pytest tests/evaluations -v

# Запуск unit тестов
test-unit:
	docker compose exec api pytest tests/unit/ tests/services/ -v -k "not integration"

# Запуск интеграционных тестов
test-integration:
	docker compose exec api pytest tests/integration/ -v -k "integration"

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