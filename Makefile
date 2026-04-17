.PHONY: install lint format typecheck test run clean ci help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies including dev
	uv sync --extra dev
	uv run pre-commit install

lint: ## Run ruff linter
	uv run ruff check src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck: ## Run mypy type checker
	uv run mypy

test: ## Run tests with coverage
	uv run pytest

run: ## Run the FastAPI dev server
	uv run uvicorn f1_predictor.api.main:app --reload --port 8000

clean: ## Remove build artifacts and caches
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

ci: lint typecheck test ## Run all CI checks locally
