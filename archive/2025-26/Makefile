.PHONY: help install install-dev test test-integration lint lint-fix format format-check clean train

help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	uv sync --locked --no-dev

install-dev:  ## Install package with development dependencies
	uv sync --locked

test:  ## Run tests with pytest
	uv run pytest

test-integration:  ## Run tests requiring the private Hugging Face dataset
	uv run pytest -m integration

test-cov:  ## Run tests with coverage report
	uv run pytest --cov --cov-report=html --cov-report=term

test-fast:  ## Run tests excluding slow tests
	uv run pytest -m "not slow and not integration"

lint:  ## Run linters (ruff, mypy)
	uv run ruff check src/ tests/
	uv run mypy src/

lint-fix:  ## Fix linting issues automatically
	uv run ruff check --fix src/ tests/

format:  ## Format code with black and ruff
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

format-check:  ## Check code formatting without modifying files
	uv run black --check src/ tests/
	uv run ruff check src/ tests/

clean:  ## Remove build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

train:  ## Run training script
	uv run scripts/train_tcn.py
