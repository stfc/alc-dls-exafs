.PHONY: help install install-dev test test-verbose clean lint format type-check pre-commit build docs uv-sync uv-lock

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

uv-sync:  ## Sync dependencies with uv (if using pyproject.toml dependencies)
	uv sync

uv-lock:  ## Generate uv.lock file
	uv lock

install:  ## Install the package
	uv pip install -e .

install-dev:  ## Install development dependencies
	uv pip install -e .
	uv pip install -r requirements-dev.txt

install-uv-dev:  ## Install development dependencies using uv sync (recommended)
	uv sync --extra dev --extra test

test:  ## Run tests
	uv run pytest

test-verbose:  ## Run tests with verbose output
	uv run pytest -v

test-coverage:  ## Run tests with coverage report
	uv run pytest --cov=larch_cli_wrapper --cov-report=html --cov-report=term

clean:  ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

lint:  ## Run linting (flake8)
	uv run flake8 src/ tests/

format:  ## Format code with black and isort
	uv run black src/ tests/
	uv run isort src/ tests/

format-check:  ## Check code formatting
	uv run black --check src/ tests/
	uv run isort --check-only src/ tests/

type-check:  ## Run type checking with mypy
	uv run mypy src/

pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install

pre-commit:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

build:  ## Build the package
	uv build

build-check:  ## Check the built package
	uv build
	uv run twine check dist/*

docs:  ## Generate documentation (if sphinx is set up)
	@echo "Documentation generation not yet configured"

dev-setup:  ## Complete development setup
	make install-dev
	make pre-commit-install
	@echo "Development environment is ready!"

check-all:  ## Run all checks (format, lint, type, test)
	make format-check
	make lint  
	make type-check
	make test
