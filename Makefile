# Makefile for Sales Assistant

.PHONY: help install install-dev test test-cov lint format clean build upload run dev docs conda-activate db-reset db-clear db-status db-count

# Conda environment
CONDA_ENV = llms
SHELL := /bin/bash
CONDA_ACTIVATE = source /home/siva01/miniconda3/etc/profile.d/conda.sh && conda activate $(CONDA_ENV) &&

# Default target
help:
	@echo "Sales Assistant - Development Commands"
	@echo "======================================"
	@echo ""
	@echo "Environment:"
	@echo "  conda-activate   Activate conda environment (llms)"
	@echo "  conda-setup      Create and setup conda environment"
	@echo ""
	@echo "Installation:"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package in development mode"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage (min 80%)"
	@echo "  test-watch   Run tests in watch mode"
	@echo "  test-fast    Run tests, stop on first failure"
	@echo "  test-specific TEST=name  Run specific test file"
	@echo "  test-parallel Run tests in parallel"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-basic   Run basic smoke tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run linting (flake8, mypy)"
	@echo "  format       Format code (black, isort)"
	@echo "  format-check Check code formatting"
	@echo ""
	@echo "Development:"
	@echo "  run          Run the application"
	@echo "  dev          Run in development mode"
	@echo "  ps           Show running servers"
	@echo "  kill         Stop all running servers"
	@echo "  docs         Generate documentation"
	@echo ""
	@echo "Database:"
	@echo "  db-reset     Reset ChromaDB (remove all data)"
	@echo "  db-clear     Clear ChromaDB storage"
	@echo "  db-status    Show ChromaDB status"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  upload       Upload to PyPI"
	@echo ""

# Conda Environment
conda-activate:
	@echo "Run: conda activate $(CONDA_ENV)"

conda-setup:
	conda create -n $(CONDA_ENV) python=3.11 -y
	$(CONDA_ACTIVATE) pip install -e .[dev,test]
	$(CONDA_ACTIVATE) pre-commit install

conda-check:
	@echo "Checking conda environment..."
	@if conda info --envs | grep -q $(CONDA_ENV); then \
		echo "✅ Environment '$(CONDA_ENV)' exists"; \
	else \
		echo "❌ Environment '$(CONDA_ENV)' not found. Run 'make conda-setup'"; \
		exit 1; \
	fi

env-info:
	@echo "Environment Information:"
	@echo "======================="
	@echo "Python: $$(python --version)"
	@echo "Conda Environment: $$CONDA_DEFAULT_ENV"
	@echo "Working Directory: $$(pwd)"
	@echo "PYTHONPATH: $$PYTHONPATH"

# Installation
install:
	$(CONDA_ACTIVATE) pip install -e .

install-dev:
	$(CONDA_ACTIVATE) pip install -e .[dev,test]
	$(CONDA_ACTIVATE) pre-commit install

# Testing
test:
	$(CONDA_ACTIVATE) PYTHONPATH=src pytest tests/ -v --tb=short

test-cov:
	$(CONDA_ACTIVATE) PYTHONPATH=src pytest tests/ --cov=src/assistant --cov-report=html --cov-report=term-missing --cov-fail-under=80

test-watch:
	$(CONDA_ACTIVATE) PYTHONPATH=src pytest-watch tests/ --runner "PYTHONPATH=src pytest tests/ -v"

test-fast:
	$(CONDA_ACTIVATE) PYTHONPATH=src pytest tests/ -x --tb=short

test-specific:
	@echo "Usage: make test-specific TEST=test_file_name"
	@echo "Example: make test-specific TEST=test_chat_services"
	@if [ -n "$(TEST)" ]; then $(CONDA_ACTIVATE) PYTHONPATH=src pytest tests/$(TEST).py -v; fi

test-parallel:
	$(CONDA_ACTIVATE) PYTHONPATH=src pytest tests/ -n auto -v

test-integration:
	$(CONDA_ACTIVATE) PYTHONPATH=src pytest tests/ -m integration -v

test-unit:
	$(CONDA_ACTIVATE) PYTHONPATH=src pytest tests/ -m "not integration" -v

test-basic:
	$(CONDA_ACTIVATE) python test_runner.py

# Code Quality
lint:
	$(CONDA_ACTIVATE) flake8 src/assistant tests/
	$(CONDA_ACTIVATE) mypy src/assistant

format:
	$(CONDA_ACTIVATE) black src/assistant tests/
	$(CONDA_ACTIVATE) isort src/assistant tests/

format-check:
	$(CONDA_ACTIVATE) black --check src/assistant tests/
	$(CONDA_ACTIVATE) isort --check-only src/assistant tests/

# Development
run:
	$(CONDA_ACTIVATE) PYTHONPATH=src python -m assistant.api_server

dev:
	$(CONDA_ACTIVATE) PYTHONPATH=src uvicorn assistant.api_server:app --reload --host 0.0.0.0 --port 5000

ps:
	@echo "Running servers and processes:"
	@echo "=============================="
	@ps aux | grep -E "(make|uvicorn|python.*assistant)" | grep -v grep | awk '{print $$2 " " $$11 " " $$12 " " $$13 " " $$14}' | sort -k2 || echo "No running servers found"

kill:
	@echo "Stopping all running servers..."
	@pkill -f "uvicorn assistant.api_server" 2>/dev/null || true
	@pkill -f "make dev" 2>/dev/null || true
	@sleep 1
	@echo "All servers stopped."

docs:
	@echo "Documentation is available in docs/ directory"
	@echo "API docs available at http://localhost:8000/docs when running"

# Build & Deploy
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

build: clean
	$(CONDA_ACTIVATE) python -m build

upload: build
	$(CONDA_ACTIVATE) twine upload dist/*

# Docker commands
docker-build:
	docker build -t sales-assistant .

docker-run:
	docker run -p 8000:8000 --env-file .env sales-assistant

# Database commands
db-reset:
	@echo "Resetting ChromaDB database..."
	@echo "🔄 Stopping any running servers..."
	@pkill -f "uvicorn assistant.api_server" 2>/dev/null || true
	@pkill -f "make dev" 2>/dev/null || true
	@sleep 1
	@echo "🗑️  Removing ChromaDB files..."
	rm -rf chromadb/
	@echo "✅ ChromaDB database reset complete"
	@echo "All vector embeddings and data have been removed"
	@echo "💡 Start server with 'make dev' and call /index_knowledge to reinitialize"

db-clear:
	@echo "Clearing ChromaDB storage..."
	@echo "⚠️  This will permanently delete all vector embeddings and data"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(CONDA_ACTIVATE) echo "Using conda environment: $(CONDA_ENV)"; \
		echo "🔄 Stopping any running servers..."; \
		pkill -f "uvicorn assistant.api_server" 2>/dev/null || true; \
		pkill -f "make dev" 2>/dev/null || true; \
		sleep 1; \
		echo "🗑️  Removing ChromaDB files..."; \
		rm -rf chromadb/; \
		echo "✅ ChromaDB storage cleared successfully"; \
		echo "💡 Start server with 'make dev' and call /index_knowledge to reinitialize"; \
	else \
		echo "❌ Operation cancelled"; \
	fi

db-status:
	@echo "ChromaDB Status:"
	@echo "================"
	@if [ -d "chromadb" ]; then \
		echo "✅ ChromaDB directory exists"; \
		echo "📂 Directory size: $$(du -sh chromadb 2>/dev/null || echo 'N/A')"; \
		echo "📄 Files: $$(find chromadb -type f | wc -l) files"; \
		if [ -f "chromadb/chroma.sqlite3" ]; then \
			echo "🗄️  SQLite database exists"; \
		else \
			echo "❌ SQLite database not found"; \
		fi; \
	else \
		echo "❌ ChromaDB directory not found"; \
		echo "💡 Run 'make run' or call /index_knowledge endpoint to initialize"; \
	fi
	@echo ""
	@echo "💡 To check record count, use: make db-count"

db-count:
	@echo "Counting ChromaDB records..."
	@$(CONDA_ACTIVATE) PYTHONPATH=src python -c "import asyncio; from assistant.services.vector_store import VectorStore; exec(open('scripts/db_count_inline.py').read())" 2>/dev/null || \
	$(CONDA_ACTIVATE) PYTHONPATH=src python -c "import asyncio; from assistant.services.vector_store import VectorStore; store = VectorStore(); records = asyncio.run(store.get_all_records()); print(f'📊 Total records: {len(records[\"ids\"])}')"

# Environment setup
env-setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
		echo "Please edit .env file with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi

# Development setup
setup: env-setup install-dev
	@echo "Development environment setup complete!"
	@echo "Edit .env file with your configuration, then run 'make dev'"

# Check project structure
check-structure:
	@echo "Project structure:"
	@tree -I '__pycache__|*.pyc|*.pyo|chromadb|logs|.git|.env|venv|node_modules'

# Version management
version:
	@python -c "import assistant; print(assistant.__version__)"

# Security check
security-check:
	pip-audit
	safety check

# All checks before commit
pre-commit: format lint test
	@echo "All checks passed! Ready to commit."