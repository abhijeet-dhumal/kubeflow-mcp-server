.PHONY: help install dev install-agents lint format check typecheck
.PHONY: test test-unit test-bench test-cov benchmark
.PHONY: serve serve-debug serve-http status
.PHONY: agent agent-progressive agent-semantic
.PHONY: docs docs-serve docs-clean
.PHONY: build clean pre-commit

# Default target
help:
	@echo "Kubeflow MCP Server - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install base dependencies"
	@echo "  make dev              Install all dev dependencies"
	@echo "  make install-agents   Install agent dependencies (for Ollama)"
	@echo ""
	@echo "Server:"
	@echo "  make serve            Start MCP server (stdio transport)"
	@echo "  make serve-debug      Start with DEBUG logging"
	@echo "  make serve-http       Start with streamable-http transport"
	@echo "  make status           Show server status and tools"
	@echo ""
	@echo "Agent (requires Ollama at localhost:11434):"
	@echo "  make agent            Full mode - all tools via MCP"
	@echo "  make agent-progressive Progressive mode - 91% fewer tokens"
	@echo "  make agent-semantic   Semantic mode - 92% fewer tokens"
	@echo "  (Override model: make agent OLLAMA_MODEL=llama3.2:3b)"
	@echo ""
	@echo "Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Auto-format code"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make check            Run all quality checks"
	@echo "  make pre-commit       Run all checks before committing"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-bench       Run benchmark tests only"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark        Generate dashboard (PNG + JSON)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build HTML documentation"
	@echo "  make docs-serve       Build and serve docs locally"
	@echo "  make docs-clean       Remove docs build artifacts"
	@echo ""
	@echo "Build:"
	@echo "  make build            Build wheel package"
	@echo "  make clean            Remove build artifacts"

# Setup
install:
	uv sync --extra trainer

dev:
	uv sync --extra dev --extra trainer --extra benchmark --extra agents

install-agents:
	uv sync --extra trainer --extra agents

# Server (auto-installs trainer deps)
serve:
	@uv sync --extra trainer --quiet
	uv run kubeflow-mcp serve

serve-debug:
	@uv sync --extra trainer --quiet
	uv run kubeflow-mcp serve --log-level DEBUG

serve-http:
	@uv sync --extra trainer --quiet
	@echo "Starting MCP server with streamable-http transport..."
	uv run kubeflow-mcp serve --transport http

status:
	@uv sync --extra trainer --quiet
	uv run kubeflow-mcp status

# Agent (auto-installs agents deps, requires Ollama at localhost:11434)
OLLAMA_MODEL ?= qwen3:8b

agent:
	@uv sync --extra trainer --extra agents --quiet
	uv run python -m kubeflow_mcp.agents.ollama --model $(OLLAMA_MODEL) --mode full

agent-progressive:
	@uv sync --extra trainer --extra agents --quiet
	uv run python -m kubeflow_mcp.agents.ollama --model $(OLLAMA_MODEL) --mode progressive

agent-semantic:
	@uv sync --extra trainer --extra agents --quiet
	uv run python -m kubeflow_mcp.agents.ollama --model $(OLLAMA_MODEL) --mode semantic

# Quality checks (auto-installs dev deps)
lint:
	@uv sync --extra dev --quiet
	uv run ruff check .

format:
	@uv sync --extra dev --quiet
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	@uv sync --extra dev --extra trainer --quiet
	uv run mypy src/kubeflow_mcp --ignore-missing-imports

check: lint typecheck

pre-commit: format check test-unit
	@echo "All checks passed - ready to commit"

# Testing (auto-installs dev + trainer deps)
test:
	@uv sync --extra dev --extra trainer --quiet
	uv run pytest tests/ -v --tb=short

test-unit:
	@uv sync --extra dev --extra trainer --quiet
	uv run pytest tests/unit/ -v --tb=short

test-bench:
	@uv sync --extra dev --extra trainer --quiet
	uv run pytest tests/benchmarks/ -v --tb=short

test-cov:
	@uv sync --extra dev --extra trainer --quiet
	uv run pytest tests/unit/ --cov=kubeflow_mcp --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Benchmarks (pytest-benchmark is included in dev extra)
benchmark:
	@uv sync --extra dev --extra trainer --quiet
	@echo "Generating benchmark dashboard..."
	uv run pytest tests/benchmarks/test_report.py -v -s
	@echo ""
	@echo "Output:"
	@echo "  PNG: tests/benchmarks/results/benchmark_report_latest.png"
	@echo "  JSON: tests/benchmarks/results/benchmark_report_latest.json"

# Build
build:
	uv build
	@echo "Built packages in dist/"

# Documentation (auto-installs docs deps)
docs:
	@uv sync --extra docs --extra trainer --extra agents --quiet
	@echo "Building documentation..."
	uv run sphinx-build -b html docs/source docs/_build/html
	@echo ""
	@echo "Documentation built: docs/_build/html/index.html"

docs-serve: docs
	@echo "Serving docs at http://localhost:8000"
	@cd docs/_build/html && python -m http.server 8000

docs-clean:
	rm -rf docs/_build
	@echo "Cleaned docs build artifacts"

# Cleanup
clean: docs-clean
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"
