SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: uv
uv: ## Install UV
	@command -v uv &> /dev/null || { \
	  curl -LsSf https://astral.sh/uv/install.sh | sh; \
	  echo "✅ uv has been installed."; \
	}

.PHONY: verify
verify: ## Run linting, formatting and type checking
	@uv lock --check
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run mypy kubeflow_mcp

.PHONY: test-python
test-python: ## Run Python unit tests
	@uv sync --all-extras
	@uv run pytest --cov=kubeflow_mcp --cov-report=$(or $(report),term)

.PHONY: install-dev
install-dev: uv ## Install dependencies and tools
	@echo "Syncing dependencies with uv..."
	@uv sync --all-extras
	@echo "Environment is ready."
