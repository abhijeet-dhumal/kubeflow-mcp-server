# Contributing to Kubeflow MCP Server

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/mcp-server.git
   cd mcp-server
   ```

3. Set up development environment:
   ```bash
   uv sync --all-extras
   uv run pre-commit install
   ```

4. Create a branch:
   ```bash
   git checkout -b feat/your-feature
   ```

## Development Workflow

### Running Tests
```bash
uv run pytest                    # All tests
uv run pytest tests/unit         # Unit tests only
uv run pytest -k "test_name"     # Specific test
```

### Linting
```bash
uv run ruff check .              # Check
uv run ruff check . --fix        # Auto-fix
uv run ruff format .             # Format
```

### Type Checking
```bash
uv run mypy src/
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types:** feat, fix, docs, style, refactor, perf, test, build, ci, chore

**Examples:**
- `feat(trainer): add create_training_job tool`
- `fix(core): handle timeout in k8s client`
- `docs: update README with usage examples`

## Pull Request Process

1. Update tests for your changes
2. Ensure all checks pass
3. Update documentation if needed
4. Request review from maintainers

## Areas Open for Contribution

- **OptimizerClient tools** - Hyperparameter optimization integration
- **ModelRegistryClient tools** - Model registry integration
- **Documentation** - Examples and tutorials
- **Testing** - Increase test coverage

## Code of Conduct

This project follows the [Kubeflow Code of Conduct](https://github.com/kubeflow/community/blob/master/CODE_OF_CONDUCT.md).

## Questions?

Open an issue or reach out to maintainers.
