# Example agent runners

Use the main CLI (recommended):

```bash
uv sync --extra agents-litellm
uv run kubeflow-mcp agent --provider litellm --model ollama/qwen3:8b
# Smaller tool schema (same as serve --mode progressive):
uv run kubeflow-mcp agent --provider litellm --model ollama/qwen3:8b --mode progressive
```

LiteLLM with cloud models:

```bash
uv sync --extra agents-litellm
uv run kubeflow-mcp agent --provider litellm --model gpt-4o-mini
```

All framework backends: `uv sync --extra agents`.

Or run the thin wrappers in this directory (same behavior, explicit `PYTHONPATH` not required when the package is installed):

```bash
uv run python examples/agents/litellm/run.py --model ollama/qwen3:8b
uv run python examples/agents/litellm/run.py --model gpt-4o-mini
```

See each subfolder `README.md` for provider-specific notes.
