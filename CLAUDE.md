# Kubeflow MCP (this repository)

- **MCP server**: `uv run kubeflow-mcp serve` — exposes Kubeflow training tools over MCP (stdio by default).
- **Local agent** (optional LLM): `uv sync --extra agents-langchain` then `uv run kubeflow-mcp agent --provider litellm --model ollama/qwen3:8b` (defaults: `--framework langchain`, `--mode progressive` for local Ollama). Use `--extra agents` for all backends.
- **Docs**: `README.md`, `ROADMAP.md`, `docs/design/agent-provider-architecture.md`.

Use MCP tools instead of guessing kubectl; respect preview-before-submit (`confirmed` flags on mutating tools).
