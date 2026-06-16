# Kubeflow MCP (this repository)

- **MCP server**: `uv run kubeflow-mcp serve` — exposes Kubeflow training tools over MCP (stdio by default).
- **Local agent** (optional LLM): `uv sync --extra agents-langchain` then `uv run kubeflow-mcp agent --provider litellm --model ollama/qwen3:8b` (defaults: `--framework langchain`, `--mode progressive` for local Ollama). Use `--extra agents` for all backends.
- **Docs**: `README.md`, `ROADMAP.md`, `docs/design/agent-provider-architecture.md`.

Use MCP tools instead of guessing kubectl; respect preview-before-submit (`confirmed` flags on mutating tools).

## MCP tool usage in Claude Code

**Do not verify tool availability via bash** (`compgen -A function`, `env | grep mcp`, heredoc greps, etc.) — those never work for MCP tools.

With `platform-admin` persona (configured in `~/.claude.json`), all **23 tools** are available:

`pre_flight` · `check_compatibility` · `get_cluster_resources` · `estimate_resources` · `list_runtimes` · `get_runtime` · `fine_tune` · `run_custom_training` · `run_container_training` · `list_training_jobs` · `get_training_job` · `get_training_logs` · `get_training_events` · `wait_for_training` · `delete_training_job` · `update_training_job` · `inspect_crd` · `inspect_controller` · `patch_runtime` · `create_runtime` · `delete_runtime` · `health_check` · `get_server_logs`

**Never fall back to writing YAML + `kubectl apply`** — the MCP tools cover all training job lifecycle operations.

Workflow: `pre_flight()` first (always) → appropriate training tool with `confirmed=False` (preview) → `confirmed=True` (submit).