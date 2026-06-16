# Copyright The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kubeflow MCP Server CLI."""

import importlib.util
import os
import shutil
import subprocess
import sys
import warnings
from typing import Any

import click

from kubeflow_mcp import __version__

# Suppress pydantic warnings from fastmcp/mcp dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ── Dependency auto-sync ───────────────────────────────────────────────────────

# (module_to_probe, uv_extra_name) — ordered: base framework first, obs add-ons after.
_FRAMEWORK_DEPS: dict[str, tuple[str, list[str]]] = {
    "langchain":  ("agents-langchain",  ["langchain_core", "langchain_litellm"]),
    "smolagents": ("agents-smolagents", ["smolagents"]),
    "llamaindex": ("agents-llamaindex", ["llama_index.core"]),
    "litellm":    ("agents-litellm",    ["litellm"]),
}
_OBS_DEPS: dict[str, tuple[str, list[str]]] = {
    "otel":    ("agents-otel", ["opentelemetry.sdk"]),
    "mlflow":  ("agents-obs",  ["mlflow"]),
    "langfuse":("agents-obs",  ["langfuse"]),
}


def _missing(modules: list[str]) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def _ensure_agent_deps(
    framework: str,
    *,
    otel_endpoint: str | None,
    mlflow_uri: str | None,
    langfuse: bool,
) -> None:
    """Check required packages and offer to auto-sync if anything is missing.

    Uses importlib.util.find_spec() — no imports triggered, runs in <1ms.
    """
    needed_extras: list[str] = []
    missing_modules: list[str] = []

    # Framework deps — always resolve the extra for this framework so that any
    # subsequent `uv sync` below includes it (uv sync replaces the whole env).
    framework_extra, framework_modules = _FRAMEWORK_DEPS.get(
        framework, ("agents-langchain", [])
    )
    gaps = _missing(framework_modules)
    if gaps:
        needed_extras.append(framework_extra)
        missing_modules.extend(gaps)

    # Observability deps — only checked when the corresponding flag is active
    obs_extras_needed: list[str] = []
    for flag_active, key in [
        (bool(otel_endpoint), "otel"),
        (bool(mlflow_uri), "mlflow"),
        (langfuse, "langfuse"),
    ]:
        if flag_active:
            obs_extra, obs_modules = _OBS_DEPS[key]
            gaps = _missing(obs_modules)
            if gaps:
                if obs_extra not in needed_extras:
                    needed_extras.append(obs_extra)
                    obs_extras_needed.append(obs_extra)
                missing_modules.extend(g for g in gaps if g not in missing_modules)

    # If any obs extra is needed, always pin the framework extra too — even if
    # its modules are currently installed — because `uv sync --extra agents-obs`
    # alone will drop the framework packages (uv replaces the whole environment).
    if obs_extras_needed and framework_extra not in needed_extras:
        needed_extras.insert(0, framework_extra)

    if not needed_extras:
        return

    sync_cmd = "uv sync " + " ".join(f"--extra {e}" for e in dict.fromkeys(needed_extras))
    click.echo()
    click.echo(click.style("  Missing packages detected:", fg="yellow", bold=True))
    for m in missing_modules:
        click.echo(click.style(f"    · {m}", fg="yellow"))
    click.echo()
    click.echo(f"  Run:  {click.style(sync_cmd, fg='cyan', bold=True)}")
    click.echo()

    uv = shutil.which("uv")
    if uv and click.confirm("  Auto-install now?", default=True):
        args = [uv, "sync"] + [a for e in dict.fromkeys(needed_extras) for a in ("--extra", e)]
        click.echo()
        result = subprocess.run(args, check=False)
        if result.returncode != 0:
            click.echo(click.style("  uv sync failed — fix errors above and retry.", fg="red"), err=True)
            raise SystemExit(1)
        click.echo()
        click.echo(click.style("  Packages installed. Restarting agent…", fg="green"))
        click.echo()
        os.execv(sys.argv[0], sys.argv)  # re-exec with fresh environment
    else:
        click.echo(f"  Install manually:  {sync_cmd}", err=True)
        raise SystemExit(1)


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Kubeflow MCP Server - AI interface for Kubeflow Training."""
    pass


@cli.command()
@click.option(
    "--clients",
    "-c",
    default=None,
    help="Comma-separated client modules (trainer, optimizer, hub). "
    "Falls back to KUBEFLOW_MCP_CLIENTS env var, config file, then 'trainer'.",
)
@click.option(
    "--persona",
    "-p",
    default=None,
    type=click.Choice(["readonly", "data-scientist", "ml-engineer", "platform-admin"]),
    help="Persona for tool filtering. "
    "Falls back to KUBEFLOW_MCP_PERSONA env var, config file, then 'readonly'.",
)
@click.option(
    "--transport",
    "-t",
    default=None,
    type=click.Choice(["stdio", "http", "sse"]),
    help="MCP transport protocol. Falls back to MCP_TRANSPORT env var, config file, then 'stdio'.",
)
@click.option(
    "--log-level",
    "-l",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level. Falls back to LOG_LEVEL env var, config file, then 'INFO'.",
)
@click.option(
    "--mode",
    "-m",
    default="full",
    type=click.Choice(["full", "progressive", "semantic"]),
    help="Tool loading mode: full (all tools), progressive (hierarchical discovery), semantic (embedding search)",
)
@click.option(
    "--log-format",
    default=None,
    type=click.Choice(["json", "console"]),
    help="Log format (auto-detects if not specified). Falls back to LOG_FORMAT env var, config file.",
)
@click.option(
    "--instruction-tier",
    default=None,
    type=click.Choice(["full", "compact", "minimal"]),
    help="Instruction verbosity: full (all guidance), compact (no resource refs), minimal (tool names only). "
    "Falls back to KUBEFLOW_MCP_INSTRUCTION_TIER env var, config file, then 'full'.",
)
@click.option(
    "--no-banner",
    is_flag=True,
    default=False,
    help="Hide FastMCP startup banner",
)
@click.option(
    "--auth-token",
    default=None,
    help="Bearer token for HTTP auth (dev/staging). "
    "Falls back to KUBEFLOW_MCP_AUTH_TOKEN env var, config file. "
    "Ignored for stdio transport.",
)
@click.option(
    "--otel-endpoint",
    default=None,
    envvar="OTEL_EXPORTER_OTLP_ENDPOINT",
    help=(
        "OTLP HTTP endpoint for server-side OTel traces, e.g. http://localhost:4318. "
        "Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var. "
        "Run deploy/otel/docker-compose.yml to start a local collector."
    ),
)
def serve(
    clients: str | None,
    persona: str | None,
    transport: str | None,
    mode: str,
    log_level: str | None,
    log_format: str | None,
    instruction_tier: str | None,
    no_banner: bool,
    auth_token: str | None,
    otel_endpoint: str | None,
) -> None:
    """Start the MCP server.

    Options fall back to env vars / config file (~/.kubeflow-mcp.yaml) when
    not provided on the command line.  See ``kubeflow_mcp.core.config`` for the
    full precedence chain: CLI flag > env var > config file > built-in default.
    """
    from kubeflow_mcp.core.auth import build_auth_provider
    from kubeflow_mcp.core.config import load_config
    from kubeflow_mcp.core.logging import setup_logging
    from kubeflow_mcp.core.resilience import configure_circuit_breaker
    from kubeflow_mcp.core.server import configure_resilience, create_server
    from kubeflow_mcp.core.telemetry import setup_tracing

    cfg = load_config()

    clients = clients or ",".join(cfg.server.clients)
    persona = persona or cfg.server.persona
    transport = transport or cfg.server.transport
    instruction_tier = instruction_tier or cfg.server.instruction_tier
    log_level = log_level or cfg.logging.level
    log_format = log_format or cfg.logging.format

    if auth_token:
        cfg.auth.auth_token = auth_token
    if otel_endpoint:
        cfg.observability.otel_endpoint = otel_endpoint

    logger = setup_logging(level=log_level, format=log_format)
    tracing_enabled = setup_tracing(endpoint=cfg.observability.otel_endpoint)
    logger.info(
        "Starting kubeflow-mcp",
        extra={
            "clients": clients,
            "persona": persona,
            "transport": transport,
            "mode": mode,
            "instruction_tier": instruction_tier,
            "tracing_enabled": tracing_enabled,
        },
    )

    # Wire server-side OTel tracing (Gap 6A).
    try:
        from kubeflow_mcp.core.telemetry import setup_tracing

        tracing_active = setup_tracing(otel_endpoint)
        logger.info("server_otel_tracing", extra={"active": tracing_active})
    except Exception:
        pass

    configure_resilience(
        rate_limit=cfg.resilience.rate_limit,
        rate_capacity=cfg.resilience.rate_capacity,
    )
    configure_circuit_breaker(
        failure_threshold=cfg.resilience.cb_failure_threshold,
        recovery_timeout=cfg.resilience.cb_recovery_timeout,
    )

    auth_provider = None
    if transport != "stdio":
        auth_provider = build_auth_provider(cfg.auth)
        if auth_provider is None:
            logger.warning(
                "HTTP transport with no auth configured — server is open. "
                "Set --auth-token or KUBEFLOW_MCP_AUTH_TOKEN for bearer auth, "
                "or KUBEFLOW_MCP_JWKS_URI for JWT verification."
            )

    client_list = [c.strip() for c in clients.split(",")]
    server = create_server(
        clients=client_list,
        persona=persona,
        mode=mode,
        instruction_tier=instruction_tier,
        auth_provider=auth_provider,
    )

    show_banner = not no_banner
    _host = os.environ.get("MCP_HOST", "127.0.0.1")
    _port = int(os.environ.get("MCP_PORT", "8000"))
    if transport == "stdio":
        server.run(show_banner=show_banner)
    elif transport == "sse":
        server.run(transport="sse", host=_host, port=_port, show_banner=show_banner)
    else:
        server.run(transport="streamable-http", host=_host, port=_port, show_banner=show_banner)


@cli.command()
def status() -> None:
    """Show server status and enabled tools."""
    from kubeflow_mcp.core.server import CLIENT_MODULES

    click.echo("Kubeflow MCP Server Status")
    click.echo("-" * 40)
    click.echo(f"Version: {__version__}")
    click.echo("\nAvailable clients:")
    for name, module_path in CLIENT_MODULES.items():
        try:
            import importlib

            module = importlib.import_module(module_path)
            info = getattr(module, "MODULE_INFO", {})
            status = info.get("status", "unknown")
            tools = len(getattr(module, "TOOLS", []))
            click.echo(f"  {name}: {status} ({tools} tools)")
        except ImportError:
            click.echo(f"  {name}: not installed")


def _is_local_agent(*, provider: str, model: str | None, base_url: str | None) -> bool:
    """True when the agent talks to local Ollama (not cloud APIs)."""
    if base_url:
        host = base_url.lower()
        if "localhost" in host or "127.0.0.1" in host:
            return True
    if model:
        lowered = model.lower()
        if lowered.startswith("ollama/") or lowered.startswith("ollama_chat/"):
            return True
    return provider == "litellm" and model is None


def _resolve_agent_mode(
    mode: str | None,
    *,
    provider: str,
    model: str,
    base_url: str | None,
) -> str:
    if mode is not None:
        return mode
    if _is_local_agent(provider=provider, model=model, base_url=base_url):
        return "progressive"
    return "full"


def _provider_entry_point_map() -> dict[str, Any]:
    from importlib.metadata import entry_points

    eps = entry_points()
    selected = eps.select(group="kubeflow_mcp.providers")  # type: ignore[union-attr]
    return {ep.name: ep for ep in selected}


@cli.command()
@click.option(
    "--provider",
    "-p",
    default="litellm",
    help="Agent provider (see entry-points group kubeflow_mcp.providers)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model name (provider default if omitted)",
)
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["full", "progressive", "semantic"]),
    help="Tool loading mode (default: progressive for local Ollama, full for cloud)",
)
@click.option(
    "--thinking/--no-thinking",
    default=True,
    help="Enable extended thinking for supported models (default: on; use --no-thinking to disable)",
)
@click.option(
    "--base-url",
    default=None,
    help="LiteLLM base URL for on-prem proxy or local endpoint (litellm provider only)",
)
@click.option(
    "--fallback-model",
    default=None,
    help="Secondary LiteLLM model tried when primary fails (litellm provider only)",
)
@click.option(
    "--cache/--no-cache",
    default=False,
    help="Enable LiteLLM semantic cache — Redis if REDIS_URL is set, else in-memory",
)
@click.option(
    "--langfuse/--no-langfuse",
    default=False,
    help="Enable Langfuse LLM tracing (requires LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY env vars)",
)
@click.option(
    "--mlflow-uri",
    default=None,
    help="Enable MLflow callback and set tracking URI (e.g. http://localhost:5000)",
)
@click.option(
    "--framework",
    default="langchain",
    type=click.Choice(["langchain", "smolagents", "llamaindex", "litellm"]),
    help="Agent framework (default: langchain; litellm = legacy native loop)",
)
@click.option(
    "--otel-endpoint",
    default=None,
    help=(
        "OTLP HTTP endpoint for OTel traces, e.g. http://localhost:4318. "
        "Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var. "
        "Run deploy/otel/docker-compose.yml to start a local collector."
    ),
)
def agent(
    provider: str,
    model: str | None,
    mode: str,
    thinking: bool,
    base_url: str | None,
    fallback_model: str | None,
    cache: bool,
    langfuse: bool,
    mlflow_uri: str | None,
    framework: str,
    otel_endpoint: str | None,
) -> None:
    """Run an interactive AI agent backed by a registered provider."""
    _ensure_agent_deps(
        framework,
        otel_endpoint=otel_endpoint,
        mlflow_uri=mlflow_uri,
        langfuse=langfuse,
    )
    eps = _provider_entry_point_map()
    if provider not in eps:
        available = ", ".join(sorted(eps)) or "none installed"
        click.echo(f"Unknown provider '{provider}'. Available: {available}", err=True)
        raise SystemExit(1)

    try:
        provider_cls = eps[provider].load()
    except ImportError as e:
        click.echo(f"Provider '{provider}' dependencies missing: {e}", err=True)
        raise SystemExit(1) from None

    # Initialize agent-side OTel tracer so spans flow to Jaeger / OTel Collector.
    # Falls back to OTEL_EXPORTER_OTLP_ENDPOINT env var when --otel-endpoint is omitted.
    try:
        from kubeflow_mcp.agents.observability._otel import setup_otel_tracer

        setup_otel_tracer(endpoint=otel_endpoint)
    except Exception:
        pass

    instance = provider_cls()
    resolved_model = model or instance.default_model
    resolved_mode = _resolve_agent_mode(
        mode,
        provider=provider,
        model=resolved_model,
        base_url=base_url,
    )
    kwargs: dict[str, Any] = {
        "thinking": thinking,
        "cache": cache,
        "langfuse": langfuse,
        "mlflow_uri": mlflow_uri,
        "framework": framework,
        "otel_endpoint": otel_endpoint,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    if fallback_model is not None:
        kwargs["fallback_model"] = fallback_model
    instance.run(model=resolved_model, mode=resolved_mode, **kwargs)


@cli.command("eval")
@click.option("--model", "-m", default="ollama/qwen3:8b", help="LiteLLM model string for the agent")
@click.option("--base-url", default=None, help="Agent model base URL (OpenAI-compatible)")
@click.option("--case", default=None, metavar="CASE_ID", help="Run a single eval case by ID")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Stream agent output")
@click.option("--thinking", is_flag=True, default=False, help="Enable extended thinking (off by default)")
@click.option("--timeout", type=float, default=120.0, metavar="SECS", help="Per-case timeout (default 120s)")
@click.option(
    "--judge",
    default="rule",
    type=click.Choice(["rule", "llm", "all"]),
    help="Judge mode: rule (fast), llm (LLM quality), all (both)",
)
@click.option("--judge-model", default="openai/qwen36-27b", help="LiteLLM model for LLM judge")
@click.option("--judge-base-url", default=None, help="Base URL for judge model endpoint")
@click.option(
    "--langfuse/--no-langfuse",
    default=False,
    help="Enable Langfuse experiment tracking (requires LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY)",
)
@click.option("--experiment", default=None, help="Langfuse experiment name (auto-generated if omitted)")
@click.option(
    "--dataset",
    default="kubeflow-mcp-eval",
    help="Langfuse dataset name (default: kubeflow-mcp-eval)",
)
@click.option("--upload-only", is_flag=True, default=False, help="Only sync cases to Langfuse, skip agent run")
def eval_cmd(
    model: str,
    base_url: str | None,
    case: str | None,
    verbose: bool,
    thinking: bool,
    timeout: float,
    judge: str,
    judge_model: str,
    judge_base_url: str | None,
    langfuse: bool,
    experiment: str | None,
    dataset: str,
    upload_only: bool,
) -> None:
    """Run the agent eval suite (rule judges + optional LLM judge + Langfuse tracking)."""
    import sys

    if langfuse or upload_only:
        from eval.langfuse_eval import cli as _langfuse_cli
        # Rebuild sys.argv so the langfuse_eval CLI parser sees the right flags.
        argv = ["langfuse-eval", "--model", model, "--judge", judge, "--judge-model", judge_model]
        if base_url:
            argv += ["--base-url", base_url]
        if case:
            argv += ["--case", case]
        if verbose:
            argv.append("--verbose")
        if thinking:
            argv.append("--thinking")
        if timeout != 120.0:
            argv += ["--timeout", str(timeout)]
        if judge_base_url:
            argv += ["--judge-base-url", judge_base_url]
        if experiment:
            argv += ["--experiment", experiment]
        if dataset != "kubeflow-mcp-eval":
            argv += ["--dataset", dataset]
        if upload_only:
            argv.append("--upload-only")
        sys.argv = argv
        _langfuse_cli()
        return

    # Plain eval (no Langfuse).
    import asyncio
    from eval.run_eval import cli as _eval_cli
    argv = ["eval", "--model", model, "--judge", judge, "--judge-model", judge_model]
    if base_url:
        argv += ["--base-url", base_url]
    if case:
        argv += ["--case", case]
    if verbose:
        argv.append("--verbose")
    if thinking:
        argv.append("--thinking")
    if timeout != 120.0:
        argv += ["--timeout", str(timeout)]
    if judge_base_url:
        argv += ["--judge-base-url", judge_base_url]
    sys.argv = argv
    _eval_cli()


if __name__ == "__main__":
    cli()
