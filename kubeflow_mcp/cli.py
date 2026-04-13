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

import warnings

import click

from kubeflow_mcp import __version__

# Suppress pydantic warnings from fastmcp/mcp dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Kubeflow MCP Server - AI interface for Kubeflow Training."""
    pass


@cli.command()
@click.option(
    "--clients",
    "-c",
    default="trainer",
    help="Comma-separated client modules (trainer, optimizer, hub)",
)
@click.option(
    "--persona",
    "-p",
    default="ml-engineer",
    type=click.Choice(["readonly", "data-scientist", "ml-engineer", "platform-admin"]),
    help="Persona for tool filtering",
)
@click.option(
    "--transport",
    "-t",
    default="stdio",
    type=click.Choice(["stdio", "http"]),
    help="MCP transport protocol",
)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.option(
    "--log-format",
    default=None,
    type=click.Choice(["json", "console"]),
    help="Log format (auto-detects if not specified)",
)
@click.option(
    "--no-banner",
    is_flag=True,
    default=False,
    help="Hide FastMCP startup banner",
)
def serve(
    clients: str,
    persona: str,
    transport: str,
    log_level: str,
    log_format: str | None,
    no_banner: bool,
) -> None:
    """Start the MCP server."""
    from kubeflow_mcp.core.logging import setup_logging
    from kubeflow_mcp.core.server import create_server

    logger = setup_logging(level=log_level, format=log_format)
    logger.info(
        "Starting kubeflow-mcp",
        extra={"clients": clients, "persona": persona, "transport": transport},
    )

    client_list = [c.strip() for c in clients.split(",")]
    server = create_server(clients=client_list, persona=persona)

    show_banner = not no_banner
    if transport == "stdio":
        server.run(show_banner=show_banner)
    else:
        server.run(transport="streamable-http", show_banner=show_banner)


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


@cli.command()
@click.option(
    "--backend",
    "-b",
    default="ollama",
    type=click.Choice(["ollama"]),
    help="Agent backend (ollama, openai, anthropic)",
)
@click.option(
    "--model",
    "-m",
    default="qwen3:8b",
    help="Model name for the agent",
)
@click.option(
    "--mode",
    default="full",
    type=click.Choice(["full", "progressive", "semantic", "static", "mcp"]),
    help="Tool loading mode (static/mcp are legacy aliases for full)",
)
@click.option(
    "--thinking/--no-thinking",
    default=False,
    help="Enable thinking output for supported models",
)
def agent(backend: str, model: str, mode: str, thinking: bool) -> None:
    """Run an interactive AI agent."""
    if backend == "ollama":
        try:
            from kubeflow_mcp.agents.ollama import main as ollama_main
        except ImportError:
            click.echo("Error: Agent dependencies not installed.", err=True)
            click.echo("Install with: pip install kubeflow-mcp[agents]", err=True)
            raise SystemExit(1) from None

        import sys

        sys.argv = [
            "kubeflow-mcp-agent",
            "--model",
            model,
            "--mode",
            mode,
        ]
        if thinking:
            sys.argv.append("--thinking")
        ollama_main()
    else:
        click.echo(f"Backend '{backend}' not yet implemented.", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
