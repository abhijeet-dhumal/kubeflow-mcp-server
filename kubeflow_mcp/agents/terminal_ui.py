# Copyright 2026 The Kubeflow Authors.
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

"""Shared Rich terminal helpers for interactive CLI agents."""

from __future__ import annotations

import json
import sys
from typing import Any

_SPHINX_BUILD = "sphinx" in sys.modules

try:
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
except ImportError:
    if not _SPHINX_BUILD:
        sys.exit(
            "Error: required packages not installed\n"
            "Run: uv sync --extra agents-langchain   # or agents for all backends"
        )
    Columns = None  # type: ignore[misc, assignment]
    Console = None  # type: ignore[misc, assignment]
    Group = None  # type: ignore[misc, assignment]
    Markdown = None  # type: ignore[misc, assignment]
    Padding = None  # type: ignore[misc, assignment]
    Panel = None  # type: ignore[misc, assignment]
    Rule = None  # type: ignore[misc, assignment]
    Table = None  # type: ignore[misc, assignment]
    Text = None  # type: ignore[misc, assignment]

_console: Console | None = None


def get_console() -> Console:
    global _console  # noqa: PLW0603
    if _console is None:
        _console = Console()
    return _console


def setup_readline_history(history_file: str | None = None) -> None:
    try:
        import atexit
        import os
        import readline  # noqa: F401

        path = history_file or os.path.expanduser("~/.kubeflow_mcp_history")
        try:
            readline.read_history_file(path)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file, path)
    except ImportError:
        pass


# ── Welcome ───────────────────────────────────────────────────────────────────


_PANEL_MAX_WIDTH = 88  # caps render width — panel never overflows on resize


def print_welcome_panel(
    *,
    panel_title: str,
    border_style: str,
    rows: list[tuple[str, str]],
) -> None:
    """Render a Rich Panel for the agent welcome block.

    Width is capped at _PANEL_MAX_WIDTH regardless of the terminal size.
    This prevents the panel from spanning hundreds of columns on a wide
    terminal and then wrapping when the user resizes.  Content longer than
    the inner width is truncated with an ellipsis.
    """
    c = get_console()
    panel_width = min(c.width or _PANEL_MAX_WIDTH, _PANEL_MAX_WIDTH)
    # 4 = 2 border chars (│) + 2 padding chars (1 each side)
    inner_width = max(panel_width - 4, 20)

    lines: list[Any] = []
    for style, text in rows:
        if text:
            t = Text(text, style=style, overflow="ellipsis", no_wrap=True)
            t.truncate(inner_width, overflow="ellipsis")
            lines.append(t)
        else:
            lines.append(Text(""))

    c.print()
    c.print(
        Panel(
            Group(*lines),
            title=f"[bold {border_style}]{panel_title}[/bold {border_style}]",
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
            width=panel_width,
            expand=False,
        )
    )
    c.print()


# ── Turn I/O ──────────────────────────────────────────────────────────────────


def print_user_panel(c: Console, user_text: str) -> None:
    """Echo a completed user turn in history (no separators — those are for live input only)."""
    c.print()
    c.print(f"[bold bright_cyan]❯[/bold bright_cyan] [bright_white]{user_text}[/bright_white]")


def print_assistant_panel(c: Console, markdown_text: str) -> None:
    c.print()
    c.print("[dim]Assistant[/dim]")
    c.print()
    c.print(Padding(Markdown(markdown_text), (0, 0, 0, 2)))
    c.print()


def print_error_panel(c: Console, exc: Exception) -> None:
    c.print()
    c.print(f"[bold red]✗[/bold red] [red]{type(exc).__name__}:[/red] {exc}")
    c.print()


# ── Tool display ──────────────────────────────────────────────────────────────


def print_tool_call_panel(c: Console, tool_name: str, args: dict[str, Any]) -> None:
    is_mutating = any(
        tool_name.startswith(p)
        for p in ("fine_tune", "run_", "delete_", "update_", "patch_", "create_")
    )
    colour = "yellow" if is_mutating else "cyan"
    args_inline = _format_args_inline(args)
    c.print()
    c.print(f"[{colour}]⚙  {tool_name}[/{colour}][dim]{args_inline}[/dim]")


def print_tool_result_panel(c: Console, result_text: str) -> None:
    """Display a smart Rich summary for known tool result shapes."""
    c.print()
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        data = None

    if isinstance(data, dict):
        rendered = _render_tool_result(c, data)
        if rendered:
            return

    # Fallback: plain dim lines, capped
    lines = result_text.split("\n")
    limit = 12
    for line in lines[:limit]:
        c.print(f"  [dim]{line}[/dim]")
    if len(lines) > limit:
        c.print(f"  [dim]… ({len(lines) - limit} more lines)[/dim]")


def print_confirm_gate(c: Console, preview_text: str) -> None:
    """Distinct preview panel for confirmed=False tool calls."""
    c.print()
    c.print(Rule(title="[bold yellow]⚠  Preview — not yet submitted[/bold yellow]", style="yellow"))
    c.print()
    for line in preview_text.split("\n")[:30]:
        c.print(f"  [dim]{line}[/dim]")
    c.print()


# ── Rich-native result renderers ──────────────────────────────────────────────


def _render_tool_result(c: Console, data: dict[str, Any]) -> bool:
    """Render known tool result shapes using native Rich components. Return True if handled."""
    success = data.get("success")
    inner = data.get("data", {})
    if not isinstance(inner, dict):
        return False

    # ── pre_flight / check_compatibility ──────────────────────────────────────
    # Match broadly on "compatibility" key — the sibling keys vary by version
    # (cluster_resources vs cluster vs resources).
    if "compatibility" in inner:
        compat = inner["compatibility"]
        ok = "✅" if compat.get("compatible") else "❌"
        # platform is inside compatibility dict, not at top level of pre_flight result
        platform = compat.get("platform", "")
        # cluster GPU count — key name differs across versions
        cluster = inner.get("cluster_resources") or inner.get("cluster") or inner.get("resources") or {}
        gpus = cluster.get("gpu_total", "?")
        estimate = inner.get("resource_estimate") or inner.get("estimate") or {}
        mem = estimate.get("gpu_memory_required") or estimate.get("gpu_memory_gb") or estimate.get("memory_gb")
        model_type = estimate.get("model_type")
        is_moe = estimate.get("is_moe", False)
        architectures = estimate.get("architectures", [])
        # runtimes value can be a dict {available:[...], total:N} or a list
        raw_runtimes = inner.get("runtimes", {})
        if isinstance(raw_runtimes, dict):
            runtimes = raw_runtimes.get("total") or len(raw_runtimes.get("available", []))
        elif isinstance(raw_runtimes, list):
            runtimes = len(raw_runtimes)
        else:
            runtimes = "?"
        recommended = inner.get("tool_selection", {}).get("recommended", "")

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", width=20)
        t.add_column()
        t.add_row(f"{ok} Compatibility", "pass" if compat.get("compatible") else "[red]FAIL[/red]")
        if platform and platform != "kubernetes":
            t.add_row("Platform", f"[yellow]{platform}[/yellow]")
        t.add_row("GPUs", f"[cyan]{gpus}[/cyan] available")
        if mem:
            mem_str = str(mem) if not str(mem).endswith("GB") else mem
            t.add_row("Model memory", f"[cyan]{mem_str}[/cyan] estimated")
        if model_type:
            arch_str = model_type
            if architectures:
                arch_str += f"  ({architectures[0]})"
            if is_moe:
                arch_str += "  [dim]MoE — all expert weights in memory during training[/dim]"
            t.add_row("Architecture", arch_str)
        t.add_row("Runtimes", f"[cyan]{runtimes}[/cyan] available")
        if recommended:
            t.add_row("Recommended", f"[bold]{recommended}()[/bold]")
        c.print(Padding(t, (0, 0, 0, 2)))
        return True

    # ── list_runtimes ─────────────────────────────────────────────────────────
    if "runtimes" in inner and "compatibility" not in inner:
        raw = inner["runtimes"]
        # runtimes can be a list OR a dict like {available:[...], total:N}
        runtimes = raw.get("available", []) if isinstance(raw, dict) else raw
        names = [r.get("name", r) if isinstance(r, dict) else r for r in runtimes]
        status = "✅" if success else "❌"

        # Group names into a 2-column Rich table
        t = Table(show_header=False, box=None, padding=(0, 2), expand=False)
        t.add_column(style="cyan", no_wrap=True)
        t.add_column(style="cyan", no_wrap=True)
        pairs = [names[i : i + 2] for i in range(0, len(names), 2)]
        for pair in pairs:
            t.add_row(*pair) if len(pair) == 2 else t.add_row(pair[0], "")

        c.print(f"  {status} [cyan]{len(names)} runtimes[/cyan]")
        c.print(Padding(t, (0, 0, 0, 4)))
        return True

    # ── list_training_jobs ────────────────────────────────────────────────────
    if "jobs" in inner:
        jobs = inner["jobs"]
        by_status: dict[str, list[str]] = {}
        for j in jobs:
            s = j.get("status", "Unknown") if isinstance(j, dict) else "Unknown"
            by_status.setdefault(s, []).append(j.get("name", "?") if isinstance(j, dict) else str(j))

        t = Table(show_header=True, box=None, padding=(0, 2), header_style="dim")
        t.add_column("Status")
        t.add_column("Job name", style="cyan")
        for s, job_names in by_status.items():
            colour = {"Running": "green", "Complete": "bright_green", "Failed": "red"}.get(s, "yellow")
            for name in job_names:
                t.add_row(f"[{colour}]{s}[/{colour}]", name)

        c.print(f"  ✅ [cyan]{len(jobs)} jobs[/cyan]")
        c.print(Padding(t, (0, 0, 0, 4)))
        return True

    # ── get_training_job ──────────────────────────────────────────────────────
    if "name" in inner and "status" in inner and "runtimes" not in inner:
        name = inner["name"]
        status = inner["status"]
        colour = {"Running": "green", "Complete": "bright_green", "Failed": "red"}.get(status, "yellow")

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", width=12)
        t.add_column()
        t.add_row("Job", f"[cyan]{name}[/cyan]")
        t.add_row("Status", f"[{colour}]{status}[/{colour}]")
        if inner.get("namespace"):
            t.add_row("Namespace", inner["namespace"])
        c.print(Padding(t, (0, 0, 0, 2)))
        return True

    # ── get_training_logs ─────────────────────────────────────────────────────
    if "logs" in inner:
        log_text = str(inner.get("logs", ""))
        lines_count = len(log_text.split("\n"))
        hint = inner.get("failure_hint", "")
        c.print(f"  ✅ [cyan]{lines_count} log lines[/cyan]")
        if hint:
            c.print(f"  [yellow]→ Hint:[/yellow] {hint}")
        return True

    # ── generic error ─────────────────────────────────────────────────────────
    if success is False:
        error = data.get("error", "unknown error")
        c.print(f"  [red]✗[/red] {error}")
        return True

    return False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_args_inline(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        val = json.dumps(v) if not isinstance(v, str) else v
        if len(val) > 40:
            val = val[:37] + "…"
        parts.append(f"{k}={val}")
    joined = "  " + "  ".join(parts)
    return joined if len(joined) < 120 else "  " + "  ".join(parts[:3]) + "  …"


def format_tool_result_display(result: Any, max_lines: int = 15) -> str:
    """Fallback plain-text formatter (used by streaming chunk renderer)."""
    if isinstance(result, dict):
        formatted = json.dumps(result, indent=2, default=str)
    else:
        formatted = str(result)
    lines = formatted.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return formatted


# ── Misc helpers ──────────────────────────────────────────────────────────────


def print_tip(c: Console, message: str, *, style: str = "bright_yellow") -> None:
    c.print()
    c.print(f"[dim]→[/dim] [{style}]{message}[/{style}]")


def print_tools_table(
    c: Console,
    tools: list[tuple[str, str]],
    *,
    header_style: str = "bold cyan",
    title: str = "Active Tools",
) -> None:
    """Render available tools using a Rich Table."""
    c.print()
    t = Table(title=title, title_style=header_style, box=None, padding=(0, 2), show_header=False)
    t.add_column(style="bright_white", no_wrap=True, width=28)
    t.add_column(style="dim")
    for tool_name, description in tools:
        t.add_row(tool_name, description[:70])
    c.print(t)
    c.print()


def print_goodbye(c: Console) -> None:
    c.print()
    c.print("[dim]Goodbye![/dim]")


def print_plain(c: Console, message: str, *, style: str | None = None) -> None:
    if style:
        c.print(Text(message, style=style))
    else:
        c.print(message)


def print_separator(c: Console, *, title: str = "", style: str = "dim") -> None:
    """Rich Rule — auto-sizes to terminal width, safe to resize."""
    if title:
        c.print(Rule(title=f"[{style}]{title}[/{style}]", style=style))
    else:
        c.print(Rule(style=style))


def print_section(c: Console, title: str, *, style: str = "bold") -> None:
    c.print()
    c.print(f"[{style}]{title}[/{style}]")
    c.print()


def print_key_value(
    c: Console,
    key: str,
    value: str,
    *,
    key_style: str = "cyan",
    value_style: str = "white",
) -> None:
    c.print(f"  [{key_style}]{key:20}[/{key_style}] [{value_style}]{value}[/{value_style}]")
