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

"""Preview-before-submit confirm gate — framework-agnostic.

Moved from frameworks/_confirm.py to core/ because it is shared by every
framework adapter and by AgentSession (Gap 3C runtime/session.py).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

ConfirmHandler = Callable[[str, dict[str, Any]], bool]

_handler: ConfirmHandler | None = None


def set_confirm_handler(handler: ConfirmHandler | None) -> None:
    global _handler
    _handler = handler


def make_console_confirm_handler(console: Any) -> ConfirmHandler:
    """Rich y/n prompt before mutating tools with confirmed=False."""

    def ask(tool_name: str, args: dict[str, Any]) -> bool:
        from rich.panel import Panel
        from rich.prompt import Confirm
        from rich.table import Table

        # Only show non-null, non-confirmed fields so the panel is concise
        preview = {k: v for k, v in args.items() if k != "confirmed" and v is not None}

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", min_width=20)
        t.add_column(style="bright_yellow")
        for k, v in preview.items():
            val = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
            if len(val) > 120:
                val = val[:120] + "…"
            t.add_row(k, val)

        console.print()
        console.print(
            Panel(
                t,
                title=f"[bold red]⚠  Preview — {tool_name}[/bold red]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return Confirm.ask("  Submit?", default=False)

    return ask


def _declined() -> dict[str, Any]:
    return {"cancelled": True, "reason": "User declined the mutating action"}


def wrap_with_confirm(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Audit-wrap and pause on confirmed=False (direct tools and execute_tool)."""
    from kubeflow_mcp.agents.core.tools import audit_wrap

    wrapped = audit_wrap(fn)
    sig = inspect.signature(fn)

    if fn.__name__ == "execute_tool":

        def execute_tool(
            *,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    args = parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = dict(arguments or {})
            args.update({k: v for k, v in kwargs.items() if k not in ("tool_name", "arguments")})
            if args.get("confirmed") is False:
                if _handler is None or not _handler(tool_name, args):
                    return _declined()
                args["confirmed"] = True
            return wrapped(tool_name=tool_name, arguments=args)

        execute_tool.__name__ = fn.__name__
        execute_tool.__doc__ = fn.__doc__
        return execute_tool

    if "confirmed" not in sig.parameters:
        return wrapped

    import functools

    @functools.wraps(fn)  # preserves __wrapped__ chain so StructuredTool sees original signature
    def run(**kwargs: Any) -> Any:
        if kwargs.get("confirmed") is False:
            if _handler is None or not _handler(fn.__name__, kwargs):
                return _declined()
            kwargs = {**kwargs, "confirmed": True}
        return wrapped(**kwargs)

    return run


class ConfirmMiddleware:
    """TurnMiddleware that installs the console confirm handler before each turn.

    Ensures all tools executed within the turn share one gate without
    any framework adapter needing to call set_confirm_handler directly.
    """

    def __init__(self, console: Any) -> None:
        self._console = console

    def __call__(self, ctx: Any, next: Callable[..., Any]) -> Any:
        set_confirm_handler(make_console_confirm_handler(self._console))
        try:
            return next(ctx)
        finally:
            set_confirm_handler(None)
