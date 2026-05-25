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

"""AgentSession — the single shared REPL for all framework runners (Gap 3C).

Replaces the per-framework REPL loops (run_langchain_chat, run_litellm_repl,
etc.) with one shared implementation.  Each framework contributes only a
``TurnRunner`` adapter; everything else (printing, confirm gate, OTel, MLflow,
keyboard shortcuts) lives here.

Usage::

    runner = LangChainRunner(model=model, tool_mode=tool_mode, ...)
    session = AgentSession(
        runner=runner,
        middleware=[
            UsageMiddleware(),
            OTelMiddleware(framework="langchain"),
            MLflowMiddleware(logger=mlflow_logger),
            ConfirmMiddleware(console=console),
        ],
        console=console,
    )
    session.run()
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from kubeflow_mcp.agents.runtime.contracts import TurnContext, TurnMiddleware, TurnRunner, build_chain
from kubeflow_mcp.agents.runtime.repl_commands import REPL_EXIT_COMMANDS



class AgentSession:
    """Framework-agnostic interactive REPL.

    Args:
        runner: A TurnRunner implementation for the chosen framework.
        middleware: Ordered list of TurnMiddleware (outermost first).
        console: Rich console for I/O; defaults to a plain console if omitted.
        session_id: Optional explicit session ID; auto-generated if not given.
        model: Model name passed into TurnContext (informational).
        tool_mode: Tool mode name passed into TurnContext.
        welcome_text: Optional text printed at startup.
        prompt: REPL prompt string (default ``"> "``).
        command_handler: Called with raw input before dispatching to the runner.
            Return ``True`` when the command was handled (turn is skipped).
        input_guard: Called with raw input before the turn.  Return a non-empty
            string to display instead of running the turn (pre-flight hint).
        extras: Framework-specific extras forwarded to every ``TurnContext``.
    """

    def __init__(
        self,
        *,
        runner: TurnRunner,
        middleware: list[TurnMiddleware] | None = None,
        console: Any = None,
        session_id: str | None = None,
        model: str = "unknown",
        tool_mode: str = "full",
        welcome_text: str = "",
        prompt: str = "> ",
        command_handler: Callable[[str], bool] | None = None,
        input_guard: Callable[[str], str | None] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self._runner = runner
        self._middleware = middleware or []
        self._console = console or self._default_console()
        self._session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        self._model = model
        self._tool_mode = tool_mode
        self._welcome_text = welcome_text
        self._prompt = prompt
        self._command_handler = command_handler
        self._input_guard = input_guard
        self._extras: dict[str, Any] = {"console": self._console, **(extras or {})}
        self._chain: Callable[[TurnContext], Any] = build_chain(runner, self._middleware)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the blocking REPL loop."""
        if self._welcome_text:
            self._console.print(self._welcome_text)

        try:
            while True:
                line = self._read_line()
                if line is None:
                    self._print_goodbye()
                    break
                line = line.strip()
                if not line:
                    continue
                if line.lower() in REPL_EXIT_COMMANDS:
                    self._print_goodbye()
                    break
                self._execute_turn(line)
        except KeyboardInterrupt:
            self._console.print("\n[dim]Interrupted[/dim]")
        finally:
            self._on_close()

    def execute_turn(self, user_input: str) -> Any:
        """Execute a single turn programmatically (for testing / eval)."""
        return self._execute_turn(user_input)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _execute_turn(self, user_input: str) -> Any:
        # Slash commands are handled by the runner's command_handler (framework-specific).
        if self._command_handler and self._command_handler(user_input):
            return None

        # Input guard — e.g. preflight hint when the user asks about training without running pre_flight.
        if self._input_guard:
            guard_msg = self._input_guard(user_input)
            if guard_msg:
                try:
                    from kubeflow_mcp.agents.terminal_ui import print_assistant_panel
                    print_assistant_panel(self._console, guard_msg)
                except Exception:
                    self._console.print(guard_msg)
                return None

        ctx = TurnContext(
            user_input=user_input,
            session_id=self._session_id,
            model=self._model,
            tool_mode=self._tool_mode,
            extras=self._extras,
        )
        try:
            result = self._chain(ctx)
            return result
        except KeyboardInterrupt:
            self._console.print("\n[dim]Turn cancelled[/dim]")
            return None
        except Exception as exc:
            try:
                from kubeflow_mcp.agents.terminal_ui import print_error_panel
                print_error_panel(self._console, exc)
            except Exception:
                self._console.print(f"[bold red]Error:[/bold red] {exc}")
            return None

    def _read_line(self) -> str | None:
        """Print the styled input prompt then read one line with readline history.

        Uses sys.stdout directly (not Rich console) to guarantee the prompt
        is flushed before input() blocks — Rich's internal buffering after a
        Live/status context can swallow console.print(end="") partial lines.
        """
        import sys

        try:
            self._console.print()
            sys.stdout.write("\033[1;96m❯\033[0m ")
            sys.stdout.flush()
            line = input()
            self._console.print()
            return line
        except EOFError:
            return None

    def _print_goodbye(self) -> None:
        try:
            from kubeflow_mcp.agents.observability.middleware import UsageMiddleware
            from kubeflow_mcp.agents.terminal_ui import print_goodbye

            for mw in self._middleware:
                if isinstance(mw, UsageMiddleware) and mw.totals["turns"] > 0:
                    t = mw.totals
                    cost = f"  cost=${t['total_cost']:.4f}" if t["total_cost"] > 0 else ""
                    self._console.print(
                        f"[dim]Session: {int(t['turns'])} turns  "
                        f"in={int(t['prompt_tokens'])}  "
                        f"out={int(t['completion_tokens'])}{cost}[/dim]"
                    )
                    break
            print_goodbye(self._console)
        except Exception:
            self._console.print("[dim]Goodbye.[/dim]")

    def _on_close(self) -> None:
        # Close runner first (e.g. LiteLLMRunner shuts down its event loop).
        runner_close = getattr(self._runner, "close", None)
        if callable(runner_close):
            try:
                runner_close()
            except Exception:
                pass
        # Then close each middleware (e.g. MLflowMiddleware ends the run).
        for mw in self._middleware:
            close = getattr(mw, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    @staticmethod
    def _default_console() -> Any:
        try:
            from rich.console import Console

            return Console()
        except ImportError:

            class _Plain:
                def print(self, *args: Any, **_: Any) -> None:
                    print(*args)  # noqa: T201

                def input(self, prompt: str = "") -> str:
                    return input(prompt)

            return _Plain()
