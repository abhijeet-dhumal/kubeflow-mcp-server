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

"""Interactive REPL for the LiteLLM agent.

Slash commands available at the prompt:
    /tools              list active tools with descriptions
    /model <name>       switch model mid-session (e.g. /model gpt-4.1)
    /mode <mode>        switch tool mode: full | progressive | semantic
    /think              toggle thinking/extended reasoning mode
    /eval <file.jsonl>  run a golden-prompt evaluation batch
    /export             save the current session to a timestamped JSON file
    /clear              reset conversation (keep system prompt)
    exit | quit | q     leave the REPL
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SPHINX_BUILD = "sphinx" in sys.modules

try:
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    if not _SPHINX_BUILD:
        sys.exit("Error: install kubeflow-mcp[agents-litellm]")
    Panel = None  # type: ignore[misc, assignment]
    Prompt = None  # type: ignore[misc, assignment]
    Table = None  # type: ignore[misc, assignment]
    Text = None  # type: ignore[misc, assignment]

from kubeflow_mcp.agents.litellm_agent import LiteLLMAgent  # noqa: E402
from kubeflow_mcp.agents.observability import MlflowSessionLogger  # noqa: E402
from kubeflow_mcp.agents.runtime.repl_commands import (  # noqa: E402
    REPL_EXIT_COMMANDS,
    CommonReplHandlers,
    dispatch_slash_command,
)
from kubeflow_mcp.agents.runtime.session_state import (  # noqa: E402
    export_session_snapshot,
    import_session_snapshot,
)
from kubeflow_mcp.agents.terminal_ui import (  # noqa: E402
    format_tool_result_display,
    get_console,
    print_assistant_panel,
    print_error_panel,
    print_goodbye,
    print_plain,
    print_tip,
    print_tool_call_panel,
    print_tool_result_panel,
    print_tools_table,
    print_user_panel,
    print_welcome_panel,
    setup_readline_history,
)


def _confirm_panel(console, name: str, args: dict[str, Any]) -> None:
    preview = json.dumps(args, indent=2, default=str)
    console.print()
    console.print(
        Panel(
            Text(preview, style="bright_white"),
            title=(
                f"[bold bright_cyan]⏸  Preview: {name}[/bold bright_cyan]\n"
                "[dim]This action requires your approval[/dim]"
            ),
            border_style="cyan",
            padding=(1, 2),
        )
    )


# ─── Turn runner ─────────────────────────────────────────────────────────────


async def _run_turn(agent: LiteLLMAgent, user_message: str, console) -> None:
    """Drive one full agentic turn, rendering events to the console."""
    text_buf = ""

    async def _drain_events(event_stream) -> None:
        nonlocal text_buf
        async for event_type, data in event_stream:
            if event_type == "text_delta":
                text_buf += data
            elif event_type == "tool_call":
                if text_buf.strip():
                    print_assistant_panel(console, text_buf)
                    text_buf = ""
                print_tool_call_panel(console, data["name"], data["args"])
            elif event_type == "tool_result":
                print_tool_result_panel(console, format_tool_result_display(data["result"]))
            elif event_type == "confirm_needed":
                if text_buf.strip():
                    print_assistant_panel(console, text_buf)
                    text_buf = ""
                _confirm_panel(console, data["name"], data["args"])
                choice = Prompt.ask("  Submit?", choices=["y", "n"], default="n")
                approved = choice == "y"
                await _drain_events(agent.continue_after_confirm(approved))
                return
            elif event_type == "error":
                if text_buf.strip():
                    print_assistant_panel(console, text_buf)
                    text_buf = ""
                message = data.get("message", "Unknown LiteLLM error")
                err_type = data.get("type", "LiteLLMError")
                print_error_panel(console, RuntimeError(f"{err_type}: {message}"))
            elif event_type == "done":
                break

    await _drain_events(agent._agentic_loop(user_message))

    if text_buf.strip():
        print_assistant_panel(console, text_buf)


# ─── Slash-command handlers ───────────────────────────────────────────────────


def _slash_tools(agent: LiteLLMAgent, console) -> None:
    print_tools_table(
        console,
        [(schema["function"]["name"], schema["function"].get("description", "")) for schema in agent._tool_schemas],
        header_style="bold cyan",
    )
    print_tip(console, f"Mode: {agent.tool_mode}  |  Total tools: {len(agent._tool_schemas)}")


def _slash_model(args: list[str], agent: LiteLLMAgent, console) -> None:
    if not args:
        print_tip(console, f"Current model : {agent.model}")
        print_tip(console, "Usage: /model ollama/gemma4:e4b  or  /model gpt-4.1")
        return
    agent.switch_model(args[0])
    print_tip(console, f"Switched model → {args[0]}")


def _slash_mode(args: list[str], agent: LiteLLMAgent, console) -> None:
    if not args:
        print_tip(console, f"Current mode : {agent.tool_mode}")
        print_tip(console, "Usage: /mode full  |  /mode progressive  |  /mode semantic")
        return
    try:
        agent.switch_mode(args[0])
        print_tip(console, f"Switched mode → {args[0]}  ({len(agent._tool_schemas)} tools)")
    except ValueError as exc:
        print_tip(console, f"Error: {exc}", style="red")


def _slash_eval(args: list[str], agent: LiteLLMAgent, console, loop: asyncio.AbstractEventLoop) -> None:
    if not args:
        print_tip(console, "Usage: /eval golden_prompts.jsonl")
        return
    file_path = Path(args[0])
    if not file_path.exists():
        print_tip(console, f"File not found: {file_path}", style="red")
        return

    try:
        import litellm as _litellm

        prompts: list[dict] = []
        with open(file_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        prompts.append(json.loads(line))
                    except json.JSONDecodeError:
                        prompts.append({"prompt": line})

        console.print(f"[dim]Running {len(prompts)} prompts against {agent.model}…[/dim]")

        messages_list = [
            [{"role": "user", "content": p.get("prompt", str(p))}] for p in prompts
        ]
        extra = {"base_url": agent._base_url} if agent._base_url else {}
        results = loop.run_until_complete(
            _litellm.abatch_completion(model=agent.model, messages=messages_list, **extra)
        )

        correct = 0
        eval_rows: list[dict] = []
        for res, prompt in zip(results, prompts, strict=False):
            completion = res.choices[0].message.content or ""
            expected = prompt.get("expected", prompt.get("completion", ""))
            ok = bool(expected) and expected.strip().lower() in completion.lower()
            if ok:
                correct += 1
            eval_rows.append({"prompt": prompt.get("prompt", ""), "completion": completion, "correct": ok})

        out = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w") as fh:
            json.dump({"model": agent.model, "results": eval_rows}, fh, indent=2)

        pct = 100 * correct // len(prompts) if prompts else 0
        print_tip(console, f"Correctness: {correct}/{len(prompts)} ({pct}%)  →  {out}")
    except Exception as exc:
        print_tip(console, f"Eval error: {exc}", style="red")
        logger.debug("Eval error", exc_info=True)


def _slash_export(agent: LiteLLMAgent, console) -> None:
    session = agent.export_session()
    out = export_session_snapshot(session)
    msgs = len(session["messages"])
    cost = session.get("total_cost_usd", 0.0)
    print_tip(console, f"Session exported → {out}  ({msgs} messages, ${cost:.4f})")


def _slash_import(args: list[str], agent: LiteLLMAgent, console) -> None:
    if not args:
        print_tip(console, "Usage: /import <session.json>")
        return
    try:
        payload = import_session_snapshot(args[0])
        loaded = agent.import_session(payload)
        print_tip(console, f"Session imported ← {args[0]}  ({loaded} messages)")
    except Exception as exc:
        print_tip(console, f"Import error: {exc}", style="red")


def _dispatch_slash(
    line: str,
    agent: LiteLLMAgent,
    console,
    loop: asyncio.AbstractEventLoop,
) -> bool:
    """Handle a slash command. Returns True if the REPL should exit."""
    def _custom_dispatch(command: str, args: list[str]) -> bool:
        if command == "/model":
            _slash_model(args, agent, console)
            return True
        if command == "/mode":
            _slash_mode(args, agent, console)
            return True
        if command == "/eval":
            _slash_eval(args, agent, console, loop)
            return True
        if command == "/import":
            _slash_import(args, agent, console)
            return True
        return False

    def _on_think() -> None:
        agent._thinking = not agent._thinking
        state = "on" if agent._thinking else "off"
        print_tip(console, f"Thinking mode: {state}")

    def _on_clear() -> None:
        agent.clear()
        print_tip(console, "Conversation cleared.")

    handlers = CommonReplHandlers(
        on_tools=lambda: _slash_tools(agent, console),
        on_think=_on_think,
        on_export=lambda: _slash_export(agent, console),
        on_import=lambda _path: None,
        on_clear=_on_clear,
        on_unknown=lambda command: print_tip(console, f"Unknown command: {command!r}. Type /tools for help."),
    )
    return dispatch_slash_command(
        line,
        handlers=handlers,
        custom_dispatch=_custom_dispatch,
        exit_commands=REPL_EXIT_COMMANDS,
    )


# ─── Public entry point ───────────────────────────────────────────────────────


def run_litellm_chat(
    model: str,
    tool_mode: str = "full",
    base_url: str | None = None,
    fallback_model: str | None = None,
    thinking: bool = True,
    num_retries: int = 3,
    **_kwargs: Any,
) -> None:
    """Launch the blocking interactive REPL.

    Args:
        model: LiteLLM model string (e.g. ``"ollama/gemma4:e4b"``).
        tool_mode: ``"full"`` | ``"progressive"`` | ``"semantic"``.
        base_url: Override LiteLLM base URL for on-prem / local endpoints.
        fallback_model: Optional fallback model tried when primary fails.
    """
    setup_readline_history()
    console = get_console()

    agent = LiteLLMAgent(
        model=model,
        tool_mode=tool_mode,
        base_url=base_url,
        fallback_model=fallback_model,
        thinking=thinking,
        num_retries=num_retries,
    )
    mlflow_logger = MlflowSessionLogger(model=model, tool_mode=tool_mode, framework="litellm")

    backend_label = base_url or "cloud / local auto-detect"
    print_welcome_panel(
        panel_title="kubeflow-mcp · LiteLLM Agent",
        border_style="bright_green",
        rows=[
            ("bold bright_green", "Kubeflow MCP — LiteLLM Agent"),
            ("white", f"Model   : {model}"),
            ("white", f"Backend : {backend_label}"),
            ("white", f"Mode    : {tool_mode}  ({len(agent._tool_schemas)} tools)"),
            ("dim", ""),
            ("dim", "Commands: /tools  /model <name>  /mode <mode>  /think"),
            ("dim", "          /eval <file>  /export  /import <file>  /clear  exit"),
            ("dim", f"Thinking: {'on' if thinking else 'off'} ( /think to toggle )"),
            *(
                [("dim", f"MLflow run: {mlflow_logger.run_id[:8]}…")]
                if mlflow_logger.enabled and mlflow_logger.run_id
                else []
            ),
        ],
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                console.print()
                console.print("[bold bright_blue]You[/bold bright_blue] ", end="")
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                print_goodbye(console)
                break

            if not line:
                continue

            if line.lower() in REPL_EXIT_COMMANDS:
                print_goodbye(console)
                break

            if line.startswith("/"):
                should_exit = _dispatch_slash(line, agent, console, loop)
                if should_exit:
                    print_goodbye(console)
                    break
                continue

            print_user_panel(console, line)
            try:
                tokens_before = agent.token_count()
                import time as _time

                _turn_start = _time.monotonic()
                loop.run_until_complete(_run_turn(agent, line, console))
                _turn_dur = _time.monotonic() - _turn_start
                tokens = agent.token_count()
                cost = agent._total_cost
                print_plain(
                    console,
                    f"[dim]tokens≈{tokens}  cost≈${cost:.4f}[/dim]",
                )
                if mlflow_logger is not None:
                    # LiteLLM agent exposes total token count only; derive delta.
                    token_delta = max(0, tokens - tokens_before)
                    mlflow_logger.log_turn(
                        user_input=line,
                        assistant_output="",
                        input_tokens=token_delta,
                        duration_s=_turn_dur,
                    )
            except KeyboardInterrupt:
                print_tip(console, "\nInterrupted. Type 'exit' to quit.")
            except Exception as exc:
                print_error_panel(console, exc)
                logger.debug("Turn error", exc_info=True)
    finally:
        mlflow_logger.close()
        loop.close()
