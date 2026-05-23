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

"""smolagents ToolCallingAgent backend routed through LiteLLM."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from typing import Any

from kubeflow_mcp.agents.frameworks._confirm import (
    make_console_confirm_handler,
    set_confirm_handler,
    wrap_with_confirm,
)
from kubeflow_mcp.agents.frameworks._thinking import apply_thinking_to_litellm_model
from kubeflow_mcp.agents.frameworks._tools import get_system_prompt, load_tools
from kubeflow_mcp.agents.litellm_agent import build_tool_schema
from kubeflow_mcp.agents.observability import MlflowSessionLogger
from kubeflow_mcp.agents.runtime.repl_commands import (
    REPL_EXIT_COMMANDS,
    CommonReplHandlers,
    handle_common_repl_command,
)
from kubeflow_mcp.agents.runtime.session_state import (
    build_session_snapshot,
    export_session_snapshot,
    import_session_snapshot,
)

_SPHINX_BUILD = "sphinx" in sys.modules

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    if not _SPHINX_BUILD:
        sys.exit("Error: install kubeflow-mcp[agents-smolagents]")
    Panel = None  # type: ignore[misc, assignment]
    Table = None  # type: ignore[misc, assignment]
    Text = None  # type: ignore[misc, assignment]

from kubeflow_mcp.agents.terminal_ui import (  # noqa: E402
    format_tool_result_display,
    get_console,
    print_assistant_panel,
    print_error_panel,
    print_goodbye,
    print_tip,
    print_tool_call_panel,
    print_tool_result_panel,
    print_tools_table,
    print_user_panel,
    print_welcome_panel,
    setup_readline_history,
)


def _schema_to_smolagents_inputs(
    schema: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    props = schema["function"]["parameters"].get("properties", {})
    required = set(schema["function"]["parameters"].get("required", []))
    inputs: dict[str, dict[str, Any]] = {}
    for name, prop in props.items():
        entry: dict[str, Any] = {
            "type": prop.get("type", "string"),
            "description": prop.get("description") or name,
        }
        if name not in required:
            entry["nullable"] = True
        inputs[name] = entry
    return inputs, required


def _build_forward(
    wrapped_fn: Callable[..., Any],
    param_names: list[str],
    required: set[str],
) -> Callable[..., str]:
    """Build a forward(self, ...) with explicit params — smolagents validates signatures."""
    param_defs: list[str] = []
    for name in param_names:
        param_defs.append(name if name in required else f"{name}=None")

    kwargs_expr = ", ".join(f'"{n}": {n}' for n in param_names)
    src = f"""
def forward(self, {", ".join(param_defs)}) -> str:
    args = {{{kwargs_expr}}}
    args = {{k: v for k, v in args.items() if v is not None}}
    result = _wrapped(**args)
    return json.dumps(result, default=str)
"""
    namespace: dict[str, Any] = {"json": json, "_wrapped": wrapped_fn}
    exec(src, namespace)  # noqa: S102
    return namespace["forward"]


def _make_smolagents_tool(fn: Callable[..., Any], description: str, wrapped_fn: Callable[..., Any]):
    from smolagents import Tool

    schema = build_tool_schema(fn, description)
    inputs, required = _schema_to_smolagents_inputs(schema)
    tool_name = fn.__name__
    tool_desc = description or schema["function"]["description"]
    param_names = list(inputs.keys())
    forward = _build_forward(wrapped_fn, param_names, required)

    return type(
        f"KFTool_{tool_name}",
        (Tool,),
        {
            "name": tool_name,
            "description": tool_desc,
            "inputs": inputs,
            "output_type": "string",
            "forward": forward,
        },
    )()


def _extract_tool_call(step: Any) -> tuple[str, dict[str, Any]] | None:
    """Pull the first non-final_answer tool call from a smolagents step."""
    tool_calls = getattr(step, "tool_calls", None) or []
    for tc in tool_calls:
        name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
        if not name or name == "final_answer":
            continue
        raw_args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"raw": raw_args}
        else:
            args = raw_args or {}
        return name, args
    return None


def _parse_final_answer(step: Any) -> str:
    for tc in getattr(step, "tool_calls", None) or []:
        name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
        if name != "final_answer":
            continue
        raw_args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                return raw_args
        if isinstance(raw_args, dict):
            return str(raw_args.get("answer", ""))
        return str(raw_args)
    return ""


def _run_turn_streaming(agent: Any, line: str, *, reset: bool, console) -> tuple[str, int]:
    """Run with stream=True; render tool panels, return (answer, step_count)."""
    final = ""
    steps = 0
    for step in agent.run(line, stream=True, reset=reset):
        steps += 1
        step_name = type(step).__name__
        if step_name == "ActionStep":
            answer = _parse_final_answer(step)
            if answer:
                final = answer
                continue
            call = _extract_tool_call(step)
            if call:
                name, args = call
                print_tool_call_panel(console, name, args)
            observation = getattr(step, "observations", None)
            if observation and call:
                print_tool_result_panel(console, format_tool_result_display(observation))
        elif step_name == "FinalAnswerStep":
            final = str(getattr(step, "output", "") or final)
    return final, steps


def _run_turn_blocking(agent: Any, line: str, *, reset: bool) -> str:
    """Fallback when stream mode is unavailable."""
    result = agent.run(line, reset=reset, return_full_result=True)
    if hasattr(result, "output"):
        return str(result.output)
    return str(result)


def _print_turn_stats(console, turn: _TurnSnapshot) -> None:
    console.print(
        f"[dim]steps={turn.steps}  duration={turn.duration:.1f}s  "
        f"tokens in={turn.last_input:,} out={turn.last_output:,}  "
        f"(session in={turn.session_input:,} out={turn.session_output:,})[/dim]"
    )


class _TurnSnapshot:
    def __init__(self) -> None:
        self.steps = 0
        self.duration = 0.0
        self.session_input = 0
        self.session_output = 0
        self.last_input = 0
        self.last_output = 0
        self._start = 0.0
        self._base_input = 0
        self._base_output = 0

    def begin(self, agent: Any) -> None:
        self._start = time.monotonic()
        self.steps = 0
        monitor = getattr(agent, "monitor", None)
        if monitor is not None:
            tokens = monitor.get_total_token_counts()
            self._base_input = tokens.input_tokens
            self._base_output = tokens.output_tokens

    def finish(self, agent: Any, *, steps: int) -> None:
        self.duration = time.monotonic() - self._start
        self.steps = steps
        self.last_input, self.last_output = self.token_delta(agent)
        self.session_input += self.last_input
        self.session_output += self.last_output

    def token_delta(self, agent: Any | None = None) -> tuple[int, int]:
        monitor = getattr(agent, "monitor", None) if agent else None
        if monitor is None:
            return 0, 0
        tokens = monitor.get_total_token_counts()
        return (
            max(0, tokens.input_tokens - self._base_input),
            max(0, tokens.output_tokens - self._base_output),
        )


def _reset_agent_memory(agent: Any) -> None:
    reset_fn = getattr(agent.memory, "reset", None)
    if callable(reset_fn):
        reset_fn()


def _import_smolagents_tokens(path: str, turn: _TurnSnapshot) -> None:
    payload = import_session_snapshot(path)
    tokens = payload.get("tokens", {})
    if isinstance(tokens, dict):
        turn.session_input = int(tokens.get("input", 0) or 0)
        turn.session_output = int(tokens.get("output", 0) or 0)


def _run_smolagents_repl(  # noqa: C901
    *,
    console,
    agent_holder: list[Any],
    logger: Any,
    smol_tools: list[Any],
    model: str,
    tool_mode: str,
    thinking_holder: list[bool],
    rebuild_agent: Callable[[], None],
    apply_thinking_to_existing_memory: Callable[[], None],
    mlflow_logger: MlflowSessionLogger | None = None,
) -> None:
    from smolagents import LogLevel

    turn = _TurnSnapshot()
    reset_next = True

    def _on_tools() -> None:
        print_tools_table(
            console,
            [(tool.name, getattr(tool, "description", "") or "") for tool in smol_tools],
            header_style="bold magenta",
        )
        print_tip(console, f"Mode: {tool_mode}  |  Total tools: {len(smol_tools)}")

    def _on_clear() -> None:
        nonlocal turn, reset_next
        agent = agent_holder[0]
        _reset_agent_memory(agent)
        agent.monitor.reset()
        turn = _TurnSnapshot()
        reset_next = True
        print_tip(console, "Conversation cleared.")

    def _on_think() -> None:
        thinking_holder[0] = not thinking_holder[0]
        apply_thinking_to_existing_memory()
        state = "on" if thinking_holder[0] else "off"
        print_tip(console, f"Thinking mode: {state}")

    def _on_export() -> None:
        session = build_session_snapshot(
            model=model,
            framework="smolagents",
            tool_mode=tool_mode,
            token_input=turn.session_input,
            token_output=turn.session_output,
        )
        out = export_session_snapshot(session)
        print_tip(console, f"Session exported → {out}")

    def _on_import(path: str) -> None:
        nonlocal turn, reset_next
        try:
            _on_clear()
            _import_smolagents_tokens(path, turn)
            print_tip(
                console,
                f"Session imported ← {path}  (token totals restored; chat replay is not available in smolagents)",
            )
        except Exception as exc:
            print_tip(console, f"Import error: {exc}", style="red")

    def _on_unknown(command: str) -> None:
        print_tip(console, f"Unknown command: {command!r}. Try /tools, /think, /export, /import, /clear.")

    common_handlers = CommonReplHandlers(
        on_tools=_on_tools,
        on_think=_on_think,
        on_export=_on_export,
        on_import=_on_import,
        on_clear=_on_clear,
        on_unknown=_on_unknown,
    )

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

        if handle_common_repl_command(line, common_handlers):
            continue

        print_user_panel(console, line)
        try:
            agent = agent_holder[0]
            turn.begin(agent)
            prev_level = logger.level
            logger.level = LogLevel.ERROR
            try:
                try:
                    answer, step_count = _run_turn_streaming(
                        agent, line, reset=reset_next, console=console
                    )
                except TypeError:
                    answer = _run_turn_blocking(agent, line, reset=reset_next)
                    step_count = 1
            finally:
                logger.level = prev_level

            turn.finish(agent, steps=step_count)
            if answer.strip():
                print_assistant_panel(console, answer)
            _print_turn_stats(console, turn)
            if mlflow_logger is not None:
                mlflow_logger.log_turn(
                    user_input=line,
                    assistant_output=answer,
                    llm_call_count=turn.steps,
                    input_tokens=turn.last_input,
                    output_tokens=turn.last_output,
                    duration_s=turn.duration,
                )
            reset_next = False
        except KeyboardInterrupt:
            print_tip(console, "\nInterrupted.")
        except Exception as exc:
            print_error_panel(console, exc)


def run_smolagents_chat(
    model: str,
    tool_mode: str = "full",
    base_url: str | None = None,
    thinking: bool = True,
    num_retries: int = 3,
    **_kwargs: Any,
) -> None:
    """Launch smolagents ToolCallingAgent with Kubeflow tools and LiteLLM routing."""
    try:
        from smolagents import AgentLogger, LiteLLMModel, LogLevel, ToolCallingAgent
    except ImportError as exc:
        msg = "Install optional deps: uv sync --extra agents-smolagents"
        raise RuntimeError(msg) from exc

    setup_readline_history()
    console = get_console()
    set_confirm_handler(make_console_confirm_handler(console))
    logger = AgentLogger(level=LogLevel.INFO, console=console)
    mlflow_logger = MlflowSessionLogger(model=model, tool_mode=tool_mode, framework="smolagents")

    tool_fns, descriptions = load_tools(tool_mode)
    smol_tools = [
        _make_smolagents_tool(fn, descriptions.get(fn.__name__, ""), wrap_with_confirm(fn))
        for fn in tool_fns
    ]
    thinking_holder = [thinking]
    agent_holder: list[Any] = [None]

    def _build_agent() -> Any:
        model_kwargs: dict[str, Any] = {"model_id": model, "num_retries": num_retries}
        if base_url:
            model_kwargs["api_base"] = base_url
        apply_thinking_to_litellm_model(model_kwargs, enabled=thinking_holder[0], model=model)
        llm = LiteLLMModel(**model_kwargs)
        return ToolCallingAgent(
            tools=smol_tools,
            model=llm,
            instructions=get_system_prompt(),
            logger=logger,
            add_base_tools=False,
        )

    def rebuild_agent() -> None:
        agent_holder[0] = _build_agent()

    def apply_thinking_to_existing_memory() -> None:
        old_agent = agent_holder[0]
        new_agent = _build_agent()
        old_memory = getattr(old_agent, "memory", None)
        new_memory = getattr(new_agent, "memory", None)
        if old_memory is not None and new_memory is not None:
            if hasattr(old_memory, "steps") and hasattr(new_memory, "steps"):
                new_memory.steps = list(old_memory.steps)
            if hasattr(old_memory, "system_prompt") and hasattr(new_memory, "system_prompt"):
                new_memory.system_prompt = old_memory.system_prompt
        old_monitor = getattr(old_agent, "monitor", None)
        new_monitor = getattr(new_agent, "monitor", None)
        if old_monitor is not None and new_monitor is not None:
            for field in ("total_input_token_count", "total_output_token_count"):
                if hasattr(old_monitor, field) and hasattr(new_monitor, field):
                    setattr(new_monitor, field, getattr(old_monitor, field))
        agent_holder[0] = new_agent

    rebuild_agent()

    backend_label = base_url or "cloud / local auto-detect"
    print_welcome_panel(
        panel_title="kubeflow-mcp · smolagents",
        border_style="bright_magenta",
        rows=[
            ("bold bright_magenta", "Kubeflow MCP — smolagents + LiteLLM"),
            ("white", f"Model   : {model}"),
            ("white", f"Backend : {backend_label}"),
            ("white", f"Mode    : {tool_mode}  ({len(smol_tools)} tools)"),
            ("dim", ""),
            ("dim", "Commands: /tools  /think  /export  /import <file>  /clear  exit"),
            ("dim", f"Thinking: {'on' if thinking else 'off'} ( /think to toggle )"),
            ("dim", "Native: ToolCallingAgent + LiteLLMModel + monitor"),
            ("dim", "Confirm gate on mutating tools (confirmed=False)"),
            *(
                [("dim", f"MLflow run: {mlflow_logger.run_id[:8]}…")]
                if mlflow_logger.enabled and mlflow_logger.run_id
                else []
            ),
        ],
    )

    try:
        _run_smolagents_repl(
            console=console,
            agent_holder=agent_holder,
            logger=logger,
            smol_tools=smol_tools,
            model=model,
            tool_mode=tool_mode,
            thinking_holder=thinking_holder,
            rebuild_agent=rebuild_agent,
            apply_thinking_to_existing_memory=apply_thinking_to_existing_memory,
            mlflow_logger=mlflow_logger,
        )
    finally:
        mlflow_logger.close()
