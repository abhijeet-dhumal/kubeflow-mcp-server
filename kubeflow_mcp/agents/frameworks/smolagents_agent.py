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
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from kubeflow_mcp.agents.core.confirm import wrap_with_confirm
from kubeflow_mcp.agents.core.tools import get_system_prompt, load_tools
from kubeflow_mcp.agents.frameworks._thinking import apply_thinking_to_litellm_model
from kubeflow_mcp.agents.core.schema import build_tool_schema
from kubeflow_mcp.agents.observability import MlflowSessionLogger
from kubeflow_mcp.agents.observability.middleware import (
    LangfuseMiddleware,
    MLflowMiddleware,
    OTelMiddleware,
    UsageMiddleware,
)
from kubeflow_mcp.agents.core.confirm import ConfirmMiddleware
from kubeflow_mcp.agents.runtime.session import AgentSession
from kubeflow_mcp.agents.runtime.repl_commands import (
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
    print_tip,
    print_tool_call_panel,
    print_tool_result_panel,
    print_tools_table,
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




def run_smolagents_chat(
    model: str,
    tool_mode: str = "full",
    base_url: str | None = None,
    thinking: bool = True,
    num_retries: int = 3,
    langfuse: bool = False,
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
        panel_title="kubeflow-mcp · LiteLLM · smolagents",
        border_style="bright_magenta",
        rows=[
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

    runner = SmolagentsRunner(
        agent_holder=agent_holder,
        smol_tool_list=smol_tools,
        logger=logger,
        model=model,
        tool_mode=tool_mode,
        thinking_holder=thinking_holder,
        rebuild_agent_fn=rebuild_agent,
        apply_thinking_fn=apply_thinking_to_existing_memory,
        console=console,
    )

    session_id = f"smol-{os.urandom(6).hex()}"
    langfuse_mw = LangfuseMiddleware(session_id=session_id, model=model, framework="smolagents") if langfuse else None
    session = AgentSession(
        runner=runner,
        middleware=[
            UsageMiddleware(),
            OTelMiddleware(framework="smolagents"),
            MLflowMiddleware(mlflow_logger),
            *(([langfuse_mw]) if langfuse_mw else []),
            ConfirmMiddleware(console),
        ],
        console=console,
        model=model,
        tool_mode=tool_mode,
        command_handler=runner.handle_command,
    )
    session.run()


# ─── SmolagentsRunner (TurnRunner adapter) ────────────────────────────────────


class SmolagentsRunner:
    """TurnRunner adapter wrapping smolagents ToolCallingAgent."""

    def __init__(
        self,
        *,
        agent_holder: list[Any],
        smol_tool_list: list[Any],
        logger: Any,
        model: str,
        tool_mode: str,
        thinking_holder: list[bool],
        rebuild_agent_fn: Callable[[], None],
        apply_thinking_fn: Callable[[], None],
        console: Any,
    ) -> None:
        self._agent_holder = agent_holder
        self._smol_tools = smol_tool_list
        self._logger = logger
        self._model = model
        self._tool_mode = tool_mode
        self._thinking_holder = thinking_holder
        self._rebuild_agent_fn = rebuild_agent_fn
        self._apply_thinking_fn = apply_thinking_fn
        self._console = console
        self._turn = _TurnSnapshot()
        self._reset_next = True

    def run_turn(self, ctx: Any) -> Any:
        from smolagents import LogLevel
        from kubeflow_mcp.agents.runtime.contracts import TurnResult

        console = ctx.extras.get("console", self._console)

        agent = self._agent_holder[0]
        self._turn.begin(agent)
        prev_level = self._logger.level
        self._logger.level = LogLevel.ERROR
        try:
            try:
                answer, step_count = _run_turn_streaming(
                    agent, ctx.user_input, reset=self._reset_next, console=console
                )
            except TypeError:
                answer = _run_turn_blocking(agent, ctx.user_input, reset=self._reset_next)
                step_count = 1
        finally:
            self._logger.level = prev_level

        self._turn.finish(agent, steps=step_count)
        self._reset_next = False
        if answer.strip():
            print_assistant_panel(console, answer)
        _print_turn_stats(console, self._turn)

        return TurnResult(
            text=answer,
            tool_calls=[],
            usage={
                "prompt_tokens": self._turn.last_input,
                "completion_tokens": self._turn.last_output,
            },
        )

    def rebuild(self, *, model: str | None = None, tool_mode: str | None = None) -> None:
        self._rebuild_agent_fn()

    def handle_command(self, line: str) -> bool:
        from kubeflow_mcp.agents.runtime.repl_commands import handle_common_repl_command, CommonReplHandlers

        def _on_tools() -> None:
            print_tools_table(
                self._console,
                [(t.name, getattr(t, "description", "") or "") for t in self._smol_tools],
                header_style="bold magenta",
            )
            print_tip(self._console, f"Mode: {self._tool_mode}  |  Total tools: {len(self._smol_tools)}")

        def _on_think() -> None:
            self._thinking_holder[0] = not self._thinking_holder[0]
            self._apply_thinking_fn()
            state = "on" if self._thinking_holder[0] else "off"
            print_tip(self._console, f"Thinking mode: {state}")

        def _on_export() -> None:
            session = build_session_snapshot(
                model=self._model,
                framework="smolagents",
                tool_mode=self._tool_mode,
                token_input=self._turn.session_input,
                token_output=self._turn.session_output,
            )
            out = export_session_snapshot(session)
            print_tip(self._console, f"Session exported → {out}")

        def _on_import(path: str) -> None:
            try:
                agent = self._agent_holder[0]
                _reset_agent_memory(agent)
                agent.monitor.reset()
                self._turn = _TurnSnapshot()
                self._reset_next = True
                _import_smolagents_tokens(path, self._turn)
                print_tip(self._console, f"Session imported ← {path}  (token totals restored)")
            except Exception as exc:
                print_tip(self._console, f"Import error: {exc}", style="red")

        def _on_clear() -> None:
            agent = self._agent_holder[0]
            _reset_agent_memory(agent)
            agent.monitor.reset()
            self._turn = _TurnSnapshot()
            self._reset_next = True
            print_tip(self._console, "Conversation cleared.")

        def _on_unknown(cmd: str) -> None:
            print_tip(self._console, f"Unknown command: {cmd!r}. Try /tools, /think, /export, /import, /clear.")

        return handle_common_repl_command(line, CommonReplHandlers(
            on_tools=_on_tools, on_think=_on_think, on_export=_on_export,
            on_import=_on_import, on_clear=_on_clear, on_unknown=_on_unknown,
        ))
