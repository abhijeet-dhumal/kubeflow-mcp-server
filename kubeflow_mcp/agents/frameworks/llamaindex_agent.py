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

"""LlamaIndex ReAct agent backend routed through llama-index-llms-litellm."""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable
from typing import Any

from kubeflow_mcp.agents.core.tool_dispatch import normalize_execute_tool_args
from kubeflow_mcp.agents.core.confirm import wrap_with_confirm
from kubeflow_mcp.agents.core.tools import get_system_prompt, load_tools
from kubeflow_mcp.agents.frameworks._thinking import apply_thinking_to_llamaindex
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
    reset_token_totals,
)

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
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


def _wrap_tool_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Confirm gate + gemma-style execute_tool arg normalization."""
    wrapped = wrap_with_confirm(fn)
    if fn.__name__ != "execute_tool":
        return wrapped

    def run(*, tool_name: str | None = None, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        raw_args: dict[str, Any] = dict(kwargs)
        if tool_name is not None:
            raw_args["tool_name"] = tool_name
        if arguments is not None:
            raw_args["arguments"] = arguments
        normalized = normalize_execute_tool_args(raw_args)
        return wrapped(
            tool_name=normalized.get("tool_name"),
            arguments=normalized.get("arguments") or None,
        )

    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    return run


def _estimate_tokens(model: str, text: str) -> int:
    if not text:
        return 0
    try:
        import litellm

        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        return max(1, len(text) // 4)


def _sanitize_react_answer(text: str) -> str:
    """Hide ReAct scaffolding (Thought/Answer) from the assistant panel."""
    if not text:
        return text
    for marker in ("Final Answer:", "Answer:"):
        idx = text.rfind(marker)
        if idx >= 0:
            return text[idx + len(marker) :].strip()
    if text.startswith("Thought:"):
        parts = re.split(r"\n(?:Answer|Final Answer):\s*", text, maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return text.strip()


class _UsageTracker:
    """Per-turn metrics via LlamaIndex TokenCountingHandler + event counts."""

    def __init__(self, model: str, token_handler: Any) -> None:
        self.model = model
        self.token_handler = token_handler
        self.turn_tools = 0
        self.turn_llm_calls = 0
        self.turn_duration = 0.0
        self.turn_estimated = False
        self.turn_input = 0
        self.turn_output = 0
        self.session_input = 0
        self.session_output = 0
        self._turn_start = 0.0
        self._turn_prompt_base = 0
        self._turn_completion_base = 0

    def reset_turn(self) -> None:
        self.turn_tools = 0
        self.turn_llm_calls = 0
        self.turn_duration = 0.0
        self.turn_estimated = False
        self.turn_input = 0
        self.turn_output = 0
        self._turn_start = time.monotonic()
        self.token_handler.reset_counts()
        self._turn_prompt_base = self.token_handler.prompt_llm_token_count or 0
        self._turn_completion_base = self.token_handler.completion_llm_token_count or 0

    def finish_turn(self, streamed_text: str = "") -> None:
        self.turn_duration = time.monotonic() - self._turn_start
        turn_in = (self.token_handler.prompt_llm_token_count or 0) - self._turn_prompt_base
        turn_out = (self.token_handler.completion_llm_token_count or 0) - self._turn_completion_base
        if turn_in <= 0 and turn_out <= 0 and streamed_text:
            turn_in = _estimate_tokens(self.model, streamed_text)
            self.turn_estimated = True
        self.turn_input = max(0, turn_in)
        self.turn_output = max(0, turn_out)
        if self.turn_llm_calls == 0:
            self.turn_llm_calls = 1
        self.session_input += self.turn_input
        self.session_output += self.turn_output


def _print_turn_stats(console, tracker: _UsageTracker, model: str) -> None:
    cost_str = ""
    try:
        import litellm

        cost = litellm.completion_cost(
            model=model,
            prompt_tokens=tracker.turn_input,
            completion_tokens=tracker.turn_output,
        )
        cost_str = f"  cost≈${cost:.4f}"
    except Exception:
        pass

    est = "~" if tracker.turn_estimated else ""
    console.print(
        f"[dim]llm_calls={tracker.turn_llm_calls}  tools={tracker.turn_tools}  "
        f"duration={tracker.turn_duration:.1f}s  "
        f"tokens in={est}{tracker.turn_input:,} out={est}{tracker.turn_output:,}{cost_str}  "
        f"(session in={tracker.session_input:,} out={tracker.session_output:,})[/dim]"
    )


def _llamatrace_status() -> str | None:
    if os.environ.get("LLAMA_CLOUD_API_KEY"):
        project = os.environ.get("LLAMA_CLOUD_PROJECT", "default")
        return f"LlamaCloud → {project}"
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return f"OpenTelemetry → {os.environ['OTEL_EXPORTER_OTLP_ENDPOINT']}"
    return None


def _response_text(stop_event: Any) -> str:
    result = getattr(stop_event, "result", None)
    if result is None:
        return ""
    response = getattr(result, "response", None)
    if response is not None:
        content = getattr(response, "content", None)
        if content:
            return str(content)
    return str(result)


def _tool_output_text(tool_output: Any) -> str:
    raw = getattr(tool_output, "raw_output", None)
    if raw is not None:
        return format_tool_result_display(raw)
    blocks = getattr(tool_output, "blocks", None)
    if blocks:
        return format_tool_result_display(blocks)
    return str(tool_output)


def _import_llamaindex_tokens(path: str, tracker: _UsageTracker) -> None:
    payload = import_session_snapshot(path)
    tokens = payload.get("tokens", {})
    if isinstance(tokens, dict):
        tracker.session_input = int(tokens.get("input", 0) or 0)
        tracker.session_output = int(tokens.get("output", 0) or 0)


async def _run_turn_async(
    agent: Any,
    ctx: Any,
    line: str,
    *,
    console,
    tracker: _UsageTracker,
) -> str:
    from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult

    handler = agent.run(line, ctx=ctx)
    chunks: list[str] = []

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            if event.thinking_delta:
                console.print(f"[dim italic]{event.thinking_delta}[/dim italic]", end="")
            if event.delta:
                chunks.append(event.delta)
        elif isinstance(event, ToolCall):
            tracker.turn_tools += 1
            print_tool_call_panel(console, event.tool_name, event.tool_kwargs or {})
        elif isinstance(event, ToolCallResult):
            print_tool_result_panel(console, _tool_output_text(event.tool_output))

    stop = await handler
    if tracker.turn_tools:
        tracker.turn_llm_calls = tracker.turn_tools + 1
    else:
        tracker.turn_llm_calls = 1
    streamed = _sanitize_react_answer("".join(chunks).strip())
    if streamed:
        return streamed
    return _sanitize_react_answer(_response_text(stop).strip())



def run_llamaindex_chat(
    model: str,
    tool_mode: str = "full",
    base_url: str | None = None,
    thinking: bool = True,
    num_retries: int = 3,
    langfuse: bool = False,
    **_kwargs: Any,
) -> None:
    """Launch LlamaIndex ReActAgent with Kubeflow tools and LiteLLM LLM."""
    try:
        from llama_index.core import Settings
        from llama_index.core.agent.workflow import ReActAgent
        from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
        from llama_index.core.tools import FunctionTool
        from llama_index.core.workflow import Context
        from llama_index.llms.litellm import LiteLLM
    except ImportError as exc:
        msg = "Install optional deps: uv sync --extra agents-llamaindex"
        raise RuntimeError(msg) from exc

    setup_readline_history()
    console = get_console()
    mlflow_logger = MlflowSessionLogger(model=model, tool_mode=tool_mode, framework="llamaindex")

    token_handler = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([token_handler])
    tracker = _UsageTracker(model, token_handler)

    tool_fns, descriptions = load_tools(tool_mode)
    li_tools = [
        FunctionTool.from_defaults(
            fn=_wrap_tool_fn(fn),
            name=fn.__name__,
            description=descriptions.get(fn.__name__, fn.__doc__ or fn.__name__),
        )
        for fn in tool_fns
    ]

    thinking_holder = [thinking]
    pending_rebuild_holder = [False]
    agent_holder: list[Any] = [None]
    ctx_holder: list[Any] = [None]

    def rebuild_agent() -> None:
        llm_kwargs: dict[str, Any] = {"model": model, "num_retries": num_retries}
        if base_url:
            llm_kwargs["api_base"] = base_url
        apply_thinking_to_llamaindex(llm_kwargs, enabled=thinking_holder[0], model=model)
        llm = LiteLLM(**llm_kwargs)
        agent_holder[0] = ReActAgent(
            tools=li_tools,
            llm=llm,
            system_prompt=get_system_prompt(),
            verbose=False,
            timeout=120,
        )
        ctx_holder[0] = Context(agent_holder[0])

    rebuild_agent()

    backend_label = base_url or "cloud / local auto-detect"
    tracing = _llamatrace_status()

    print_welcome_panel(
        panel_title="kubeflow-mcp · LiteLLM · LlamaIndex",
        border_style="bright_green",
        rows=[
            ("white", f"Model   : {model}"),
            ("white", f"Backend : {backend_label}"),
            ("white", f"Mode    : {tool_mode}  ({len(li_tools)} tools)"),
            *([("dim", f"Tracing : {tracing}")] if tracing else []),
            ("dim", ""),
            ("dim", "Commands: /tools  /think  /export  /import <file>  /clear  exit"),
            ("dim", f"Thinking: {'on' if thinking else 'off'} ( /think to toggle )"),
            ("dim", "Native: workflow stream, TokenCountingHandler, thinking_delta"),
            ("dim", "Confirm gate on mutating tools (confirmed=False)"),
            ("dim", "Tracing: LLAMA_CLOUD_API_KEY or OTEL_EXPORTER_OTLP_ENDPOINT"),
            *(
                [("dim", f"MLflow run: {mlflow_logger.run_id[:8]}…")]
                if mlflow_logger.enabled and mlflow_logger.run_id
                else []
            ),
        ],
    )

    runner = LlamaIndexRunner(
        agent_holder=agent_holder,
        ctx_holder=ctx_holder,
        tracker=tracker,
        li_tool_list=li_tools,
        model=model,
        tool_mode=tool_mode,
        thinking_holder=thinking_holder,
        pending_rebuild_holder=pending_rebuild_holder,
        rebuild_agent_fn=rebuild_agent,
        console=console,
    )

    session_id = f"lli-{os.urandom(6).hex()}"
    langfuse_mw = LangfuseMiddleware(session_id=session_id, model=model, framework="llamaindex") if langfuse else None
    session = AgentSession(
        runner=runner,
        middleware=[
            UsageMiddleware(),
            OTelMiddleware(framework="llamaindex"),
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


# ─── LlamaIndexRunner (TurnRunner adapter) ────────────────────────────────────


class LlamaIndexRunner:
    """TurnRunner adapter wrapping LlamaIndex ReActAgent."""

    def __init__(
        self,
        *,
        agent_holder: list[Any],
        ctx_holder: list[Any],
        tracker: _UsageTracker,
        li_tool_list: list[Any],
        model: str,
        tool_mode: str,
        thinking_holder: list[bool],
        pending_rebuild_holder: list[bool],
        rebuild_agent_fn: Callable[[], None],
        console: Any,
    ) -> None:
        self._agent_holder = agent_holder
        self._ctx_holder = ctx_holder
        self._tracker = tracker
        self._li_tools = li_tool_list
        self._model = model
        self._tool_mode = tool_mode
        self._thinking_holder = thinking_holder
        self._pending_rebuild_holder = pending_rebuild_holder
        self._rebuild_agent_fn = rebuild_agent_fn
        self._console = console

    def run_turn(self, ctx: Any) -> Any:
        from kubeflow_mcp.agents.runtime.contracts import TurnResult

        console = ctx.extras.get("console", self._console)

        self._tracker.reset_turn()
        answer = asyncio.run(
            _run_turn_async(
                self._agent_holder[0],
                self._ctx_holder[0],
                ctx.user_input,
                console=console,
                tracker=self._tracker,
            )
        )
        self._tracker.finish_turn(answer)
        if answer.strip():
            print_assistant_panel(console, answer)
        _print_turn_stats(console, self._tracker, self._model)

        return TurnResult(
            text=answer,
            tool_calls=[],
            usage={
                "prompt_tokens": self._tracker.turn_input,
                "completion_tokens": self._tracker.turn_output,
            },
        )

    def rebuild(self, *, model: str | None = None, tool_mode: str | None = None) -> None:
        self._rebuild_agent_fn()

    def handle_command(self, line: str) -> bool:
        from kubeflow_mcp.agents.runtime.repl_commands import handle_common_repl_command, CommonReplHandlers

        def _on_tools() -> None:
            print_tools_table(
                self._console,
                [
                    (
                        getattr(t.metadata, "name", t.__class__.__name__),
                        getattr(t.metadata, "description", "") or "",
                    )
                    for t in self._li_tools
                ],
                header_style="bold green",
            )
            print_tip(self._console, f"Mode: {self._tool_mode}  |  Total tools: {len(self._li_tools)}")

        def _on_think() -> None:
            self._thinking_holder[0] = not self._thinking_holder[0]
            self._pending_rebuild_holder[0] = True
            state = "on" if self._thinking_holder[0] else "off"
            print_tip(self._console, f"Thinking mode: {state} (applies on /clear)")

        def _on_export() -> None:
            session = build_session_snapshot(
                model=self._model,
                framework="llamaindex",
                tool_mode=self._tool_mode,
                token_input=self._tracker.session_input,
                token_output=self._tracker.session_output,
            )
            out = export_session_snapshot(session)
            print_tip(self._console, f"Session exported → {out}")

        def _on_import(path: str) -> None:
            try:
                _import_llamaindex_tokens(path, self._tracker)
                print_tip(self._console, f"Session imported ← {path}  (token totals restored)")
            except Exception as exc:
                print_tip(self._console, f"Import error: {exc}", style="red")

        def _on_clear() -> None:
            if self._pending_rebuild_holder[0]:
                self._rebuild_agent_fn()
                self._pending_rebuild_holder[0] = False
            else:
                from llama_index.core.workflow import Context
                self._ctx_holder[0] = Context(self._agent_holder[0])
            reset_token_totals(self._tracker)
            print_tip(self._console, "Conversation cleared.")

        def _on_unknown(cmd: str) -> None:
            print_tip(self._console, f"Unknown command: {cmd!r}. Try /tools, /think, /export, /import, /clear.")

        return handle_common_repl_command(line, CommonReplHandlers(
            on_tools=_on_tools, on_think=_on_think, on_export=_on_export,
            on_import=_on_import, on_clear=_on_clear, on_unknown=_on_unknown,
        ))
