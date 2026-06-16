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

"""LangChain ReAct agent backend routed through ChatLiteLLM."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any


from kubeflow_mcp.agents.core.tool_dispatch import (
    compact_execute_tool_result,
    normalize_execute_tool_args,
)
from kubeflow_mcp.agents.core.confirm import wrap_with_confirm
from kubeflow_mcp.agents.core.tools import get_system_prompt, load_tools
from kubeflow_mcp.agents.frameworks._observability import is_local_ollama_model, setup_langsmith
from kubeflow_mcp.agents.frameworks._thinking import (
    apply_thinking_to_chat_litellm,
    extract_thinking_delta,
    is_answer_content_token,
)
from kubeflow_mcp.agents.observability import (
    MlflowSessionLogger,
    invoke_with_mlflow_span,
    trace_text,
    trim_preview,
    update_trace_context,
)
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
from kubeflow_mcp.agents.observability.middleware import (
    LangfuseMiddleware,
    MLflowMiddleware,
    OTelMiddleware,
    UsageMiddleware,
)
from kubeflow_mcp.agents.core.confirm import ConfirmMiddleware
from kubeflow_mcp.agents.runtime.session import AgentSession

_SPHINX_BUILD = "sphinx" in sys.modules

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    if not _SPHINX_BUILD:
        sys.exit("Error: install kubeflow-mcp[agents-langchain]")
    BaseCallbackHandler = None  # type: ignore[misc, assignment]
    Panel = None  # type: ignore[misc, assignment]
    Table = None  # type: ignore[misc, assignment]
    Text = None  # type: ignore[misc, assignment]

from kubeflow_mcp.agents.terminal_ui import (  # noqa: E402
    format_tool_result_display,
    get_console,
    print_assistant_panel,
    print_confirm_gate,
    print_error_panel,
    print_tip,
    print_tool_call_panel,
    print_tool_result_panel,
    print_tools_table,
    print_welcome_panel,
    setup_readline_history,
)

_INTERNAL_TOOLS = frozenset({"_Exception"})
_PSEUDO_TOOL_NAMES = frozenset({"final answer", "final_answer", "answer"})
_PREFLIGHT_QUERY_RE = re.compile(r"\bpre[\s_-]?flight\b", re.IGNORECASE)
_MODEL_TAG_RE = re.compile(r"\b(?:hf://)?[A-Za-z0-9._-]+:[A-Za-z0-9._-]+\b")
_MULTI_STEP_QUERY_RE = re.compile(
    r"\b(continue|next(?:\s+step)?|workflow|end[\s-]?to[\s-]?end|e2e|then|and then|after that)\b",
    re.IGNORECASE,
)
_HIGH_RISK_MODEL_HINT_RE = re.compile(
    r"\b(uncensored|jailbreak|nsfw|roleplay|merged|gguf|awq|gptq|mlx|quant)\b",
    re.IGNORECASE,
)
_MULTI_CALL_BLOCK_ERROR = "Multiple tool calls blocked for this turn"

_TURN_POLICY: dict[str, Any] = {"allow_multi_step": False, "tool_calls": 0}

_PARSING_HINT = (
    "Format error — respond with EXACTLY one of these:\n"
    "Thought: I now know the final answer\n"
    "Final Answer: <your answer>\n"
    "OR\n"
    "Thought: <reason>\n"
    "Action: <tool_name>\n"
    "Action Input: <json>"
)

_REACT_BODY = """You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action as valid JSON (use {{}} for no arguments)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT:
- If the user is only greeting or asking what you can do, respond with Final Answer immediately.
  Do NOT call tools for simple greetings.
- After a tool Observation, summarize the result for the user — do NOT repeat the full JSON.
  Use: Thought: I now know the final answer / Final Answer: <summary>
- Every response MUST include either "Final Answer:" or "Action:" on its own line.
- Action Input MUST be valid JSON on its own line (e.g. {{}} or {{"tool_name": "list_runtimes"}}).
- For execute_tool use {{"tool_name": "<name>", "arguments": {{...}}}} for tools with parameters.
- If user gives an Ollama chat tag (e.g. gemma4:e4b), explain pre_flight needs a HuggingFace model ID
  (e.g. google/gemma-2-2b or hf://google/gemma-2-2b).
- For one user request, run exactly one tool call unless the user explicitly asks to continue with more steps.
- Keep Final Answer concise (max 4 short lines) unless user asks for full detail.
- Do not ask the user to pick numbered workflow steps unless they asked for options.
- Never use Action: None.

Begin!

Previous conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""


def _react_prompt_template(system_prompt: str):
    from langchain_core.prompts import PromptTemplate

    escaped = system_prompt.replace("{", "{{").replace("}", "}}")
    return PromptTemplate.from_template(f"{escaped}\n\n{_REACT_BODY}")


def _normalize_react_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Fix common small-model mistakes (especially execute_tool arg shapes)."""
    if tool_name != "execute_tool":
        return args
    normalized = normalize_execute_tool_args(args)
    if (
        isinstance(normalized.get("tool_name"), str)
        and normalized.get("tool_name")
        and not isinstance(normalized.get("arguments"), dict)
    ):
        normalized["arguments"] = {}
    if isinstance(normalized.get("tool_name"), str):
        return normalized
    parsed = _parse_function_like_tool_input(args.get("tool_input"))
    if parsed:
        merged = dict(normalized)
        merged.update(parsed)
        return merged
    return normalized


def _parse_function_like_tool_input(raw: Any) -> dict[str, Any] | None:
    """Parse strings like ``pre_flight(model='x')`` into execute_tool payload."""
    if not isinstance(raw, str) or "(" not in raw or ")" not in raw:
        return None
    text = raw.strip()
    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    call = node.body
    if not isinstance(call, ast.Call):
        return None
    if not isinstance(call.func, ast.Name) or not call.func.id:
        return None
    kwargs: dict[str, Any] = {}
    for item in call.keywords:
        if item.arg is None:
            return None
        try:
            kwargs[item.arg] = ast.literal_eval(item.value)
        except Exception:
            return None
    if call.args:
        try:
            kwargs.setdefault("input", ast.literal_eval(call.args[0]))
        except Exception:
            return None
    payload: dict[str, Any] = {"tool_name": call.func.id}
    if kwargs:
        payload["arguments"] = kwargs
    return payload


def _validate_execute_tool_payload(args: dict[str, Any]) -> str | None:
    tool_name = args.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return (
            "Invalid execute_tool payload: missing tool_name. "
            'Use {"tool_name": "<name>", "arguments": {...}}.'
        )
    arguments = args.get("arguments")
    if arguments is None:
        args["arguments"] = {}
        arguments = args["arguments"]
    if not isinstance(arguments, dict):
        return (
            "Invalid execute_tool payload: arguments must be an object. "
            'Use {"tool_name": "<name>", "arguments": {...}}.'
        )
    requested = tool_name.strip()
    if requested == "pre_flight" and not isinstance(arguments.get("model"), str):
        return (
            "Invalid execute_tool payload for pre_flight: model is required. "
            'Use {"tool_name": "pre_flight", "arguments": {"model": "google/gemma-2-2b"}}.'
        )
    return None


def _compact_react_result(tool_name: str, result: Any) -> Any:
    """Compact high-volume tool output before feeding it back to ReAct."""
    return compact_execute_tool_result(tool_name, result)


def _parse_react_tool_input(tool_input: str) -> dict[str, Any]:
    """Parse ReAct Action Input string into kwargs for Kubeflow tools."""
    raw = (tool_input or "").strip()
    if not raw or raw.lower() in ("none", "null", "{}"):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"input": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"input": parsed}


def _is_invalid_preflight_model(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.removeprefix("hf://")
    return "/" not in candidate or ":" in candidate


def _score_hf_model_id(model_id: str, *, target_query: str) -> float:
    lower = model_id.lower()
    score = 0.0
    if "/" not in model_id:
        score += 50.0
    org = lower.split("/", 1)[0]
    if target_query.startswith("gemma") and org == "google":
        score -= 25.0
    if org in {"google", "meta-llama", "mistralai", "qwen", "microsoft"}:
        score -= 6.0
    if lower.endswith(("-it", "-instruct", "-chat")):
        score -= 3.0
    if _HIGH_RISK_MODEL_HINT_RE.search(lower):
        score += 30.0
    for match in re.finditer(r"(\d+)\s*[bB]\b", lower):
        size_b = int(match.group(1))
        if size_b >= 30:
            score += 20.0
        elif size_b >= 20:
            score += 12.0
        elif size_b >= 10:
            score += 6.0
        elif size_b <= 4:
            score -= 2.0
    return score


def _suggest_hf_model_ids(raw: Any, *, limit: int = 3) -> list[str]:
    if not isinstance(raw, str):
        return []
    query = raw.removeprefix("hf://").strip()
    if ":" in query:
        query = query.split(":", 1)[0]
    query = re.sub(r"\d+$", "", query).strip("-_./")
    if not query:
        return []
    try:
        from huggingface_hub import list_models

        matches = list_models(search=query, limit=limit, full=False)
    except Exception:
        return []
    scored: list[tuple[float, str]] = []
    target_query = query.lower()

    deduped: set[str] = set()
    for item in matches:
        model_id = getattr(item, "id", None)
        if not isinstance(model_id, str) or "/" not in model_id or model_id in deduped:
            continue
        deduped.add(model_id)
        scored.append((_score_hf_model_id(model_id, target_query=target_query), model_id))
    scored.sort(key=lambda pair: (pair[0], pair[1].lower()))
    return [model_id for _, model_id in scored[:limit]]


def _allow_multi_step_for_query(query: str) -> bool:
    return bool(_MULTI_STEP_QUERY_RE.search(query or ""))


def _is_multi_call_blocked_observation(observation: str | None) -> bool:
    return bool(observation and _MULTI_CALL_BLOCK_ERROR in observation)


def _multi_call_blocked_reply() -> str:
    return (
        "Done with this step. I stopped before auto-running another tool. "
        "If you want me to continue, say `continue` and name the next action "
        "(for example: `continue with get_runtime openmpi-cuda`)."
    )




def _extract_text_from_content_blocks(content: Any) -> str:
    """Extract plain text from vLLM/Anthropic-style content block lists.

    vLLM with Qwen3 thinking returns content as a list like:
      ['', {'type': 'thinking', 'thinking': '...'}, "actual answer"]
    Strip thinking blocks and join only text items.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            # skip type="thinking" blocks
        return "".join(parts)
    return str(content)


def _sanitize_react_output(text: Any) -> str:
    if not isinstance(text, str):
        text = _extract_text_from_content_blocks(text)
    if not text:
        return ""
    marker_idx = text.rfind("Final Answer:")
    if marker_idx >= 0:
        return text[marker_idx + len("Final Answer:") :].strip()
    cleaned_lines = [
        line
        for line in text.splitlines()
        if not line.startswith(("Thought:", "Action:", "Action Input:", "Observation:"))
    ]
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text.strip()


def _preflight_input_guard_reply(line: str) -> str | None:
    if not _PREFLIGHT_QUERY_RE.search(line):
        return None
    match = _MODEL_TAG_RE.search(line)
    if not match:
        return None
    model_id = match.group(0)
    if not _is_invalid_preflight_model(model_id):
        return None
    suggestions = _suggest_hf_model_ids(model_id)
    suggestion_line = (
        f" Suggested IDs: {', '.join(suggestions)}." if suggestions else ""
    )
    return (
        "pre_flight requires a HuggingFace model ID like `google/gemma-2-2b` "
        f"(not `{model_id}`).{suggestion_line}"
    )


def _make_react_tool(fn: Callable[..., Any], description: str):
    """Wrap a Kubeflow tool for LangChain ReAct (single string Action Input)."""
    from langchain_classic.tools import Tool

    desc = description or (fn.__doc__ or fn.__name__).split("\n")[0]

    def run(tool_input: str = "") -> str:
        args = _normalize_react_args(fn.__name__, _parse_react_tool_input(tool_input))
        if fn.__name__ == "execute_tool":
            payload_error = _validate_execute_tool_payload(args)
            if payload_error:
                payload = {
                    "error": payload_error,
                    "tool": fn.__name__,
                    "args": args,
                }
                return json.dumps(payload, indent=2, default=str)
            if not _TURN_POLICY["allow_multi_step"] and _TURN_POLICY["tool_calls"] >= 1:
                payload = {
                    "error": "Multiple tool calls blocked for this turn",
                    "tool": fn.__name__,
                    "args": args,
                    "hint": (
                        "Do not auto-advance workflow steps. Ask user for confirmation "
                        "before running the next tool."
                    ),
                }
                return json.dumps(payload, indent=2, default=str)
            requested = str(args.get("tool_name") or "").strip()
            if requested.lower() in _PSEUDO_TOOL_NAMES:
                payload = {
                    "error": "Invalid tool_name: pseudo final-answer marker is not a real tool",
                    "tool": fn.__name__,
                    "args": args,
                    "hint": "Respond with Final Answer directly after observations.",
                }
                return json.dumps(payload, indent=2, default=str)
            call_args = args.get("arguments") or {}
            if requested == "pre_flight" and isinstance(call_args, dict):
                model_id = call_args.get("model")
                if _is_invalid_preflight_model(model_id):
                    suggestions = _suggest_hf_model_ids(model_id)
                    suggestion_text = (
                        f" Try one of: {', '.join(suggestions)}." if suggestions else ""
                    )
                    payload = {
                        "error": "Invalid HuggingFace model ID for pre_flight",
                        "tool": fn.__name__,
                        "args": args,
                        "hint": (
                            "Use model like google/gemma-2-2b or hf://google/gemma-2-2b "
                            f"(not gemma4:e4b).{suggestion_text}"
                        ),
                    }
                    return json.dumps(payload, indent=2, default=str)
        try:
            _rc = get_console()
            with _rc.status(f"[dim]running {fn.__name__}…[/dim]", spinner="dots"):
                result = invoke_with_mlflow_span(fn, args, framework="langchain")
            if fn.__name__ == "execute_tool":
                _TURN_POLICY["tool_calls"] += 1
        except Exception as exc:
            payload = {"error": str(exc), "tool": fn.__name__, "args": args}
            return json.dumps(payload, indent=2, default=str)
        result = _compact_react_result(fn.__name__, result)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    return Tool(name=fn.__name__, func=run, description=desc)


def _vllm_safe_formatter(
    intermediate_steps: Any,
) -> list[Any]:
    """Wrap format_to_tool_messages to sanitize AIMessages for vLLM.

    langchain_litellm injects reasoning tokens into AIMessage content as typed
    blocks: [{"type": "thinking", "thinking": "..."}, {"type": "text", ...}].
    vLLM rejects "thinking" as an invalid content part type on the second LLM
    call. This formatter strips all non-text blocks and collapses content to a
    plain string before the scratchpad reaches vLLM.
    """
    from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
    from langchain_core.messages import AIMessage

    messages = format_to_tool_messages(intermediate_steps)
    result = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            content = msg.content
            if not content:
                msg = msg.model_copy(update={"content": " "})
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip() or " "
                msg = msg.model_copy(update={"content": text})
        result.append(msg)
    return result


def _build_executor(
    *,
    model: str,
    llm: Any,
    tool_fns: list[Callable[..., Any]],
    descriptions: dict[str, str],
    system_prompt: str,
    run_config: dict[str, Any],
    memory: Any,
    use_vllm_safe_formatter: bool = False,
) -> tuple[Any, list[Any], str]:
    """ReAct for local Ollama; native tool-calling for cloud models."""
    from langchain_classic.agents import (
        AgentExecutor,
        create_react_agent,
        create_tool_calling_agent,
    )
    from langchain_classic.tools import StructuredTool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    wrapped = [wrap_with_confirm(fn) for fn in tool_fns]
    use_react = is_local_ollama_model(model)
    agent_mode = "ReAct" if use_react else "ToolCalling"

    if use_react:
        lc_tools = [_make_react_tool(fn, descriptions.get(fn.__name__, "")) for fn in wrapped]
        prompt = _react_prompt_template(system_prompt)
        agent = create_react_agent(llm, lc_tools, prompt)
        extra = {"handle_parsing_errors": _PARSING_HINT}
    else:
        def _make_tool_calling_tool(fn: Callable[..., Any], desc: str) -> Any:
            """StructuredTool wrapper that adds OTel + MLflow spans (mirrors _make_react_tool)."""
            import inspect as _inspect
            _has_confirmed = "confirmed" in _inspect.signature(fn).parameters

            def run(**kwargs: Any) -> str:
                _rc = get_console()
                # Mutating tools (confirmed param) may show a Rich Confirm.ask() dialog.
                # Running that inside console.status() deadlocks stdin — skip the spinner.
                if _has_confirmed:
                    result = invoke_with_mlflow_span(fn, kwargs, framework="langchain")
                else:
                    with _rc.status(f"[dim]running {fn.__name__}…[/dim]", spinner="dots"):
                        result = invoke_with_mlflow_span(fn, kwargs, framework="langchain")
                result = _compact_react_result(fn.__name__, result)
                if isinstance(result, (dict, list)):
                    return json.dumps(result, indent=2, default=str)
                return str(result)

            run.__name__ = fn.__name__
            # Unwrap confirm/audit layers to expose the original tool signature to
            # StructuredTool — otherwise the schema sees (**kwargs) and the model
            # wraps all arguments under a "kwargs" key causing TypeError.
            import functools
            original = fn
            while hasattr(original, "__wrapped__"):
                original = original.__wrapped__
            functools.update_wrapper(run, original)
            run.__name__ = fn.__name__  # keep the registered tool name
            return StructuredTool.from_function(
                func=run,
                name=fn.__name__,
                description=desc,
            )

        lc_tools = [
            _make_tool_calling_tool(fn, descriptions.get(fn.__name__, fn.__doc__ or fn.__name__))
            for fn in wrapped
        ]
        tool_calling_system = (
            system_prompt
            + "\n\nFor greetings, small talk, or general questions that do not require "
            "live Kubernetes or training data, reply directly WITHOUT calling any tool."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", tool_calling_system),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages

        agent = create_tool_calling_agent(
            llm,
            lc_tools,
            prompt,
            message_formatter=_vllm_safe_formatter if use_vllm_safe_formatter else format_to_tool_messages,
        )
        extra = {}

    executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        verbose=False,
        max_iterations=15,
        max_execution_time=None,  # no wall-clock cap — K8s tool timeout (45s) prevents hangs
        memory=memory,
        return_intermediate_steps=False,
        tags=run_config["tags"],
        metadata=run_config["metadata"],
        **extra,
    )
    return executor, lc_tools, agent_mode


def _estimate_tokens(model: str, text: str) -> int:
    """Estimate tokens when the provider omits usage (common with Ollama)."""
    if not text:
        return 0
    try:
        import litellm

        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        return max(1, len(text) // 4)


class _UsageTracker:
    """Accumulate per-turn and session LLM/tool metrics via LangChain callbacks."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.turn_input = 0
        self.turn_output = 0
        self.turn_tools = 0
        self.turn_tool_names: list[str] = []
        self.turn_llm_calls = 0
        self.turn_duration = 0.0
        self.turn_estimated = False
        self.session_input = 0
        self.session_output = 0
        self._turn_start = 0.0
        self._handler = None
        self._last_prompts: list[str] = []

    def reset_turn(self) -> None:
        self.turn_input = 0
        self.turn_output = 0
        self.turn_tools = 0
        self.turn_tool_names = []
        self.turn_llm_calls = 0
        self.turn_duration = 0.0
        self.turn_estimated = False
        self._last_prompts = []
        self._turn_start = time.monotonic()

    def finish_turn(self) -> None:
        self.turn_duration = time.monotonic() - self._turn_start

    @property
    def handler(self):  # noqa: C901
        if self._handler is None:
            from langchain_core.callbacks.base import BaseCallbackHandler

            tracker = self

            class _Handler(BaseCallbackHandler):
                def on_llm_start(self, serialized, prompts, **kwargs) -> None:
                    tracker._last_prompts = [str(p) for p in prompts]
                    prompt_preview = trim_preview(trace_text("\n".join(tracker._last_prompts)))
                    update_trace_context(framework="langchain", request_preview=prompt_preview)

                def on_llm_end(self, response, **kwargs) -> None:
                    tracker.turn_llm_calls += 1
                    usage: dict[str, Any] = {}
                    if response.llm_output:
                        usage = response.llm_output.get("token_usage") or {}
                    if not usage and response.generations:
                        gen = response.generations[0][0]
                        info = getattr(gen, "generation_info", None) or {}
                        usage = info.get("token_usage") or info
                    inp = int(
                        usage.get("prompt_tokens")
                        or usage.get("input_tokens")
                        or 0
                    )
                    out = int(
                        usage.get("completion_tokens")
                        or usage.get("output_tokens")
                        or 0
                    )
                    if inp == 0 and out == 0:
                        prompt_text = "\n".join(tracker._last_prompts)
                        gen_text = ""
                        for gen_list in response.generations:
                            for gen in gen_list:
                                gen_text += getattr(gen, "text", "") or ""
                        if prompt_text or gen_text:
                            inp = _estimate_tokens(tracker.model, prompt_text)
                            out = _estimate_tokens(tracker.model, gen_text)
                            tracker.turn_estimated = True
                    tracker.turn_input += inp
                    tracker.turn_output += out
                    tracker.session_input += inp
                    tracker.session_output += out
                    if response.generations:
                        text_parts: list[str] = []
                        for gen_list in response.generations:
                            for gen in gen_list:
                                text = getattr(gen, "text", None)
                                if isinstance(text, str) and text:
                                    text_parts.append(text)
                        if text_parts:
                            reply_preview = trim_preview(trace_text("".join(text_parts)))
                            update_trace_context(framework="langchain", response_preview=reply_preview)

                def on_tool_start(self, serialized, input_str, **kwargs) -> None:
                    name = ""
                    if isinstance(serialized, dict):
                        name = str(serialized.get("name") or "")
                    if name in _INTERNAL_TOOLS:
                        return
                    tracker.turn_tools += 1
                    if name:
                        tracker.turn_tool_names.append(name)

            self._handler = _Handler()
        return self._handler




_THINKING_LINE_CAP = 8   # max lines of thinking shown per LLM call


def _make_thinking_display_handler(console, enabled_holder: list[bool]):
    """LangChain callback: stream reasoning_content to the terminal."""
    class _ThinkingHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            self._active = False
            self._lines = 0      # newlines rendered so far this call
            self._capped = False # True once the line cap is hit

        def reset(self) -> None:
            if self._active:
                console.print()
            self._active = False
            self._lines = 0
            self._capped = False

        def on_llm_new_token(self, token, *, chunk=None, **kwargs) -> None:
            if not enabled_holder[0]:
                return
            delta = extract_thinking_delta(token, chunk)
            if delta:
                if not self._active:
                    self._active = True
                    self._lines = 0
                    self._capped = False
                    console.print()
                    console.print("[dim italic]💭 ", end="")
                if self._capped:
                    return
                # Count newlines to enforce the line cap
                newlines_in_delta = delta.count("\n")
                if self._lines + newlines_in_delta >= _THINKING_LINE_CAP:
                    # Print up to the cap then truncate
                    remaining = _THINKING_LINE_CAP - self._lines
                    truncated = "\n".join(delta.split("\n")[:remaining])
                    if truncated:
                        console.print(f"[dim italic]{truncated}[/dim italic]", end="")
                    console.print()
                    console.print("[dim]… (thinking truncated)[/dim]")
                    self._capped = True
                    return
                self._lines += newlines_in_delta
                console.print(f"[dim italic]{delta}[/dim italic]", end="")
            elif self._active and is_answer_content_token(token, chunk):
                console.print()
                self._active = False

        def on_llm_end(self, *args, **kwargs) -> None:
            self.reset()

        def on_llm_error(self, *args, **kwargs) -> None:
            self.reset()

    return _ThinkingHandler()


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


def _is_internal_observation(tool: str | None, observation: str) -> bool:
    if tool in _INTERNAL_TOOLS:
        return True
    obs = (observation or "").strip()
    return obs.startswith("Invalid Format:") or obs.startswith("Invalid or incomplete")


def _is_preview_result(observation: str) -> bool:
    """Return True when a tool returned a confirmed=False preview (not yet submitted)."""
    try:
        data = json.loads(observation)
        inner = data.get("data", {}) if isinstance(data, dict) else {}
        return bool(inner.get("preview") or inner.get("confirmed") is False)
    except (json.JSONDecodeError, TypeError):
        return False


def _render_stream_chunk(console, chunk: dict[str, Any]) -> str | None:
    """Render tool call / observation panels; return last real tool observation."""
    last_obs: str | None = None
    for action in chunk.get("actions") or []:
        tool = getattr(action, "tool", None)
        if tool and tool not in _INTERNAL_TOOLS:
            raw_input = getattr(action, "tool_input", "") or ""
            # ToolCalling agents pass tool_input as a dict; ReAct passes a JSON string.
            args = raw_input if isinstance(raw_input, dict) else _normalize_react_args(
                tool, _parse_react_tool_input(raw_input)
            )
            print_tool_call_panel(console, tool, args)

    for step in chunk.get("steps") or []:
        action = getattr(step, "action", None)
        tool = getattr(action, "tool", None) if action else None
        obs = getattr(step, "observation", None)
        if obs and not _is_internal_observation(tool, obs):
            if _is_preview_result(obs):
                print_confirm_gate(console, obs)
            else:
                print_tool_result_panel(console, obs)
            last_obs = obs

    for action, observation in chunk.get("intermediate_step") or []:
        tool = getattr(action, "tool", None)
        if tool and tool not in _INTERNAL_TOOLS:
            args = _normalize_react_args(
                tool, _parse_react_tool_input(getattr(action, "tool_input", "") or "")
            )
            print_tool_call_panel(console, tool, args)
        if observation and not _is_internal_observation(tool, observation):
            print_tool_result_panel(console, observation)
            last_obs = observation
    return last_obs


def _fallback_from_observation(observation: str, user_query: str) -> str:
    """Build a reply when the model fails to produce Final Answer after a tool call."""
    if '"error"' in observation or "tool_call_failed" in observation.lower():
        return (
            f"Tool call did not complete successfully for “{user_query}”. "
            "See the Result panel above for details."
        )
    try:
        data = json.loads(observation)
    except json.JSONDecodeError:
        return observation[:2000]

    if isinstance(data, dict) and data.get("success") and "data" in data:
        inner = data["data"]
        if isinstance(inner, dict) and "runtimes" in inner:
            names = [r.get("name", r) if isinstance(r, dict) else r for r in inner["runtimes"]]
            preview = ", ".join(str(n) for n in names[:8])
            suffix = f" (+{len(names) - 8} more)" if len(names) > 8 else ""
            return f"For “{user_query}”: found {len(names)} runtimes — {preview}{suffix}."
        return json.dumps(inner, indent=2, default=str)[:2000]

    return json.dumps(data, indent=2, default=str)[:2000]



def _run_turn_streaming(
    executor,
    line: str,
    run_config: dict[str, Any],
    console,
    thinking_handler: Any | None = None,
    is_react: bool = True,
) -> str:
    """Stream agent steps via sync .stream(); thinking display handled by registered callback."""
    if thinking_handler is not None:
        thinking_handler.reset()
    _TURN_POLICY["allow_multi_step"] = _allow_multi_step_for_query(line)
    _TURN_POLICY["tool_calls"] = 0
    output = ""
    last_obs: str | None = None
    parse_retries = 0

    # ToolCalling (cloud) emits a JSON function call with no streamed text, so
    # there is no visible feedback during LLM inference. Show a spinner that
    # must stop before thinking tokens print — Rich Live suppresses other console
    # output while the spinner is active. A _SpinnerStopper callback stops it on
    # the very first on_llm_new_token, before the thinking handler renders anything.
    # ReAct (Ollama) streams its chain-of-thought directly — no spinner needed.
    _status_holder: list[Any] = [None]

    class _SpinnerStopper(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            s = _status_holder[0]
            if s is not None:
                s.stop()
                _status_holder[0] = None

    if not is_react:
        _status_holder[0] = console.status("[dim]Thinking…[/dim]", spinner="dots")
        _status_holder[0].start()

    turn_config = {
        **run_config,
        "callbacks": [*run_config.get("callbacks", []), _SpinnerStopper()],
    }

    for chunk in executor.stream({"input": line}, config=turn_config):
        if _status_holder[0] is not None:
            _status_holder[0].stop()
            _status_holder[0] = None
        if not isinstance(chunk, dict):
            continue
        obs = _render_stream_chunk(console, chunk)
        if obs:
            last_obs = obs
            if _is_multi_call_blocked_observation(obs):
                output = _multi_call_blocked_reply()
                break
        for action, _observation in chunk.get("intermediate_step") or []:
            if getattr(action, "tool", None) in _INTERNAL_TOOLS:
                parse_retries += 1
        if chunk.get("output"):
            output = _sanitize_react_output(_extract_text_from_content_blocks(chunk["output"]))
    if _status_holder[0] is not None:  # stopped before any chunk (e.g. empty stream)
        _status_holder[0].stop()
    if not output.strip() and last_obs:
        output = _fallback_from_observation(last_obs, line)
        if parse_retries:
            print_tip(
                console,
                f"Model failed ReAct formatting after tool call ({parse_retries} retries). "
                "Showing tool result summary — try qwen3:8b for better tool use.",
                style="dim",
            )
    return _sanitize_react_output(output)


def _run_turn_blocking(executor, line: str, run_config: dict[str, Any]) -> str:
    _TURN_POLICY["allow_multi_step"] = _allow_multi_step_for_query(line)
    _TURN_POLICY["tool_calls"] = 0
    result = executor.invoke({"input": line}, config=run_config)
    raw = result.get("output", "") if isinstance(result, dict) else result
    output = _extract_text_from_content_blocks(raw)
    if _is_multi_call_blocked_observation(output):
        return _multi_call_blocked_reply()
    return _sanitize_react_output(output)


def _build_run_config(
    tracker: _UsageTracker,
    model: str,
    tool_mode: str,
    thinking_handler: Any | None = None,
) -> dict[str, Any]:
    callbacks: list[Any] = [tracker.handler]
    if thinking_handler is not None:
        callbacks.append(thinking_handler)
    return {
        "callbacks": callbacks,
        "tags": ["kubeflow-mcp", "framework:langchain", f"mode:{tool_mode}"],
        "metadata": {"model": model, "tool_mode": tool_mode, "framework": "langchain"},
        "run_name": "kubeflow-mcp-langchain",
    }


def _restore_langchain_memory(memory, payload: dict[str, Any]) -> int:
    """Restore user/assistant turns from known session payload shapes."""
    candidates: list[Any] = []
    chat_messages = payload.get("chat_messages")
    if isinstance(chat_messages, list):
        candidates = chat_messages
    elif isinstance(payload.get("messages"), list):
        candidates = payload["messages"]
    elif isinstance(payload.get("chat_history"), dict):
        nested = payload["chat_history"].get("chat_messages")
        if isinstance(nested, list):
            candidates = nested

    restored = 0
    for item in candidates:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            memory.chat_memory.add_user_message(content)
            restored += 1
        elif role == "assistant":
            memory.chat_memory.add_ai_message(content)
            restored += 1
    return restored


def _build_langchain_export_payload(memory, *, model: str, tool_mode: str, tracker: _UsageTracker) -> dict[str, Any]:
    chat_messages = []
    for item in getattr(memory.chat_memory, "messages", []):
        msg_type = getattr(item, "type", "")
        if msg_type in ("human", "ai"):
            chat_messages.append(
                {"role": "user" if msg_type == "human" else "assistant", "content": str(item.content)}
            )
    return build_session_snapshot(
        model=model,
        framework="langchain",
        tool_mode=tool_mode,
        token_input=tracker.session_input,
        token_output=tracker.session_output,
        extra={
            "chat_history": memory.load_memory_variables({}),
            "chat_messages": chat_messages,
        },
    )


def _import_langchain_session(memory, tracker: _UsageTracker, path: str) -> int:
    payload = import_session_snapshot(path)
    memory.clear()
    restored = _restore_langchain_memory(memory, payload)
    tokens = payload.get("tokens", {})
    reset_token_totals(tracker)
    if isinstance(tokens, dict):
        tracker.session_input = int(tokens.get("input", 0) or 0)
        tracker.session_output = int(tokens.get("output", 0) or 0)
    return restored



def run_langchain_chat(
    model: str,
    tool_mode: str = "full",
    base_url: str | None = None,
    langfuse: bool = False,
    thinking: bool = True,
    num_retries: int = 3,
    **_kwargs: Any,
) -> None:
    """Launch LangChain agent — ReAct (Ollama) or native tool-calling (cloud)."""
    try:
        from langchain_classic.memory import ConversationBufferMemory
        from langchain_litellm import ChatLiteLLM
    except ImportError as exc:
        msg = "Install optional deps: uv sync --extra agents-langchain"
        raise RuntimeError(msg) from exc

    setup_readline_history()
    if base_url:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Disable LiteLLM's built-in observability auto-callbacks — we manage our
    # own via MlflowSessionLogger and LangfuseMiddleware. Without this, LiteLLM
    # detects MLFLOW_TRACKING_URI / LANGFUSE_* env vars and adds its own callbacks,
    # which causes AttributeError on langfuse 3+/4+ (expects langfuse.version) and
    # spams the console with connection-refused / auth warnings.
    try:
        import litellm as _litellm
        _litellm.success_callback = [
            cb for cb in _litellm.success_callback
            if not any(s in str(cb).lower() for s in ("mlflow", "langfuse"))
        ]
        _litellm.failure_callback = [
            cb for cb in _litellm.failure_callback
            if not any(s in str(cb).lower() for s in ("mlflow", "langfuse"))
        ]
    except Exception:
        pass

    console = get_console()
    os.environ["KUBEFLOW_MCP_MODEL"] = model
    tracker = _UsageTracker(model)
    mlflow_turn_logger = MlflowSessionLogger(model=model, tool_mode=tool_mode, framework="langchain")
    thinking_holder = [thinking]
    thinking_handler = _make_thinking_display_handler(console, thinking_holder)
    run_config = _build_run_config(tracker, model, tool_mode, thinking_handler)

    # Mutable holder so /mode can swap tool sets without rebuilding the closure.
    tool_holder: dict[str, Any] = {}
    _initial_fns, _initial_descs = load_tools(tool_mode)
    tool_holder.update({"fns": _initial_fns, "descs": _initial_descs, "mode": tool_mode})

    # Compact tier for local Ollama — fewer tokens → faster inference.
    tier = "compact" if is_local_ollama_model(model) else "full"
    system_prompt = get_system_prompt(instruction_tier=tier)
    executor_holder: dict[str, Any] = {}

    # vLLM endpoints (RHOAI, local) return AIMessages with content='' when
    # thinking is active. The safe formatter replaces empty content with a
    # single space before the scratchpad is sent back to vLLM, which allows
    # thinking to be enabled without triggering BadRequestError.
    is_vllm = bool(base_url)

    def rebuild() -> None:
        llm_kwargs: dict[str, Any] = {"model": model, "streaming": True, "num_retries": num_retries}
        if base_url:
            llm_kwargs["api_base"] = base_url
            # vLLM Qwen3: enable_thinking must be sent via extra_body. The empty-content
            # scratchpad issue is handled by _vllm_safe_formatter in _build_executor.
            llm_kwargs.setdefault("model_kwargs", {})["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking_holder[0]}
            }
        apply_thinking_to_chat_litellm(llm_kwargs, enabled=thinking_holder[0], model=model)
        llm = ChatLiteLLM(**llm_kwargs)
        executor, lc_tools, agent_mode = _build_executor(
            model=model,
            llm=llm,
            tool_fns=tool_holder["fns"],
            descriptions=tool_holder["descs"],
            system_prompt=system_prompt,
            run_config=run_config,
            use_vllm_safe_formatter=is_vllm,
            memory=memory,
        )
        executor_holder["executor"] = executor
        executor_holder["lc_tools"] = lc_tools
        executor_holder["agent_mode"] = agent_mode

    def change_mode(new_mode: str) -> None:
        new_fns, new_descs = load_tools(new_mode)
        tool_holder.update({"fns": new_fns, "descs": new_descs, "mode": new_mode})
        rebuild()

    class _NormalizedMemory(ConversationBufferMemory):
        """ConversationBufferMemory with two native LangChain enhancements:

        1. ``save_context`` normalizes AIMessage content blocks to plain text.
           Models like Qwen3 on vLLM return content as a typed-block list
           ``[{'type': 'thinking', ...}, {'type': 'text', 'text': '...'}]``.
           Storing raw causes pydantic errors on the next turn.

        2. ``load_memory_variables`` applies ``trim_messages`` to cap history
           within the model's context window before each call.
        """

        def save_context(self, inputs: dict, outputs: dict) -> None:
            normalized = {
                k: _extract_text_from_content_blocks(v) if not isinstance(v, str) else v
                for k, v in outputs.items()
            }
            super().save_context(inputs, normalized)

        def load_memory_variables(self, inputs: dict) -> dict:
            variables = super().load_memory_variables(inputs)
            history = variables.get(self.memory_key)
            if isinstance(history, list) and len(history) > 4:
                try:
                    from langchain_core.messages import trim_messages
                    variables[self.memory_key] = trim_messages(
                        history,
                        max_tokens=6000,
                        strategy="last",
                        token_counter=lambda msgs: sum(
                            len(str(getattr(m, "content", ""))) // 3 for m in msgs
                        ),
                        start_on="human",
                        include_system=False,
                        allow_partial=False,
                    )
                except Exception:
                    pass  # never block on trim failure — history is better than nothing
            return variables

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        memory = _NormalizedMemory(
            memory_key="chat_history",
            output_key="output",
            return_messages=True,
        )

    rebuild()
    agent_mode = executor_holder["agent_mode"]
    lc_tools = executor_holder["lc_tools"]

    backend_label = base_url or "cloud / local auto-detect"
    tracing = setup_langsmith(langsmith=False)  # LangSmith only when LANGCHAIN_TRACING_V2 is set externally

    print_welcome_panel(
        panel_title="kubeflow-mcp · LiteLLM · LangChain",
        border_style="bright_cyan",
        rows=[
            ("white", f"Model   : {model}"),
            ("white", f"Agent   : {agent_mode}"),
            ("white", f"Backend : {backend_label}"),
            ("white", f"Mode    : {tool_mode}  ({len(lc_tools)} tools)"),
            *([("dim", f"Tracing : {tracing}")] if tracing else []),
            ("dim", ""),
            ("dim", "Commands: /help  /tools  /mode <mode>  /think  /export  /import <file>  /clear  exit"),
            ("dim", f"Thinking: {'on' if thinking else 'off'} ( /think to toggle )"),
            ("dim", "Confirm gate on mutating tools (confirmed=False)"),
            *([("dim", "Langfuse: enabled")] if langfuse else []),
            *(
                [("dim", f"MLflow run: {mlflow_turn_logger.run_id[:8]}…")]
                if mlflow_turn_logger.enabled and mlflow_turn_logger.run_id
                else []
            ),
            *(
                [("dim", f"MLflow trace mode: {mlflow_turn_logger._trace_mode}")]
                if mlflow_turn_logger.enabled
                else []
            ),
        ],
    )
    runner = LangChainRunner(
        executor_holder=executor_holder,
        memory=memory,
        tracker=tracker,
        model=model,
        tool_mode=tool_mode,
        tool_holder=tool_holder,
        run_config=run_config,
        thinking_holder=thinking_holder,
        thinking_handler=thinking_handler,
        rebuild_fn=rebuild,
        change_mode_fn=change_mode,
        console=console,
    )

    session_id = f"lc-{os.urandom(6).hex()}"
    usage_mw = UsageMiddleware()
    langfuse_mw = LangfuseMiddleware(session_id=session_id, model=model, framework="langchain") if langfuse else None
    middleware = [
        usage_mw,
        OTelMiddleware(framework="langchain"),
        MLflowMiddleware(mlflow_turn_logger),
        *(([langfuse_mw]) if langfuse_mw else []),
        ConfirmMiddleware(console),
    ]

    session = AgentSession(
        runner=runner,
        middleware=middleware,
        console=console,
        model=model,
        tool_mode=tool_mode,
        command_handler=runner.handle_command,
        input_guard=_preflight_input_guard_reply,
        extras={"thinking_holder": thinking_holder},
    )
    session.run()


# ─── LangChainRunner (TurnRunner adapter) ────────────────────────────────────


class LangChainRunner:
    """Thin TurnRunner adapter wrapping the LangChain executor.

    All LangChain-specific logic (streaming, memory, tracker, thinking) lives
    here.  The REPL loop, OTel, MLflow, and confirm gate are handled by
    AgentSession + middleware, keeping this class focused on execution only.
    """

    def __init__(
        self,
        *,
        executor_holder: dict[str, Any],
        memory: Any,
        tracker: _UsageTracker,
        model: str,
        tool_mode: str,
        tool_holder: dict[str, Any] | None = None,
        run_config: dict[str, Any],
        thinking_holder: list[bool],
        thinking_handler: Any,
        rebuild_fn: Callable[[], None],
        change_mode_fn: Callable[[str], None] | None = None,
        console: Any,
    ) -> None:
        self._executor_holder = executor_holder
        self._memory = memory
        self._tracker = tracker
        self._model = model
        self._tool_mode = tool_mode
        self._tool_holder = tool_holder
        self._run_config = run_config
        self._thinking_holder = thinking_holder
        self._thinking_handler = thinking_handler
        self._rebuild_fn = rebuild_fn
        self._change_mode_fn = change_mode_fn
        self._console = console

    # ── TurnRunner protocol ───────────────────────────────────────────────────

    def run_turn(self, ctx: Any) -> Any:
        from kubeflow_mcp.agents.runtime.contracts import TurnResult

        console = ctx.extras.get("console", self._console)

        self._tracker.reset_turn()
        executor = self._executor_holder["executor"]
        is_react = self._executor_holder.get("agent_mode") == "ReAct"
        try:
            output = _run_turn_streaming(
                executor,
                ctx.user_input,
                self._run_config,
                console,
                thinking_handler=self._thinking_handler,
                is_react=is_react,
            )
        except TypeError:
            output = _run_turn_blocking(executor, ctx.user_input, self._run_config)

        self._tracker.finish_turn()
        if output.strip():
            print_assistant_panel(console, output)
        _print_turn_stats(console, self._tracker, self._model)

        return TurnResult(
            text=output,
            tool_calls=[{"name": n} for n in self._tracker.turn_tool_names],
            usage={
                "prompt_tokens": self._tracker.turn_input,
                "completion_tokens": self._tracker.turn_output,
            },
            llm_calls=self._tracker.turn_llm_calls,
        )

    def rebuild(self, *, model: str | None = None, tool_mode: str | None = None) -> None:
        self._rebuild_fn()

    def change_mode(self, new_mode: str) -> None:
        """Switch tool mode, reload tools, and rebuild the executor."""
        if self._change_mode_fn is None:
            return
        self._change_mode_fn(new_mode)
        self._tool_mode = new_mode

    # ── REPL command handler ──────────────────────────────────────────────────

    def handle_command(self, line: str) -> bool:
        """Return True when the input is a slash command that was handled."""
        from kubeflow_mcp.agents.runtime.repl_commands import handle_common_repl_command, CommonReplHandlers

        def _on_tools() -> None:
            print_tools_table(
                self._console,
                [(t.name, t.description or "") for t in self._executor_holder["lc_tools"]],
                header_style="bold cyan",
            )
            print_tip(self._console, f"Mode: {self._tool_mode}  |  Total tools: {len(self._executor_holder['lc_tools'])}")

        def _on_think() -> None:
            self._thinking_holder[0] = not self._thinking_holder[0]
            self._rebuild_fn()
            state = "on" if self._thinking_holder[0] else "off"
            print_tip(self._console, f"Thinking mode: {state}  (agent: {self._executor_holder['agent_mode']})")

        def _on_export() -> None:
            session = _build_langchain_export_payload(
                self._memory,
                model=self._model,
                tool_mode=self._tool_mode,
                tracker=self._tracker,
            )
            out = export_session_snapshot(session)
            print_tip(self._console, f"Session exported → {out}")

        def _on_import(path: str) -> None:
            try:
                restored = _import_langchain_session(self._memory, self._tracker, path)
                print_tip(self._console, f"Session imported ← {path}  ({restored} turns restored)")
            except Exception as exc:
                print_tip(self._console, f"Import error: {exc}", style="red")

        def _on_clear() -> None:
            self._memory.clear()
            reset_token_totals(self._tracker)
            print_tip(self._console, "Conversation cleared.")

        def _on_mode(arg: str) -> None:
            from kubeflow_mcp.agents.runtime.repl_commands import VALID_MODES

            if not arg:
                tool_count = len(self._executor_holder.get("lc_tools") or [])
                print_tip(
                    self._console,
                    f"Current mode: {self._tool_mode}  ({tool_count} tools)  |  "
                    f"Available: {', '.join(VALID_MODES)}  |  Usage: /mode <mode>",
                )
                return
            new_mode = arg.lower()
            if new_mode not in VALID_MODES:
                print_tip(
                    self._console,
                    f"Unknown mode {new_mode!r}. Choose: {', '.join(VALID_MODES)}",
                    style="red",
                )
                return
            if new_mode == self._tool_mode:
                print_tip(self._console, f"Already in {new_mode!r} mode.")
                return
            print_tip(self._console, f"Switching to {new_mode!r} mode…", style="dim")
            try:
                self.change_mode(new_mode)
            except Exception as exc:
                print_tip(self._console, f"Mode switch failed: {exc}", style="red")
                return
            tool_count = len(self._executor_holder.get("lc_tools") or [])
            print_tip(self._console, f"Mode: {new_mode}  ({tool_count} tools)")

        def _on_help() -> None:
            from kubeflow_mcp.agents.runtime.repl_commands import VALID_MODES

            rows = [
                ("/help", "Show this help message"),
                ("/tools", "List active tools for the current mode"),
                (f"/mode [{' | '.join(VALID_MODES)}]", "Show or switch tool mode (no arg = show current)"),
                ("/think", "Toggle chain-of-thought thinking display on/off"),
                ("/export", "Save current session to a JSON snapshot file"),
                ("/import <file>", "Restore session from a JSON snapshot file"),
                ("/clear", "Clear conversation history and token counters"),
                ("exit / quit / q", "Exit the agent"),
            ]
            self._console.print()
            from rich.table import Table
            t = Table(show_header=False, box=None, padding=(0, 2))
            t.add_column(style="bright_white", no_wrap=True, min_width=32)
            t.add_column(style="dim")
            for cmd, desc in rows:
                t.add_row(cmd, desc)
            self._console.print(t)
            self._console.print()

        def _on_unknown(command: str) -> None:
            print_tip(
                self._console,
                f"Unknown command: {command!r}. "
                "Try /help  /tools  /mode <mode>  /think  /export  /import <file>  /clear",
            )

        handlers = CommonReplHandlers(
            on_tools=_on_tools,
            on_think=_on_think,
            on_export=_on_export,
            on_import=_on_import,
            on_clear=_on_clear,
            on_unknown=_on_unknown,
            on_mode=_on_mode,
            on_help=_on_help,
        )
        return handle_common_repl_command(line, handlers)
