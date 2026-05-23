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
from kubeflow_mcp.agents.frameworks._confirm import (
    make_console_confirm_handler,
    set_confirm_handler,
    wrap_with_confirm,
)
from kubeflow_mcp.agents.frameworks._observability import is_local_ollama_model, setup_langsmith
from kubeflow_mcp.agents.frameworks._thinking import (
    apply_thinking_to_chat_litellm,
    extract_thinking_delta,
    is_answer_content_token,
)
from kubeflow_mcp.agents.frameworks._tools import get_system_prompt, load_tools
from kubeflow_mcp.agents.observability import (
    MlflowSessionLogger,
    invoke_with_mlflow_span,
    trace_text,
    trim_preview,
    update_trace_context,
)
from kubeflow_mcp.agents.runtime.repl_commands import (
    REPL_EXIT_COMMANDS,
    CommonReplHandlers,
    handle_common_repl_command,
)
from kubeflow_mcp.agents.runtime.session_state import (
    build_session_snapshot,
    export_session_snapshot,
    import_session_snapshot,
    reset_token_totals,
)

_SPHINX_BUILD = "sphinx" in sys.modules

try:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    if not _SPHINX_BUILD:
        sys.exit("Error: install kubeflow-mcp[agents-langchain]")
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




def _sanitize_react_output(text: str) -> str:
    if not text:
        return text
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


def _build_executor(
    *,
    model: str,
    llm: Any,
    tool_fns: list[Callable[..., Any]],
    descriptions: dict[str, str],
    system_prompt: str,
    run_config: dict[str, Any],
    memory: Any,
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
        lc_tools = [
            StructuredTool.from_function(
                func=fn,
                name=fn.__name__,
                description=descriptions.get(fn.__name__, fn.__doc__ or fn.__name__),
            )
            for fn in wrapped
        ]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, lc_tools, prompt)
        extra = {}

    executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        verbose=False,
        max_iterations=15,
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




def _make_thinking_display_handler(console, enabled_holder: list[bool]):
    """LangChain callback: stream reasoning_content to the terminal."""
    from langchain_core.callbacks.base import BaseCallbackHandler

    class _ThinkingHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            self._active = False

        def reset(self) -> None:
            if self._active:
                console.print()
            self._active = False

        def on_llm_new_token(self, token, *, chunk=None, **kwargs) -> None:
            if not enabled_holder[0]:
                return
            delta = extract_thinking_delta(token, chunk)
            if delta:
                if not self._active:
                    self._active = True
                    console.print()
                    console.print("[dim italic]💭 ", end="")
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


def _render_stream_chunk(console, chunk: dict[str, Any]) -> str | None:
    """Render tool call / observation panels; return last real tool observation."""
    last_obs: str | None = None
    for action in chunk.get("actions") or []:
        tool = getattr(action, "tool", None)
        if tool and tool not in _INTERNAL_TOOLS:
            args = _normalize_react_args(
                tool, _parse_react_tool_input(getattr(action, "tool_input", "") or "")
            )
            print_tool_call_panel(console, tool, args)

    for step in chunk.get("steps") or []:
        action = getattr(step, "action", None)
        tool = getattr(action, "tool", None) if action else None
        obs = getattr(step, "observation", None)
        if obs and not _is_internal_observation(tool, obs):
            print_tool_result_panel(console, format_tool_result_display(obs))
            last_obs = obs

    for action, observation in chunk.get("intermediate_step") or []:
        tool = getattr(action, "tool", None)
        if tool and tool not in _INTERNAL_TOOLS:
            args = _normalize_react_args(
                tool, _parse_react_tool_input(getattr(action, "tool_input", "") or "")
            )
            print_tool_call_panel(console, tool, args)
        if observation and not _is_internal_observation(tool, observation):
            print_tool_result_panel(console, format_tool_result_display(observation))
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
) -> str:
    """Stream ReAct steps; show tool panels live, return final answer."""
    if thinking_handler is not None:
        thinking_handler.reset()
    _TURN_POLICY["allow_multi_step"] = _allow_multi_step_for_query(line)
    _TURN_POLICY["tool_calls"] = 0
    output = ""
    last_obs: str | None = None
    parse_retries = 0
    for chunk in executor.stream({"input": line}, config=run_config):
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
            output = _sanitize_react_output(str(chunk["output"]))
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
    output = result.get("output", "") if isinstance(result, dict) else str(result)
    if _is_multi_call_blocked_observation(output):
        return _multi_call_blocked_reply()
    return _sanitize_react_output(str(output))


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


def _run_langchain_repl(  # noqa: C901
    *,
    console,
    executor_holder: dict[str, Any],
    memory,
    tracker: _UsageTracker,
    model: str,
    tool_mode: str,
    run_config: dict[str, Any],
    thinking_holder: list[bool],
    thinking_handler: Any,
    mlflow_turn_logger: MlflowSessionLogger | None,
    rebuild: Callable[[], None],
) -> None:
    def _on_tools() -> None:
        print_tools_table(
            console,
            [(tool.name, tool.description or "") for tool in executor_holder["lc_tools"]],
            header_style="bold cyan",
        )
        print_tip(console, f"Mode: {tool_mode}  |  Total tools: {len(executor_holder['lc_tools'])}")

    def _on_think() -> None:
        thinking_holder[0] = not thinking_holder[0]
        rebuild()
        state = "on" if thinking_holder[0] else "off"
        print_tip(console, f"Thinking mode: {state}  (agent: {executor_holder['agent_mode']})")

    def _on_export() -> None:
        session = _build_langchain_export_payload(
            memory,
            model=model,
            tool_mode=tool_mode,
            tracker=tracker,
        )
        out = export_session_snapshot(session)
        print_tip(console, f"Session exported → {out}")

    def _on_import(path: str) -> None:
        try:
            restored = _import_langchain_session(memory, tracker, path)
            print_tip(console, f"Session imported ← {path}  ({restored} turns restored)")
        except Exception as exc:
            print_tip(console, f"Import error: {exc}", style="red")

    def _on_clear() -> None:
        memory.clear()
        reset_token_totals(tracker)
        print_tip(console, "Conversation cleared.")

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

        preflight_hint = _preflight_input_guard_reply(line)
        if preflight_hint:
            print_assistant_panel(console, preflight_hint)
            continue

        print_user_panel(console, line)
        try:
            tracker.reset_turn()
            executor = executor_holder["executor"]
            try:
                output = _run_turn_streaming(
                    executor, line, run_config, console, thinking_handler=thinking_handler
                )
            except TypeError:
                output = _run_turn_blocking(executor, line, run_config)
            tracker.finish_turn()
            if output.strip():
                print_assistant_panel(console, output)
            _print_turn_stats(console, tracker, model)
            if mlflow_turn_logger is not None:
                mlflow_turn_logger.log_turn(
                    user_input=line,
                    assistant_output=output,
                    tool_names=tracker.turn_tool_names,
                    tool_call_count=tracker.turn_tools,
                    llm_call_count=tracker.turn_llm_calls,
                    input_tokens=tracker.turn_input,
                    output_tokens=tracker.turn_output,
                    duration_s=tracker.turn_duration,
                )
        except KeyboardInterrupt:
            print_tip(console, "\nInterrupted.")
        except Exception as exc:
            print_error_panel(console, exc)


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
    console = get_console()
    set_confirm_handler(make_console_confirm_handler(console))
    os.environ["KUBEFLOW_MCP_MODEL"] = model
    tracker = _UsageTracker(model)
    mlflow_turn_logger = MlflowSessionLogger(model=model, tool_mode=tool_mode, framework="langchain")
    thinking_holder = [thinking]
    thinking_handler = _make_thinking_display_handler(console, thinking_holder)
    run_config = _build_run_config(tracker, model, tool_mode, thinking_handler)

    tool_fns, descriptions = load_tools(tool_mode)
    system_prompt = get_system_prompt()
    executor_holder: dict[str, Any] = {}

    def rebuild() -> None:
        llm_kwargs: dict[str, Any] = {"model": model, "streaming": True, "num_retries": num_retries}
        if base_url:
            llm_kwargs["api_base"] = base_url
        apply_thinking_to_chat_litellm(llm_kwargs, enabled=thinking_holder[0], model=model)
        llm = ChatLiteLLM(**llm_kwargs)
        executor, lc_tools, agent_mode = _build_executor(
            model=model,
            llm=llm,
            tool_fns=tool_fns,
            descriptions=descriptions,
            system_prompt=system_prompt,
            run_config=run_config,
            memory=memory,
        )
        executor_holder["executor"] = executor
        executor_holder["lc_tools"] = lc_tools
        executor_holder["agent_mode"] = agent_mode

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        memory = ConversationBufferMemory(memory_key="chat_history", output_key="output")

    rebuild()
    agent_mode = executor_holder["agent_mode"]
    lc_tools = executor_holder["lc_tools"]

    backend_label = base_url or "cloud / local auto-detect"
    tracing = setup_langsmith(langfuse=langfuse)

    print_welcome_panel(
        panel_title="kubeflow-mcp · LangChain",
        border_style="bright_cyan",
        rows=[
            ("bold bright_cyan", "Kubeflow MCP — LangChain + LiteLLM"),
            ("white", f"Model   : {model}"),
            ("white", f"Agent   : {agent_mode}"),
            ("white", f"Backend : {backend_label}"),
            ("white", f"Mode    : {tool_mode}  ({len(lc_tools)} tools)"),
            *([("dim", f"Tracing : {tracing}")] if tracing else []),
            ("dim", ""),
            ("dim", "Commands: /tools  /think  /export  /import <file>  /clear  exit"),
            ("dim", f"Thinking: {'on' if thinking else 'off'} ( /think to toggle )"),
            ("dim", "Confirm gate on mutating tools (confirmed=False)"),
            ("dim", "LangSmith: --langfuse or LANGCHAIN_TRACING_V2=true"),
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
    try:
        _run_langchain_repl(
            console=console,
            executor_holder=executor_holder,
            memory=memory,
            tracker=tracker,
            model=model,
            tool_mode=tool_mode,
            run_config=run_config,
            thinking_holder=thinking_holder,
            thinking_handler=thinking_handler,
            mlflow_turn_logger=mlflow_turn_logger,
            rebuild=rebuild,
        )
    finally:
        mlflow_turn_logger.close()
