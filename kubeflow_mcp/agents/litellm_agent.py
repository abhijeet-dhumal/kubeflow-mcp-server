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

"""LiteLLM-native agentic loop for kubeflow-mcp.

Replaces the LlamaIndex FunctionAgent with an explicit async generator loop that
owns confirmation gates, tool schema caching, cost tracking, and session export.

Compatible with any model supported by LiteLLM:
    ollama/gemma4:e4b, ollama/qwen2.5:7b, gpt-4.1, anthropic/claude-sonnet-4-5, …

Install:
    uv sync --extra agents-litellm   # or --extra agents for all backends
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import uuid
from collections.abc import AsyncGenerator, Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

_SPHINX_BUILD = "sphinx" in sys.modules
_PYTEST_RUNNING = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in __import__("os").environ

_litellm_module: Any = None


def _litellm() -> Any:
    """Lazily import and configure litellm on first use (~670ms deferred to first agent call)."""
    global _litellm_module  # noqa: PLW0603
    if _litellm_module is not None:
        return _litellm_module
    try:
        import litellm as _ll

        _ll.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        _litellm_module = _ll
    except ImportError:
        if not (_SPHINX_BUILD or _PYTEST_RUNNING):
            sys.exit(
                "Error: required packages not installed\n"
                "Run: uv sync --extra agents-litellm   (or --extra agents for all backends)"
            )
    return _litellm_module


# ─── Agent event types ───────────────────────────────────────────────────────
# Each iteration of _agentic_loop yields (event_type, payload):
#   "text_delta"     str    — streaming text token from the model
#   "tool_call"      dict   — {"name", "args", "call_id"} before execution
#   "tool_result"    dict   — {"name", "call_id", "result", "error"} after
#   "confirm_needed" dict   — {"name", "args", "call_id"} needs y/n from user
#   "error"          dict   — {"type", "message"} unrecoverable provider error
#   "done"           None   — loop finished cleanly
AgentEvent = tuple[str, Any]


class LoopState(Enum):
    RUNNING = "running"
    AWAITING_CONFIRM = "awaiting_confirm"
    DONE = "done"
    ERROR = "error"


# ─── Type annotation → JSON schema ───────────────────────────────────────────

from kubeflow_mcp.agents.core.schema import _annotation_to_json_schema, build_tool_schema


# ─── Streaming tool-call delta accumulation ──────────────────────────────────


def _accumulate_tool_calls(accumulated: list[dict], delta_tcs: list) -> None:
    """Merge streaming tool-call delta chunks into a running list."""
    for dtc in delta_tcs:
        idx = dtc.index
        while len(accumulated) <= idx:
            accumulated.append(
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
        tc = accumulated[idx]
        if dtc.id:
            tc["id"] = dtc.id
        if dtc.function:
            if dtc.function.name:
                tc["function"]["name"] += dtc.function.name
            if dtc.function.arguments:
                tc["function"]["arguments"] += dtc.function.arguments


# ─── Text-mode tool call fallback ────────────────────────────────────────────
# Some local models (e.g. gemma4, older mistral) don't use the native function-
# calling protocol and instead output tool call JSON as plain text. This parser
# detects those patterns and converts them to the same accumulated_tcs format.

# Patterns covered:
#   {"name": "tool", "arguments": {...}}       ← Ollama text fallback
#   {"name": "tool", "parameters": {...}}      ← some mistral variants
#   <tool_call>{"name": ...}</tool_call>       ← some llama variants
_TEXT_TC_RE = re.compile(
    r"(?:<tool_call>)?\s*"
    r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*'
    r'"(?:arguments|parameters)"\s*:\s*(?P<args>\{[^}]*\}|\{\})'
    r"\s*\}(?:</tool_call>)?",
    re.DOTALL,
)


def _extract_text_tool_calls(text: str) -> list[dict]:
    """Parse tool calls embedded as plain text (models without native function calling).

    Returns a list in the same format as ``accumulated_tcs`` so the agentic loop
    can process them identically.
    """
    results = []
    for m in _TEXT_TC_RE.finditer(text):
        try:
            args = json.loads(m.group("args"))
        except json.JSONDecodeError:
            args = {}
        results.append(
            {
                "id": f"text_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": m.group("name"), "arguments": json.dumps(args)},
            }
        )
    return results


# ─── Tool collection helpers (delegated to core/tools.py) ───────────────────
from kubeflow_mcp.agents.core.tools import _get_full_mode_tools, _get_meta_mode_tools


# ─── LiteLLMAgent ────────────────────────────────────────────────────────────


class LiteLLMAgent:
    """Explicit async agentic loop backed by LiteLLM.

    Features:
    - Streaming text + streaming tool-call delta accumulation
    - Confirmation gate — pauses before any call with ``confirmed=False``
    - ``_audit_wrap`` applied to all tools (rate-limit + circuit-breaker + log)
    - Per-call cost tracking via ``litellm.completion_cost``
    - Session export as plain JSON for replay / evaluation
    - Hot model + mode switching without losing message history

    Usage::

        agent = LiteLLMAgent("ollama/gemma4:e4b")
        async for event_type, data in agent._agentic_loop("list my training jobs"):
            ...
    """

    def __init__(
        self,
        model: str,
        tool_mode: str = "full",
        base_url: str | None = None,
        fallback_model: str | None = None,
        num_retries: int = 3,
        thinking: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tool_mode = tool_mode
        self._base_url = base_url
        self._fallback_model = fallback_model
        self._num_retries = num_retries
        self._thinking: bool = thinking
        self._extra_body = extra_body
        self._pending_confirm: dict[str, Any] | None = None
        self._total_cost: float = 0.0

        self._tool_schemas, self._tool_map = self._build_mode_assets(tool_mode)

        try:
            from kubeflow_mcp.core.server import build_agent_instruction_text

            system_prompt = build_agent_instruction_text()
        except ImportError:
            system_prompt = (
                "You are a Kubeflow training assistant. Help users manage ML training jobs."
            )

        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # ─── Setup ───────────────────────────────────────────────────────────────

    def _build_mode_assets(
        self, mode: str
    ) -> tuple[list[dict], dict[str, Callable[..., Any]]]:
        if mode == "full":
            tools, descriptions = _get_full_mode_tools()
        elif mode in ("progressive", "semantic"):
            tools, descriptions = _get_meta_mode_tools(mode)
        else:
            raise ValueError(f"Invalid tool_mode '{mode}'. Choose: full, progressive, semantic")

        try:
            from kubeflow_mcp.core.server import (  # type: ignore[attr-defined]
                _audit_wrap as audit_wrap,
            )
        except (ImportError, AttributeError):

            def audit_wrap(fn: Callable) -> Callable:  # type: ignore[misc]
                return fn

        schemas = [build_tool_schema(fn, descriptions.get(fn.__name__, "")) for fn in tools]
        tool_map: dict[str, Callable] = {fn.__name__: audit_wrap(fn) for fn in tools}
        return schemas, tool_map

    # ─── Public controls ─────────────────────────────────────────────────────

    def switch_model(self, model: str) -> None:
        """Hot-swap the model without resetting conversation history."""
        self.model = model

    def switch_mode(self, mode: str) -> None:
        """Switch tool mode; rebuilds schemas and tool map in place."""
        self._tool_schemas, self._tool_map = self._build_mode_assets(mode)
        self.tool_mode = mode

    def clear(self) -> None:
        """Reset conversation while keeping the system prompt."""
        system = self.messages[0]
        self.messages = [system]
        self._pending_confirm = None
        self._total_cost = 0.0

    def export_session(self) -> dict[str, Any]:
        """Export the full session for replay, evaluation, or archiving."""
        return {
            "model": self.model,
            "tool_mode": self.tool_mode,
            "total_cost_usd": self._total_cost,
            "messages": list(self.messages),
        }

    def import_session(self, payload: dict[str, Any]) -> int:
        """Import session payload exported by LiteLLM or framework adapters."""
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            msg = "Session payload must include a 'messages' list."
            raise ValueError(msg)
        loaded_messages: list[dict[str, Any]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if not isinstance(role, str):
                continue
            loaded_messages.append(dict(item))
        if not loaded_messages:
            msg = "Session payload had no valid chat messages."
            raise ValueError(msg)
        self.messages = loaded_messages
        self._pending_confirm = None
        self._total_cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
        return len(self.messages)

    def token_count(self) -> int:
        """Estimate total tokens in the current conversation."""
        try:
            return _litellm().token_counter(model=self.model, messages=self.messages)
        except Exception:
            return sum(len(str(m.get("content", ""))) // 4 for m in self.messages)

    # ─── Completion kwargs ────────────────────────────────────────────────────

    def _completion_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "tools": self._tool_schemas,
            "tool_choice": "auto",
            "stream": True,
            "num_retries": self._num_retries,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        from kubeflow_mcp.agents.frameworks._thinking import thinking_completion_kwargs

        kwargs.update(thinking_completion_kwargs(enabled=self._thinking, model=self.model))
        return kwargs

    # ─── Core loop ───────────────────────────────────────────────────────────

    async def _agentic_loop(self, user_message: str) -> AsyncGenerator[AgentEvent, None]:
        """Async generator: send ``user_message`` and yield events until done.

        Yields ``AgentEvent`` tuples — see module docstring for the full list.
        Halts and yields ``"confirm_needed"`` when a tool requests ``confirmed=False``;
        call :meth:`continue_after_confirm` to resume.
        """
        if user_message.strip():
            self.messages.append({"role": "user", "content": user_message})

        while True:
            try:
                response = await _litellm().acompletion(**self._completion_kwargs())
            except Exception as exc:
                logger.error("LiteLLM completion failed: %s", exc)
                yield ("error", {"type": type(exc).__name__, "message": str(exc)})
                yield ("done", None)
                return

            full_content = ""
            accumulated_tcs: list[dict] = []
            last_chunk = None

            async for chunk in response:
                last_chunk = chunk
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    full_content += delta.content
                    yield ("text_delta", delta.content)
                if getattr(delta, "tool_calls", None):
                    _accumulate_tool_calls(accumulated_tcs, delta.tool_calls)

            # Track cost when usage is available
            if last_chunk is not None:
                try:
                    cost = _litellm().completion_cost(completion_response=last_chunk)
                    self._total_cost += cost
                except Exception:
                    pass

            full_content, accumulated_tcs = self._apply_text_tc_fallback(
                full_content, accumulated_tcs
            )

            if not accumulated_tcs:
                self.messages.append({"role": "assistant", "content": full_content})
                yield ("done", None)
                return

            self.messages.append(
                {
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": accumulated_tcs,
                }
            )

            confirm_fired = False
            for tc in accumulated_tcs:
                fn_name = tc["function"]["name"]
                call_id = tc["id"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                fn_args = {k: v for k, v in fn_args.items() if v is not None}

                yield ("tool_call", {"name": fn_name, "args": fn_args, "call_id": call_id})

                if fn_args.get("confirmed") is False:
                    self._pending_confirm = {
                        "name": fn_name,
                        "args": fn_args,
                        "call_id": call_id,
                    }
                    yield ("confirm_needed", self._pending_confirm.copy())
                    confirm_fired = True
                    break

                result, is_error = await self._run_tool(fn_name, fn_args)
                yield (
                    "tool_result",
                    {"name": fn_name, "call_id": call_id, "result": result, "error": is_error},
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, default=str),
                    }
                )

            if confirm_fired:
                return
            # All tools executed — loop back for LLM synthesis

    async def continue_after_confirm(
        self, approved: bool
    ) -> AsyncGenerator[AgentEvent, None]:
        """Resume the loop after the user responds to a confirmation gate.

        Args:
            approved: ``True`` → re-run the tool with ``confirmed=True``.
                      ``False`` → tell the model the action was cancelled.

        Yields the same ``AgentEvent`` types as :meth:`_agentic_loop`.
        """
        if self._pending_confirm is None:
            return

        pending = self._pending_confirm
        self._pending_confirm = None

        if not approved:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending["call_id"],
                    "content": json.dumps({"cancelled": True, "reason": "User declined"}),
                }
            )
        else:
            approved_args = {**pending["args"], "confirmed": True}
            yield ("tool_call", {
                "name": pending["name"],
                "args": approved_args,
                "call_id": pending["call_id"],
            })
            result, is_error = await self._run_tool(pending["name"], approved_args)
            yield (
                "tool_result",
                {
                    "name": pending["name"],
                    "call_id": pending["call_id"],
                    "result": result,
                    "error": is_error,
                },
            )
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending["call_id"],
                    "content": json.dumps(result, default=str),
                }
            )

        async for event in self._agentic_loop(""):
            yield event

    @staticmethod
    def _apply_text_tc_fallback(
        full_content: str, accumulated_tcs: list[dict]
    ) -> tuple[str, list[dict]]:
        """Promote text-embedded tool calls when the model doesn't use native function calling."""
        if accumulated_tcs or not full_content:
            return full_content, accumulated_tcs
        text_tcs = _extract_text_tool_calls(full_content)
        if text_tcs:
            cleaned = _TEXT_TC_RE.sub("", full_content).strip()
            return cleaned, text_tcs
        return full_content, accumulated_tcs

    async def _run_tool(self, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
        """Execute a registered tool, returning (result, is_error)."""
        fn = self._tool_map.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name!r}"}, True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: fn(**args))
            is_error = isinstance(result, dict) and (
                "error" in result or "error_code" in result
            )
            return result, is_error
        except Exception as exc:
            logger.debug("Tool %r raised: %s", name, exc, exc_info=True)
            return {"error": str(exc), "error_type": type(exc).__name__}, True
