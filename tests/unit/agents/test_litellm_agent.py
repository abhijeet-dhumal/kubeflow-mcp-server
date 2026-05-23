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

"""Unit tests for kubeflow_mcp.agents.litellm_agent.

Tests focus on the pure logic: schema building, type annotation conversion,
delta accumulation, and agent state management. LiteLLM completions are mocked
so no network calls are made.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kubeflow_mcp.agents.litellm_agent import (
    LiteLLMAgent,
    LoopState,
    _accumulate_tool_calls,
    _annotation_to_json_schema,
    build_tool_schema,
)

# ─── _annotation_to_json_schema ──────────────────────────────────────────────


class TestAnnotationToJsonSchema:
    def test_str(self):
        assert _annotation_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert _annotation_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _annotation_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _annotation_to_json_schema(bool) == {"type": "boolean"}

    def test_list_str(self):
        result = _annotation_to_json_schema(list[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_list_int(self):
        result = _annotation_to_json_schema(list[int])
        assert result == {"type": "array", "items": {"type": "integer"}}

    def test_dict(self):
        result = _annotation_to_json_schema(dict[str, Any])
        assert result == {"type": "object"}

    def test_optional_str(self):
        result = _annotation_to_json_schema(str | None)
        assert result == {"type": "string"}

    def test_optional_int(self):
        result = _annotation_to_json_schema(int | None)
        assert result == {"type": "integer"}

    def test_union_pipe_syntax(self):
        result = _annotation_to_json_schema(str | None)
        assert result == {"type": "string"}

    def test_unknown_type_falls_back_to_string(self):
        class MyCustomType:
            pass

        result = _annotation_to_json_schema(MyCustomType)
        assert result == {"type": "string"}


# ─── build_tool_schema ───────────────────────────────────────────────────────


class TestBuildToolSchema:
    def test_simple_function(self):
        def my_tool(name: str, count: int) -> str:
            """Get something by name and count."""
            return f"{name} {count}"

        schema = build_tool_schema(my_tool)
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "my_tool"
        assert fn["description"] == "Get something by name and count."
        props = fn["parameters"]["properties"]
        assert props["name"] == {"type": "string"}
        assert props["count"] == {"type": "integer"}
        assert fn["parameters"]["required"] == ["name", "count"]

    def test_optional_param_not_required(self):
        def tool_with_default(model: str, namespace: str | None = None) -> dict:
            """A tool."""
            return {}

        schema = build_tool_schema(tool_with_default)
        required = schema["function"]["parameters"]["required"]
        assert "model" in required
        assert "namespace" not in required

    def test_description_override(self):
        def my_fn(x: str) -> None:
            """Original docstring."""

        schema = build_tool_schema(my_fn, description="Custom description")
        assert schema["function"]["description"] == "Custom description"

    def test_skips_meta_params(self):
        def fn(self, ctx, _meta, real_param: str) -> None:
            """Fn."""

        schema = build_tool_schema(fn)
        props = schema["function"]["parameters"]["properties"]
        assert "self" not in props
        assert "ctx" not in props
        assert "_meta" not in props
        assert "real_param" in props

    def test_list_param(self):
        def fn(items: list[str]) -> None:
            """Fn."""

        schema = build_tool_schema(fn)
        prop = schema["function"]["parameters"]["properties"]["items"]
        assert prop == {"type": "array", "items": {"type": "string"}}

    def test_no_docstring_uses_function_name(self):
        def undocumented(x: str) -> None:
            pass

        schema = build_tool_schema(undocumented)
        assert schema["function"]["description"] == "undocumented"


# ─── _accumulate_tool_calls ──────────────────────────────────────────────────


class TestAccumulateToolCalls:
    def _make_delta(self, index: int, tc_id: str, name: str, arguments: str):
        dtc = MagicMock()
        dtc.index = index
        dtc.id = tc_id
        dtc.function = MagicMock()
        dtc.function.name = name
        dtc.function.arguments = arguments
        return dtc

    def test_single_complete_call(self):
        accumulated: list[dict] = []
        delta = self._make_delta(0, "call_1", "my_tool", '{"x": 1}')
        _accumulate_tool_calls(accumulated, [delta])
        assert len(accumulated) == 1
        assert accumulated[0]["id"] == "call_1"
        assert accumulated[0]["function"]["name"] == "my_tool"
        assert accumulated[0]["function"]["arguments"] == '{"x": 1}'

    def test_streaming_merge(self):
        """Multiple deltas for same call_id get concatenated."""
        accumulated: list[dict] = []
        d1 = self._make_delta(0, "call_1", "my_tool", '{"x":')
        d2 = self._make_delta(0, "", "", " 1}")
        _accumulate_tool_calls(accumulated, [d1])
        _accumulate_tool_calls(accumulated, [d2])
        assert accumulated[0]["function"]["arguments"] == '{"x": 1}'

    def test_multiple_calls(self):
        accumulated: list[dict] = []
        d0 = self._make_delta(0, "call_0", "tool_a", '{}')
        d1 = self._make_delta(1, "call_1", "tool_b", '{"k": "v"}')
        _accumulate_tool_calls(accumulated, [d0, d1])
        assert len(accumulated) == 2
        assert accumulated[0]["function"]["name"] == "tool_a"
        assert accumulated[1]["function"]["name"] == "tool_b"


# ─── LoopState ───────────────────────────────────────────────────────────────


class TestLoopState:
    def test_all_states_defined(self):
        assert LoopState.RUNNING.value == "running"
        assert LoopState.AWAITING_CONFIRM.value == "awaiting_confirm"
        assert LoopState.DONE.value == "done"
        assert LoopState.ERROR.value == "error"


# ─── LiteLLMAgent (mocked LiteLLM) ──────────────────────────────────────────

_FAKE_SYSTEM_PROMPT = "You are a Kubeflow training assistant."


def _make_text_chunk(content: str, finish_reason=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = finish_reason
    return chunk


def _make_tool_chunk(index: int, tc_id: str, name: str, arguments: str):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    delta = MagicMock()
    delta.content = None
    dtc = MagicMock()
    dtc.index = index
    dtc.id = tc_id
    dtc.function = MagicMock()
    dtc.function.name = name
    dtc.function.arguments = arguments
    delta.tool_calls = [dtc]
    chunk.choices[0].delta = delta
    return chunk


async def _async_iter(items):
    for item in items:
        yield item


def _make_mock_litellm() -> MagicMock:
    """Return a MagicMock that stands in for the litellm module.

    ``acompletion`` is an AsyncMock — the loop does ``await litellm.acompletion(...)``
    and then ``async for chunk in response``, so the mock must be awaitable and its
    return_value must be an async iterable.
    """
    mock_ll = MagicMock()
    mock_ll.token_counter.return_value = 100
    mock_ll.completion_cost.return_value = 0.0
    mock_ll.acompletion = AsyncMock()
    return mock_ll


@pytest.fixture
def mock_agent():
    """LiteLLMAgent with litellm + server/trainer imports patched out."""
    import kubeflow_mcp.agents.litellm_agent as agent_module

    mock_ll = _make_mock_litellm()
    original_litellm = agent_module.litellm

    def stub_list_jobs(namespace: str | None = None) -> dict:
        """List training jobs."""
        return {"jobs": ["job-1", "job-2"]}

    def stub_delete_job(name: str, confirmed: bool = False) -> dict:
        """Delete a training job. Set confirmed=True to execute."""
        if not confirmed:
            return {"preview": True, "name": name}
        return {"deleted": name}

    # Inject the mock litellm module before any patching of its attributes
    agent_module.litellm = mock_ll

    with (
        patch("kubeflow_mcp.agents.litellm_agent._get_full_mode_tools") as mock_tools,
        patch(
            "kubeflow_mcp.core.server.build_agent_instruction_text",
            return_value=_FAKE_SYSTEM_PROMPT,
            create=True,
        ),
    ):
        mock_tools.return_value = (
            [stub_list_jobs, stub_delete_job],
            {
                "stub_list_jobs": "List training jobs.",
                "stub_delete_job": "Delete a training job.",
            },
        )

        agent = LiteLLMAgent(model="ollama/test-model", tool_mode="full")
        agent._tool_map = {
            "stub_list_jobs": stub_list_jobs,
            "stub_delete_job": stub_delete_job,
        }
        yield agent

    agent_module.litellm = original_litellm


class TestLiteLLMAgentState:
    def test_initial_messages_has_system(self, mock_agent):
        assert mock_agent.messages[0]["role"] == "system"
        assert "Kubeflow" in mock_agent.messages[0]["content"]

    def test_switch_model(self, mock_agent):
        mock_agent.switch_model("gpt-4.1")
        assert mock_agent.model == "gpt-4.1"

    def test_switch_mode_invalid(self, mock_agent):
        with pytest.raises(ValueError, match="Invalid tool_mode"):
            mock_agent.switch_mode("invalid_mode")

    def test_clear_resets_history(self, mock_agent):
        mock_agent.messages.append({"role": "user", "content": "hello"})
        mock_agent._total_cost = 0.5
        mock_agent.clear()
        assert len(mock_agent.messages) == 1
        assert mock_agent.messages[0]["role"] == "system"
        assert mock_agent._total_cost == 0.0

    def test_export_session(self, mock_agent):
        session = mock_agent.export_session()
        assert session["model"] == "ollama/test-model"
        assert session["tool_mode"] == "full"
        assert isinstance(session["messages"], list)
        assert "total_cost_usd" in session

    def test_import_session(self, mock_agent):
        payload = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "total_cost_usd": 1.23,
        }
        loaded = mock_agent.import_session(payload)
        assert loaded == 3
        assert mock_agent.messages[1]["content"] == "hi"
        assert mock_agent._total_cost == 1.23

    def test_token_count_fallback(self, mock_agent):
        import kubeflow_mcp.agents.litellm_agent as agent_module

        agent_module.litellm.token_counter.side_effect = Exception("no litellm")
        count = mock_agent.token_count()
        assert isinstance(count, int)
        assert count >= 0
        agent_module.litellm.token_counter.side_effect = None


class TestAgenticLoop:
    def _set_acompletion(self, mock_agent, return_value=None, side_effect=None):
        """Configure the AsyncMock acompletion on the injected litellm stub.

        ``return_value`` should be an async iterable (the streaming response).
        ``side_effect`` should be an async callable returning an async iterable.
        """
        import kubeflow_mcp.agents.litellm_agent as agent_module

        if side_effect is not None:
            agent_module.litellm.acompletion.side_effect = side_effect
        else:
            agent_module.litellm.acompletion.side_effect = None
            agent_module.litellm.acompletion.return_value = return_value

    def test_text_only_response(self, mock_agent):
        chunks = [_make_text_chunk("Hello "), _make_text_chunk("world!"), _make_text_chunk("")]
        self._set_acompletion(mock_agent, return_value=_async_iter(chunks))

        async def run():
            events = []
            async for ev in mock_agent._agentic_loop("hi"):
                events.append(ev)
            return events

        events = asyncio.get_event_loop().run_until_complete(run())
        types_seen = [e[0] for e in events]
        assert "text_delta" in types_seen
        assert events[-1][0] == "done"

    def test_tool_call_executed(self, mock_agent):
        """LLM returns a tool call for stub_list_jobs → result appended."""
        tool_chunk = _make_tool_chunk(0, "call_abc", "stub_list_jobs", "{}")
        text_chunk_after = _make_text_chunk("Here are your jobs.")
        call_count = 0

        async def fake_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            return _async_iter([tool_chunk] if call_count == 1 else [text_chunk_after])

        self._set_acompletion(mock_agent, side_effect=fake_acompletion)

        async def run():
            events = []
            async for ev in mock_agent._agentic_loop("list jobs"):
                events.append(ev)
            return events

        events = asyncio.get_event_loop().run_until_complete(run())
        event_types = [e[0] for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "done" in event_types

        tool_result = next(e for e in events if e[0] == "tool_result")
        assert tool_result[1]["name"] == "stub_list_jobs"
        assert not tool_result[1]["error"]

    def test_confirm_needed_pauses_loop(self, mock_agent):
        """Tool call with confirmed=False yields confirm_needed and stops."""
        delete_chunk = _make_tool_chunk(
            0, "call_del", "stub_delete_job", json.dumps({"name": "job-1", "confirmed": False})
        )
        self._set_acompletion(mock_agent, return_value=_async_iter([delete_chunk]))

        async def run():
            events = []
            async for ev in mock_agent._agentic_loop("delete job-1"):
                events.append(ev)
            return events

        events = asyncio.get_event_loop().run_until_complete(run())
        event_types = [e[0] for e in events]
        assert "confirm_needed" in event_types
        assert "done" not in event_types
        assert mock_agent._pending_confirm is not None
        assert mock_agent._pending_confirm["name"] == "stub_delete_job"

    def test_continue_after_confirm_approved(self, mock_agent):
        """Approved confirm executes tool with confirmed=True then calls LLM."""
        mock_agent._pending_confirm = {
            "name": "stub_delete_job",
            "args": {"name": "job-1", "confirmed": False},
            "call_id": "call_del",
        }
        mock_agent.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_del",
                        "type": "function",
                        "function": {
                            "name": "stub_delete_job",
                            "arguments": json.dumps({"name": "job-1", "confirmed": False}),
                        },
                    }
                ],
            }
        )
        self._set_acompletion(
            mock_agent, return_value=_async_iter([_make_text_chunk("Job deleted.")])
        )

        async def run():
            events = []
            async for ev in mock_agent.continue_after_confirm(approved=True):
                events.append(ev)
            return events

        events = asyncio.get_event_loop().run_until_complete(run())
        event_types = [e[0] for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "done" in event_types
        assert mock_agent._pending_confirm is None

    def test_continue_after_confirm_denied(self, mock_agent):
        """Denied confirm appends cancellation tool message and calls LLM."""
        mock_agent._pending_confirm = {
            "name": "stub_delete_job",
            "args": {"name": "job-1", "confirmed": False},
            "call_id": "call_del",
        }
        mock_agent.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_del",
                        "type": "function",
                        "function": {
                            "name": "stub_delete_job",
                            "arguments": json.dumps({"name": "job-1", "confirmed": False}),
                        },
                    }
                ],
            }
        )
        self._set_acompletion(
            mock_agent, return_value=_async_iter([_make_text_chunk("Understood.")])
        )

        async def run():
            events = []
            async for ev in mock_agent.continue_after_confirm(approved=False):
                events.append(ev)
            return events

        events = asyncio.get_event_loop().run_until_complete(run())
        event_types = [e[0] for e in events]
        assert "tool_call" not in event_types
        assert "done" in event_types
        tool_msgs = [m for m in mock_agent.messages if m["role"] == "tool"]
        assert any("cancelled" in m["content"] for m in tool_msgs)

    def test_unknown_tool_returns_error(self, mock_agent):
        """_run_tool for unknown name returns error without raising."""
        result, is_error = asyncio.get_event_loop().run_until_complete(
            mock_agent._run_tool("nonexistent_tool", {})
        )
        assert is_error
        assert "error" in result

    def test_run_tool_exception_is_caught(self, mock_agent):
        """Exceptions in tool execution are caught and returned as error."""
        mock_agent._tool_map["boom"] = lambda: (_ for _ in ()).throw(RuntimeError("boom!"))

        result, is_error = asyncio.get_event_loop().run_until_complete(
            mock_agent._run_tool("boom", {})
        )
        assert is_error
        assert "RuntimeError" in result.get("error_type", "")

    def test_completion_error_emits_error_event(self, mock_agent):
        import kubeflow_mcp.agents.litellm_agent as agent_module

        async def fail_acompletion(**kwargs):
            raise RuntimeError("provider unavailable")

        agent_module.litellm.acompletion.side_effect = fail_acompletion

        async def run():
            events = []
            async for ev in mock_agent._agentic_loop("hi"):
                events.append(ev)
            return events

        events = asyncio.get_event_loop().run_until_complete(run())
        assert events[0][0] == "error"
        assert events[0][1]["type"] == "RuntimeError"
        assert "provider unavailable" in events[0][1]["message"]
        assert events[-1][0] == "done"
