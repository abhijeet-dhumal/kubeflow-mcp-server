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

from kubeflow_mcp.agents.frameworks.langchain_agent import (
    _allow_multi_step_for_query,
    _compact_react_result,
    _is_invalid_preflight_model,
    _is_multi_call_blocked_observation,
    _multi_call_blocked_reply,
    _normalize_react_args,
    _parse_function_like_tool_input,
    _preflight_input_guard_reply,
    _sanitize_react_output,
    _suggest_hf_model_ids,
    _validate_execute_tool_payload,
)
from kubeflow_mcp.agents.observability._context import trace_text as _trace_text
from kubeflow_mcp.agents.observability._context import trim_preview as _trim_preview
from kubeflow_mcp.agents.observability._session import (
    _sanitize_mlflow_metric_key,
    _session_messages_from_transcript,
)
from kubeflow_mcp.agents.observability._spans import (
    _span_output_payload,
)
from kubeflow_mcp.agents.observability._spans import (
    _tool_span_name as _mlflow_tool_span_name,
)


def test_normalize_execute_tool_moves_flat_args_to_arguments():
    normalized = _normalize_react_args(
        "execute_tool",
        {"tool_name": "pre_flight", "model": "google/gemma-2-2b"},
    )

    assert normalized == {
        "tool_name": "pre_flight",
        "arguments": {"model": "google/gemma-2-2b"},
    }


def test_normalize_execute_tool_function_style_captures_params():
    normalized = _normalize_react_args(
        "execute_tool",
        {"function": "pre_flight()", "model": "google/gemma-2-2b"},
    )

    assert normalized == {
        "tool_name": "pre_flight",
        "arguments": {"model": "google/gemma-2-2b"},
    }


def test_compact_react_result_truncates_large_lists():
    result = {
        "success": True,
        "data": {"runtimes": [{"name": f"rt-{i}"} for i in range(20)]},
    }

    compact = _compact_react_result("execute_tool", result)
    assert compact.get("_note")
    runtimes = compact["data"]["runtimes"]
    assert len(runtimes) == 9
    assert runtimes[-1] == "... (+12 more)"


def test_invalid_preflight_model_rejects_ollama_style():
    assert _is_invalid_preflight_model("hf://gemma4:e4b")
    assert _is_invalid_preflight_model("gemma4:e4b")
    assert not _is_invalid_preflight_model("hf://google/gemma-2-2b")


def test_sanitize_react_output_keeps_final_answer_only():
    raw = "Thought: internal\nAction: execute_tool\nFinal Answer: done"
    assert _sanitize_react_output(raw) == "done"


def test_preflight_input_guard_flags_ollama_style_model():
    msg = _preflight_input_guard_reply("run preflight check for hf://gemma4:e4b model")
    assert msg is not None
    assert "requires a HuggingFace model ID" in msg
    assert "gemma4:e4b" in msg


def test_allow_multi_step_for_query_only_when_explicit():
    assert not _allow_multi_step_for_query("run preflight for google/gemma-2-2b")
    assert _allow_multi_step_for_query("continue with next step")


def test_suggest_hf_model_ids_prefers_safer_small_official(monkeypatch):
    class _Model:
        def __init__(self, model_id: str) -> None:
            self.id = model_id

    def _fake_list_models(*_args, **_kwargs):
        return [
            _Model("HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced"),
            _Model("google/gemma-4-31B-it"),
            _Model("google/gemma-4-E4B-it"),
        ]

    monkeypatch.setattr("huggingface_hub.list_models", _fake_list_models)
    suggestions = _suggest_hf_model_ids("hf://gemma4:e4b", limit=3)
    assert suggestions[0] == "google/gemma-4-E4B-it"


def test_multi_call_block_observation_detected():
    assert _is_multi_call_blocked_observation(
        '{"error": "Multiple tool calls blocked for this turn"}'
    )
    assert not _is_multi_call_blocked_observation('{"success": true}')


def test_multi_call_block_reply_is_actionable():
    reply = _multi_call_blocked_reply()
    assert "stopped before auto-running another tool" in reply
    assert "continue" in reply


def test_sanitize_mlflow_metric_key_normalizes_tool_names():
    assert _sanitize_mlflow_metric_key("execute_tool") == "execute_tool"
    assert _sanitize_mlflow_metric_key("Tool Call / Pre Flight") == "tool_call_pre_flight"
    assert _sanitize_mlflow_metric_key("!!!") == "unknown"


def test_mlflow_tool_span_name_prefers_inner_execute_tool_name():
    assert _mlflow_tool_span_name("execute_tool", {"tool_name": "pre_flight"}) == "tool:pre_flight"
    assert _mlflow_tool_span_name("list_tools", {}) == "tool:list_tools"


def test_span_output_payload_truncates_strings():
    payload = _span_output_payload("x" * 5000)
    assert isinstance(payload, dict)
    assert payload["text"].endswith("…")


def test_trim_preview_truncates_long_text():
    out = _trim_preview("abcdef", max_chars=3)
    assert out == "abc …"


def test_parse_function_like_tool_input_extracts_kwargs():
    parsed = _parse_function_like_tool_input('pre_flight(model="google/gemma-4-E4B-it")')
    assert parsed == {
        "tool_name": "pre_flight",
        "arguments": {"model": "google/gemma-4-E4B-it"},
    }


def test_validate_execute_tool_payload_requires_preflight_model():
    args = {"tool_name": "pre_flight", "arguments": {}}
    msg = _validate_execute_tool_payload(args)
    assert msg is not None
    assert "model is required" in msg


def test_trace_text_safe_mode_sanitizes(monkeypatch):
    monkeypatch.setenv("THINK_TRACE_MODE", "safe")
    raw = "Thought: hidden\nFinal Answer: visible"
    assert _trace_text(raw) == "visible"


def test_session_messages_from_transcript_flattens_turns():
    messages = _session_messages_from_transcript(
        [
            {"user": "hi", "assistant": "hello"},
            {"user": "list runtimes", "assistant": "found 28"},
        ]
    )
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "list runtimes"},
        {"role": "assistant", "content": "found 28"},
    ]
