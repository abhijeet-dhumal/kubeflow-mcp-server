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

"""Unit tests for kubeflow_mcp.agents.observability."""

import kubeflow_mcp.agents.observability as obs
from kubeflow_mcp.agents.observability._context import (
    _sanitize_output,
    trace_mode,
    trace_text,
    trim_preview,
    update_trace_context,
)
from kubeflow_mcp.agents.observability._session import (
    MlflowSessionLogger,
    _sanitize_mlflow_metric_key,
    _session_messages_from_transcript,
)
from kubeflow_mcp.agents.observability._spans import (
    _span_output_payload,
    _tool_span_name,
    invoke_with_mlflow_span,
)

# ── _context.py ──────────────────────────────────────────────────────────────


def test_trace_mode_defaults_to_full(monkeypatch):
    monkeypatch.delenv("THINK_TRACE_MODE", raising=False)
    assert trace_mode() == "full"


def test_trace_mode_safe_when_set(monkeypatch):
    monkeypatch.setenv("THINK_TRACE_MODE", "safe")
    assert trace_mode() == "safe"


def test_trace_mode_ignores_unknown_value(monkeypatch):
    monkeypatch.setenv("THINK_TRACE_MODE", "verbose")
    assert trace_mode() == "full"


def test_trace_text_passthrough_in_full_mode(monkeypatch):
    monkeypatch.setenv("THINK_TRACE_MODE", "full")
    raw = "Thought: hidden\nFinal Answer: visible"
    assert trace_text(raw) == raw


def test_trace_text_sanitizes_in_safe_mode(monkeypatch):
    monkeypatch.setenv("THINK_TRACE_MODE", "safe")
    raw = "Thought: hidden\nFinal Answer: visible"
    assert trace_text(raw) == "visible"


def test_sanitize_output_extracts_final_answer():
    assert _sanitize_output("Thought: x\nFinal Answer: done") == "done"


def test_sanitize_output_strips_react_lines():
    raw = "Thought: think\nAction: tool\nreal output"
    result = _sanitize_output(raw)
    assert "Thought:" not in result
    assert "real output" in result


def test_trim_preview_short_string_unchanged():
    assert trim_preview("hello") == "hello"


def test_trim_preview_truncates():
    assert trim_preview("abcdef", max_chars=3) == "abc …"


def test_update_trace_context_no_error_without_mlflow(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    update_trace_context(framework="test", request_preview="hi")


# ── _spans.py ─────────────────────────────────────────────────────────────────


def test_tool_span_name_resolves_execute_tool_inner():
    assert _tool_span_name("execute_tool", {"tool_name": "pre_flight"}) == "tool:pre_flight"


def test_tool_span_name_falls_back_to_fn_name():
    assert _tool_span_name("list_runtimes", {}) == "tool:list_runtimes"


def test_span_output_payload_passthrough_for_dict():
    d = {"a": 1}
    assert _span_output_payload(d) is d


def test_span_output_payload_wraps_long_string():
    result = _span_output_payload("x" * 5000)
    assert isinstance(result, dict)
    assert result["text"].endswith("…")


def test_invoke_with_mlflow_span_no_mlflow_uri_calls_fn(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    calls = []

    def dummy(a, b):
        calls.append((a, b))
        return a + b

    result = invoke_with_mlflow_span(dummy, {"a": 1, "b": 2}, framework="test")
    assert result == 3
    assert calls == [(1, 2)]


def test_invoke_with_mlflow_span_propagates_exceptions(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    def boom(**_):
        raise ValueError("oops")

    import pytest

    with pytest.raises(ValueError, match="oops"):
        invoke_with_mlflow_span(boom, {}, framework="test")


# ── _session.py ───────────────────────────────────────────────────────────────


def test_sanitize_mlflow_metric_key_normalizes():
    assert _sanitize_mlflow_metric_key("execute_tool") == "execute_tool"
    assert _sanitize_mlflow_metric_key("Tool Call / Pre Flight") == "tool_call_pre_flight"
    assert _sanitize_mlflow_metric_key("!!!") == "unknown"


def test_session_messages_from_transcript_flattens():
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


def test_session_messages_from_transcript_skips_empty():
    messages = _session_messages_from_transcript([{"user": "q", "assistant": ""}])
    assert messages == [{"role": "user", "content": "q"}]


def test_mlflow_session_logger_disabled_without_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    logger = MlflowSessionLogger(model="test/model", tool_mode="full", framework="test")
    assert not logger.enabled
    logger.log_turn(user_input="hello", assistant_output="hi")
    logger.close()


def test_mlflow_session_logger_log_turn_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    logger = MlflowSessionLogger(model="m", tool_mode="full", framework="langchain")
    logger.log_turn(
        user_input="u",
        assistant_output="a",
        tool_names=["pre_flight"],
        tool_call_count=1,
        llm_call_count=2,
        input_tokens=100,
        output_tokens=50,
        duration_s=1.5,
    )
    assert logger._step == 0


# ── __init__.py re-exports ────────────────────────────────────────────────────


def test_public_api_exports():
    assert hasattr(obs, "MlflowSessionLogger")
    assert hasattr(obs, "invoke_with_mlflow_span")
    assert hasattr(obs, "trace_mode")
    assert hasattr(obs, "trace_text")
    assert hasattr(obs, "trim_preview")
    assert hasattr(obs, "update_trace_context")
