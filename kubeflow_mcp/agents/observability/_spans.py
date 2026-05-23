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

"""MLflow TOOL span helpers usable by any agent framework adapter."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from ._context import trace_text, trim_preview, update_trace_context
from ._otel import _SESSION_ID_VAR, tool_call_span


def _tool_span_name(fn_name: str, args: dict[str, Any]) -> str:
    """Return a human-readable span name for a tool invocation."""
    if fn_name == "execute_tool":
        target = str(args.get("tool_name") or "").strip()
        if target:
            return f"tool:{target}"
    return f"tool:{fn_name}"


def _span_output_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value)
    if len(text) > 4000:
        text = f"{text[:4000]} …"
    return {"text": text}


def invoke_with_mlflow_span(
    fn: Callable[..., Any],
    args: dict[str, Any],
    *,
    framework: str = "unknown",
) -> Any:
    """Call ``fn(**args)`` and record a MLflow TOOL span around it.

    Falls back to a plain call when MLflow is not configured or available.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return fn(**args)
    try:
        import mlflow
        from mlflow.entities import SpanType
    except Exception:
        return fn(**args)

    # _SESSION_ID_VAR is set by MlflowSessionLogger.log_turn before invoking tools.
    session_id = _SESSION_ID_VAR.get() or os.environ.get("KUBEFLOW_MCP_SESSION_ID", "")
    tool_name = _tool_span_name(fn.__name__, args)
    try:
        span_cm = mlflow.start_span(
            name=tool_name,
            span_type=SpanType.TOOL,
            attributes={
                "agent.framework": framework,
                "agent.wrapper_tool": fn.__name__,
                "agent.session_id": session_id,
            },
        )
    except Exception:
        return fn(**args)

    with span_cm as mlflow_span:
        mlflow_span.set_inputs({"args": args})
        update_trace_context(
            framework=framework,
            request_preview=trim_preview(trace_text(json.dumps(args, default=str))),
        )
        with tool_call_span(
            tool_name,
            args,
            framework=framework,
            session_id=session_id,
        ) as otel_span:
            try:
                result = fn(**args)
            except Exception as exc:
                mlflow_span.record_exception(exc)
                mlflow_span.set_outputs({"error": str(exc)})
                raise
            otel_span.set_attribute("tool.success", True)
        mlflow_span.set_outputs(_span_output_payload(result))
        update_trace_context(
            framework=framework,
            response_preview=trim_preview(trace_text(str(_span_output_payload(result)))),
        )
        return result
