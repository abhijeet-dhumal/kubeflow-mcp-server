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

"""Shared trace context helpers for all agent framework observability."""

from __future__ import annotations

import os


def trace_mode() -> str:
    """Return 'safe' or 'full' based on THINK_TRACE_MODE env var."""
    mode = os.environ.get("THINK_TRACE_MODE", "full").strip().lower()
    return mode if mode in {"safe", "full"} else "full"


def _sanitize_output(text: str) -> str:
    """Strip ReAct scaffolding (Thought/Action/Observation) from text."""
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


def trace_text(text: str) -> str:
    """Return sanitized text when THINK_TRACE_MODE=safe, else return as-is."""
    if trace_mode() != "safe":
        return text
    return _sanitize_output(text)


def trim_preview(text: str, *, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]} …"


def update_trace_context(
    *,
    session_id: str | None = None,
    framework: str = "unknown",
    request_preview: str | None = None,
    response_preview: str | None = None,
) -> None:
    """Best-effort update of the active MLflow trace with session metadata."""
    try:
        import mlflow

        if mlflow.get_current_active_span() is None:
            return
        trace_user = (
            os.environ.get("MLFLOW_TRACE_USER") or os.environ.get("USER") or "unknown"
        )
        mlflow.update_current_trace(
            session_id=session_id or os.environ.get("KUBEFLOW_MCP_SESSION_ID"),
            user=trace_user,
            request_preview=request_preview,
            response_preview=response_preview,
            tags={
                "agent.framework": framework,
                "agent.session_id": os.environ.get("KUBEFLOW_MCP_SESSION_ID", ""),
                "agent.trace_mode": trace_mode(),
            },
        )
    except Exception:
        return
