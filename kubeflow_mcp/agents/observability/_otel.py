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

"""OpenTelemetry tracer for the Kubeflow MCP agent layer.

All imports are guarded so this module is safe to import even when the
``opentelemetry-sdk`` package is not installed.  Use ``setup_otel_tracer``
once at startup; everything else is a no-op when OTel is unconfigured.

Two OTel sources are defined in the architecture:
- ``kubeflow-mcp-agent``  — this module (agent REPL turns + tool calls)
- ``kubeflow-mcp-server`` — FastMCP native OTel (activated via env vars only)
"""

from __future__ import annotations

import contextvars
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_TRACER_NAME = "kubeflow-mcp-agent"
_SERVICE_NAME = "kubeflow-mcp-agent"

# Set by MlflowSessionLogger.log_turn(); read by invoke_with_mlflow_span via
# _spans.py so tool spans are automatically children of the correct turn.
_SESSION_ID_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "otel_session_id", default="unknown"
)

# Populated by setup_otel_tracer(); None means OTel is disabled / not installed.
_tracer: Any | None = None


def setup_otel_tracer(
    *,
    endpoint: str | None = None,
    service_name: str = _SERVICE_NAME,
    insecure: bool = True,
) -> bool:
    """Configure a global OTLP/HTTP tracer.  Returns True when active.

    Args:
        endpoint: OTLP HTTP endpoint, e.g. ``http://localhost:4318``.
                  Falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
                  No-op when neither is set.
        service_name: OTel ``service.name`` resource attribute.
        insecure: Ignored (HTTP exporter has no TLS by default); kept for
                  API compatibility if callers need to pass it explicitly.
    """
    global _tracer  # noqa: PLW0603

    resolved = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not resolved:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False

    # Normalise: traces endpoint is base_url + /v1/traces
    traces_url = resolved.rstrip("/") + "/v1/traces"

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=traces_url))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_TRACER_NAME)
    return True


def get_tracer() -> Any:
    """Return the active OTel tracer, or a no-op proxy when OTel is off."""
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace

        return trace.get_tracer(_TRACER_NAME)
    except ImportError:
        return _NoOpTracer()


# ── Span context managers ──────────────────────────────────────────────────────


@contextmanager
def agent_turn_span(
    user_input: str,
    *,
    framework: str,
    session_id: str,
    model: str,
) -> Iterator[Any]:
    """OTel span wrapping one full REPL turn.

    Attributes set on the span:
      - ``agent.framework``
      - ``agent.session_id``
      - ``agent.model``
      - ``user.input.preview`` (first 200 chars)
    """
    if _tracer is None:
        yield _NoOpSpan()
        return

    with _tracer.start_as_current_span("agent.turn") as span:
        span.set_attribute("agent.framework", framework)
        span.set_attribute("agent.session_id", session_id)
        span.set_attribute("agent.model", model)
        span.set_attribute("user.input.preview", user_input[:200])
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise


@contextmanager
def tool_call_span(
    tool_name: str,
    args: dict[str, Any],
    *,
    framework: str,
    session_id: str,
) -> Iterator[Any]:
    """OTel CLIENT span wrapping one tool invocation.

    Attributes set on the span:
      - ``tool.name``
      - ``tool.args_preview`` (first 300 chars of JSON-serialised args)
      - ``agent.framework``
      - ``agent.session_id``
    """
    if _tracer is None:
        yield _NoOpSpan()
        return

    try:
        from opentelemetry.trace import SpanKind

        kind = SpanKind.CLIENT
    except ImportError:
        kind = None  # type: ignore[assignment]

    args_preview = json.dumps(args, default=str)[:300]

    span_kwargs: dict[str, Any] = {"name": f"tool.{tool_name}"}
    if kind is not None:
        span_kwargs["kind"] = kind

    with _tracer.start_as_current_span(**span_kwargs) as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.args_preview", args_preview)
        span.set_attribute("agent.framework", framework)
        span.set_attribute("agent.session_id", session_id)
        try:
            yield span
        except Exception as exc:
            span.set_attribute("tool.success", False)
            span.record_exception(exc)
            raise
        else:
            span.set_attribute("tool.success", True)


# ── No-op stubs (used when OTel SDK is absent) ────────────────────────────────


class _NoOpSpan:
    """Minimal span stub that silently drops all calls."""

    def set_attribute(self, *_: Any, **__: Any) -> None:
        pass

    def record_exception(self, *_: Any, **__: Any) -> None:
        pass


class _NoOpTracer:
    """Minimal tracer stub returned when OTel SDK is not installed."""

    def start_as_current_span(self, *_: Any, **__: Any) -> Any:
        from contextlib import nullcontext

        return nullcontext(_NoOpSpan())
