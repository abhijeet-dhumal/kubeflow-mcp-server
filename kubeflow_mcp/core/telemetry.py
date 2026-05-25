# Copyright The Kubeflow Authors
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

"""Server-side OpenTelemetry tracing (Gap 6A).

Provides OTLP/HTTP tracer setup for the FastMCP server layer, distinct from the
agent-layer tracer in ``agents/observability/_otel.py``.

Service name: ``"kubeflow-mcp-server"`` (differentiated from the agent's
``"kubeflow-mcp-agent"`` in Jaeger to make cross-service trace stitching obvious).

Fixes vs PR #21 review:
  - ``service_name`` is ``"kubeflow-mcp-server"`` not ``"kubeflow-mcp"``.
  - ``get_tracer()`` is called *once per tool registration* (outside wrapper) in
    ``_audit_wrap``, not on every invocation.
  - Exceptions set ``StatusCode.ERROR`` via ``_audit_wrap``.
  - ``_NoopTracer.start_as_current_span`` accepts ``**kwargs``.

Usage::

    from kubeflow_mcp.core.telemetry import setup_tracing, get_tracer

    setup_tracing(endpoint="http://localhost:4318")
    tracer = get_tracer()                          # call once, share the reference
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRACER_NAME = "kubeflow_mcp_server"
_SERVICE_NAME = "kubeflow-mcp-server"

_tracer: Any | None = None
_setup_done: bool = False


def setup_tracing(
    endpoint: str | None = None,
    *,
    service_name: str = _SERVICE_NAME,
) -> bool:
    """Configure a global OTLP/HTTP tracer for the MCP server layer.

    Idempotent: subsequent calls are no-ops unless ``endpoint`` changes.

    Args:
        endpoint: OTLP HTTP base URL, e.g. ``http://localhost:4318``.
                  Also reads ``OTEL_EXPORTER_OTLP_ENDPOINT`` from the environment.
        service_name: OTel ``service.name`` resource attribute.

    Returns:
        True when tracing is successfully configured.
    """
    global _tracer, _setup_done  # noqa: PLW0603

    if _setup_done:
        return _tracer is not None

    resolved = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not resolved:
        _setup_done = True
        return False

    try:
        from opentelemetry import trace as _trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.debug("opentelemetry-sdk not installed — server tracing disabled")
        _setup_done = True
        return False

    import atexit

    traces_url = resolved.rstrip("/") + "/v1/traces"
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=traces_url, timeout=2),
            export_timeout_millis=2000,
        )
    )
    _trace.set_tracer_provider(provider)
    _tracer = _trace.get_tracer(_TRACER_NAME)
    _setup_done = True

    # Flush + shutdown before Python's atexit joins threads to avoid deadlock
    # when the collector is unreachable (e.g. --otel-endpoint with no collector).
    atexit.register(provider.shutdown)

    logger.info(f"Server OTel tracing enabled → {traces_url} (service={service_name})")
    return True


def get_tracer() -> Any:
    """Return the active server tracer, or a no-op stub when OTel is off.

    Call this ONCE per tool registration and cache the result to avoid
    repeated tracer lookups on each tool invocation.
    """
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace as _trace

        return _trace.get_tracer(_TRACER_NAME)
    except ImportError:
        return _NoopTracer()


# ── No-op stubs ───────────────────────────────────────────────────────────────


class _NoopSpan:
    """Span stub that silently drops all calls."""

    def is_recording(self) -> bool:
        return False

    def set_attribute(self, *_: Any, **__: Any) -> None:
        pass

    def set_status(self, *_: Any, **__: Any) -> None:
        pass

    def record_exception(self, *_: Any, **__: Any) -> None:
        pass

    def get_span_context(self) -> Any:
        return _NoopSpanContext()


class _NoopSpanContext:
    trace_id: int = 0
    span_id: int = 0


class _NoopTracer:
    """Tracer stub returned when the OTel SDK is not installed."""

    def start_as_current_span(self, *_: Any, **__: Any) -> Any:
        from contextlib import nullcontext

        return nullcontext(_NoopSpan())
