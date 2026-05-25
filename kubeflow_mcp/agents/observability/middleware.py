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

"""Pluggable observability middleware (Gap 3C).

Each class implements TurnMiddleware — receives a TurnContext and a ``next``
callable, performs its cross-cutting concern, and returns a TurnResult.

Data-lineage guarantee
----------------------
OTelMiddleware opens the ``agent.turn`` span BEFORE calling next(), so every
tool OTel span created by the runner is a child of that turn span.
MLflowMiddleware runs inside OTelMiddleware and sets the _SESSION_ID_VAR so
tool spans carry the right session ID as an OTel attribute.

Recommended wiring order (outermost → innermost):
    UsageMiddleware → OTelMiddleware → MLflowMiddleware → LangfuseMiddleware
      → ConfirmMiddleware → FrameworkRunner
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from kubeflow_mcp.agents.runtime.contracts import TurnContext, TurnResult


# ── OTelMiddleware ────────────────────────────────────────────────────────────


class OTelMiddleware:
    """Open an ``agent.turn`` OTel span around the framework runner.

    The span is opened BEFORE calling ``next(ctx)`` so all tool spans created
    during the turn are correctly parented.  When OTel is disabled or the SDK
    is missing this middleware is a transparent passthrough.

    Args:
        framework: Label used in span attribute ``agent.framework``.
        enabled: Set to False to skip OTel even if configured.
    """

    def __init__(self, *, framework: str, enabled: bool = True) -> None:
        self._framework = framework
        self._enabled = enabled

    def __call__(self, ctx: TurnContext, next: Callable[[TurnContext], TurnResult]) -> TurnResult:
        from kubeflow_mcp.agents.observability._otel import _SESSION_ID_VAR, agent_turn_span

        _SESSION_ID_VAR.set(ctx.session_id)

        with agent_turn_span(
            ctx.user_input,
            framework=self._framework,
            session_id=ctx.session_id,
            model=ctx.model,
        ) as span:
            result = next(ctx)
            if span is not None:
                try:
                    span.set_attribute("agent.tool_call_count", len(result.tool_calls))
                    span.set_attribute("agent.duration_ms", round(result.duration_ms))
                    if result.error:
                        span.set_attribute("error", result.error)
                    # Capture OTel trace ID for cross-correlation.
                    try:
                        from opentelemetry import trace as _trace

                        sc = _trace.get_current_span().get_span_context()
                        result.otel_trace_id = format(sc.trace_id, "032x")
                    except Exception:
                        pass
                except Exception:
                    pass
            return result


# ── MLflowMiddleware ──────────────────────────────────────────────────────────


class MLflowMiddleware:
    """Log each turn to MLflow using MlflowSessionLogger.

    Must be placed INSIDE (i.e. called after) OTelMiddleware so the active OTel
    span is already open when log_turn annotates it.

    Args:
        logger: A MlflowSessionLogger instance shared across the session.
    """

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    def __call__(self, ctx: TurnContext, next: Callable[[TurnContext], TurnResult]) -> TurnResult:
        # Get the currently active OTel span (opened by OTelMiddleware above us).
        try:
            from opentelemetry import trace as _trace

            otel_span = _trace.get_current_span()
        except ImportError:
            otel_span = None

        t0 = time.monotonic()
        result = next(ctx)
        duration_s = time.monotonic() - t0

        self._logger.log_turn(
            user_input=ctx.user_input,
            assistant_output=result.text,
            tool_names=[tc.get("name", "") for tc in result.tool_calls],
            tool_call_count=len(result.tool_calls),
            llm_call_count=result.llm_calls,
            input_tokens=result.usage.get("prompt_tokens", 0),
            output_tokens=result.usage.get("completion_tokens", 0),
            duration_s=duration_s,
            otel_span=otel_span,
        )
        result.mlflow_run_id = self._logger.run_id
        return result

    def close(self) -> None:
        """End the MLflow run — called by AgentSession._on_close()."""
        self._logger.close()


# ── LangfuseMiddleware ────────────────────────────────────────────────────────


class LangfuseMiddleware:
    """Emit a Langfuse trace + generation span for each agent turn (Gap 2b).

    Produces one Langfuse trace per turn with:
    - Session grouping via ``session_id``
    - A ``generation`` sub-span carrying model, token usage, and latency
    - ``result.langfuse_trace_id`` set so callers (e.g. eval runners) can
      post scores against the trace via ``LangfuseMiddleware.score()``

    When Langfuse is not installed or ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``
    are not set this is a transparent no-op passthrough.

    Args:
        session_id: Shared across all turns for Langfuse session grouping.
        user_id: Optional user identifier (shown in Langfuse UI).
        model: Default model name used when ``TurnContext.model`` is empty.
        framework: Agent framework label attached as trace metadata.
    """

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
        model: str = "unknown",
        framework: str = "unknown",
    ) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._model = model
        self._framework = framework
        self._lf: Any | None = None

        try:
            import langfuse as _lf_mod

            self._lf = _lf_mod.Langfuse()
        except Exception:
            pass

    def __call__(self, ctx: TurnContext, next: Callable[[TurnContext], TurnResult]) -> TurnResult:
        if self._lf is None:
            return next(ctx)

        from datetime import datetime, timezone

        model = ctx.model or self._model
        t0 = datetime.now(timezone.utc)
        trace = None

        try:
            trace = self._lf.trace(
                name="agent.turn",
                session_id=self._session_id,
                user_id=self._user_id,
                input=ctx.user_input,
                metadata={
                    "model": model,
                    "framework": self._framework,
                    "tool_mode": ctx.tool_mode,
                },
            )
        except Exception:
            pass

        result = next(ctx)

        t1 = datetime.now(timezone.utc)

        if trace is not None:
            try:
                trace.generation(
                    name="agent.turn.generation",
                    model=model,
                    input=ctx.user_input,
                    output=result.text,
                    start_time=t0,
                    end_time=t1,
                    usage={
                        "input": result.usage.get("prompt_tokens", 0),
                        "output": result.usage.get("completion_tokens", 0),
                        "total": result.usage.get("total_tokens", 0),
                    },
                    metadata={
                        "tool_calls_count": len(result.tool_calls or []),
                        "duration_ms": result.duration_ms,
                    },
                )
                trace.update(output=result.text[:2000])
                result.langfuse_trace_id = trace.id
            except Exception:
                pass

        return result

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        """Post a numeric score against a trace (for eval runners)."""
        if self._lf is None:
            return
        try:
            self._lf.score(trace_id=trace_id, name=name, value=value, comment=comment)
        except Exception:
            pass

    def flush(self) -> None:
        if self._lf is not None:
            try:
                self._lf.flush()
            except Exception:
                pass

    def close(self) -> None:
        self.flush()


# ── UsageMiddleware ───────────────────────────────────────────────────────────


class UsageMiddleware:
    """Accumulate per-turn token and cost totals across the session.

    Reads ``result.usage`` dict (keys: ``prompt_tokens``, ``completion_tokens``,
    ``total_cost``) and accumulates into ``self.totals`` for the REPL to display
    in the status line.
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_cost": 0.0,
            "turns": 0.0,
        }

    def __call__(self, ctx: TurnContext, next: Callable[[TurnContext], TurnResult]) -> TurnResult:
        result = next(ctx)
        usage = result.usage
        self.totals["prompt_tokens"] += float(usage.get("prompt_tokens", 0))
        self.totals["completion_tokens"] += float(usage.get("completion_tokens", 0))
        self.totals["total_cost"] += float(usage.get("total_cost", 0.0))
        self.totals["turns"] += 1.0
        return result
