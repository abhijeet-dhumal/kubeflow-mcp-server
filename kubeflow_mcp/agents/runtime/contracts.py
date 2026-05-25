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

"""Pluggable agent runtime contracts (Gap 3B).

Design principle
----------------
Observability and lifecycle middleware *wrap* the framework runner; they are
never *inside* it.  Every framework adapter implements ``TurnRunner``; every
cross-cutting concern implements ``TurnMiddleware``.  ``build_chain`` wires
them into a single callable that AgentSession invokes per user turn.

Middleware execution order (outermost → innermost):
  UsageMiddleware → OTelMiddleware → MLflowMiddleware → LangfuseMiddleware
    → ConfirmMiddleware → FrameworkRunner

Data lineage guarantee
----------------------
``OTelMiddleware`` opens the ``agent.turn`` span *before* calling next(), so all
tool OTel spans created by the runner inherit the correct parent context.
``MLflowMiddleware`` sets ``_SESSION_ID_VAR`` inside that span so tool spans
also carry the correct session ID as an OTel attribute.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


# ─── Core data structures ─────────────────────────────────────────────────────


@dataclasses.dataclass
class TurnContext:
    """Input to a single agent turn passed through the middleware chain."""

    user_input: str
    session_id: str
    model: str
    tool_mode: str
    # Framework-specific extras (e.g. LangChain executor ref, console handle).
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TurnResult:
    """Output of a single agent turn collected by the middleware chain."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    llm_calls: int = 0
    error: str | None = None
    duration_ms: float = 0.0
    # Filled in by observability middleware for downstream layers.
    otel_trace_id: str | None = None
    mlflow_run_id: str | None = None
    langfuse_trace_id: str | None = None


# ─── Protocols ────────────────────────────────────────────────────────────────


@runtime_checkable
class TurnRunner(Protocol):
    """A framework adapter that executes a single agent turn.

    Implementations:
        - LangChainRunner (frameworks/langchain/)
        - LiteLLMRunner   (frameworks/litellm_agent.py → thin wrapper)
        - SmolagentsRunner
        - LlamaIndexRunner
    """

    def run_turn(self, ctx: TurnContext) -> TurnResult:
        """Execute one turn synchronously and return a TurnResult."""
        ...

    def rebuild(
        self,
        *,
        model: str | None = None,
        tool_mode: str | None = None,
    ) -> None:
        """Hot-swap model or tool mode without losing session history."""
        ...


@runtime_checkable
class TurnMiddleware(Protocol):
    """A single cross-cutting concern that wraps ``TurnRunner.run_turn``.

    ``__call__`` receives the context and a ``next`` callable that invokes the
    remainder of the chain.  Middleware MUST call ``next(ctx)`` and return the
    result (possibly modified).
    """

    def __call__(self, ctx: TurnContext, next: Callable[[TurnContext], TurnResult]) -> TurnResult:
        ...


# ─── Chain builder ───────────────────────────────────────────────────────────


def build_chain(
    runner: TurnRunner,
    middleware: list[TurnMiddleware],
) -> Callable[[TurnContext], TurnResult]:
    """Compose middleware around a runner into a single callable.

    Middleware list is applied outermost-first (index 0 wraps everything).

    Example::

        chain = build_chain(runner, [UsageMiddleware(), OTelMiddleware(...)])
        result = chain(TurnContext(user_input="list runtimes", ...))
    """

    def _run(ctx: TurnContext) -> TurnResult:
        t0 = time.monotonic()
        result = runner.run_turn(ctx)
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    chain: Callable[[TurnContext], TurnResult] = _run
    for mw in reversed(middleware):
        _outer = mw
        _inner = chain

        def _wrap(ctx: TurnContext, *, _mw=_outer, _next=_inner) -> TurnResult:
            return _mw(ctx, _next)

        chain = _wrap

    return chain
