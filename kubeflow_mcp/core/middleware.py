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

"""FastMCP server middleware (Gap 6B + 6C).

Three concerns are handled here:

1. **AuditIdentityMiddleware** — bridges FastMCP's async middleware context into
   the synchronous ``_audit_wrap`` via ``contextvars.ContextVar``s.  This ensures
   each audit log entry carries ``user_id`` and ``mcp_session_id`` and that the
   active OTel span gets those attributes too.

2. **FastMCP built-in middleware** — helper that registers the
   ``ErrorHandlingMiddleware``, ``TimingMiddleware``, ``RateLimitingMiddleware``
   and ``ResponseLimitingMiddleware`` that ship with FastMCP 2.x / 3.x, replacing
   the custom ``RateLimiter`` in ``resilience.py`` over time (Gap 6C).

3. **OTel span attribute injection** — ``AuditIdentityMiddleware`` sets
   ``user_id`` and ``mcp_session_id`` on the currently active OTel span so that
   the server-side ``agent.turn`` span carries identity without requiring the
   caller to pass them explicitly through every layer.

Usage (in ``create_server`` after Gap 6C lands)::

    from kubeflow_mcp.core.middleware import (
        AuditIdentityMiddleware,
        register_fastmcp_middleware,
    )
    register_fastmcp_middleware(mcp, cfg)
    mcp.add_middleware(AuditIdentityMiddleware())
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Identity ContextVars ──────────────────────────────────────────────────────
# Set by AuditIdentityMiddleware before calling call_next; read by
# _audit_wrap (sync) without needing to thread the context explicitly.

_AUDIT_USER_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "audit_user_id", default="anonymous"
)
_AUDIT_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "audit_mcp_session_id", default="unknown"
)


# ── AuditIdentityMiddleware ───────────────────────────────────────────────────


class AuditIdentityMiddleware:
    """Propagate FastMCP client identity into audit logs and OTel spans.

    FastMCP's middleware receives async context including ``client_id`` and
    ``session_id``.  Because ``_audit_wrap`` is synchronous, those values are
    written into module-level ``ContextVar``s before the tool executes, then
    read back inside ``_audit_wrap`` after the tool call returns.

    The middleware also annotates the *currently active OTel span* (opened by
    ``_audit_wrap`` via ``core/telemetry.py``) with identity attributes so
    traces are user-correlated in Jaeger without a separate lookup.

    This class follows FastMCP's middleware protocol: ``async def __call__(self,
    context, call_next)``.
    """

    async def __call__(self, context: Any, call_next: Any) -> Any:
        user_id = "anonymous"
        session_id = "unknown"

        fctx = getattr(context, "fastmcp_context", None)
        if fctx is not None:
            raw_client = getattr(fctx, "client_id", None)
            raw_session = getattr(fctx, "session_id", None)
            if raw_client:
                user_id = str(raw_client)
            if raw_session:
                session_id = str(raw_session)

        token_user = _AUDIT_USER_ID.set(user_id)
        token_session = _AUDIT_SESSION_ID.set(session_id)

        # Annotate the active OTel span (if any) with identity.
        try:
            from opentelemetry import trace as _trace

            span = _trace.get_current_span()
            if span.is_recording():
                span.set_attribute("user.id", user_id)
                span.set_attribute("mcp.session_id", session_id)
        except Exception:
            pass

        try:
            return await call_next(context)
        finally:
            _AUDIT_USER_ID.reset(token_user)
            _AUDIT_SESSION_ID.reset(token_session)


# ── FastMCP built-in middleware registration (Gap 6C) ─────────────────────────


def register_fastmcp_middleware(mcp: Any, cfg: Any) -> None:
    """Register FastMCP's built-in middleware stack on *mcp*.

    Registers (in outermost-first order):
      - ErrorHandlingMiddleware: structured JSON errors instead of raw tracebacks.
      - TimingMiddleware: adds ``X-Duration-Ms`` to responses and logs slow calls.
      - ResponseLimitingMiddleware: caps response body size (default 1 MB).
      - AuditIdentityMiddleware: bridges identity into ``_audit_wrap`` + OTel.

    ``RateLimitingMiddleware`` is intentionally NOT registered here because the
    existing ``RateLimiter`` in ``resilience.py`` already handles token-bucket
    rate limiting inside ``_audit_wrap``.  Both would cause double-limiting.
    Gap 6C will migrate to FastMCP's built-in implementation once the custom
    ``RateLimiter`` is retired.

    Args:
        mcp: FastMCP server instance.
        cfg: ``Config`` object from ``core/config.py``.
    """
    try:
        from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

        mcp.add_middleware(ErrorHandlingMiddleware())
        logger.debug("registered ErrorHandlingMiddleware")
    except ImportError:
        logger.debug("ErrorHandlingMiddleware not available in this FastMCP version")

    try:
        from fastmcp.server.middleware.timing import TimingMiddleware

        mcp.add_middleware(TimingMiddleware())
        logger.debug("registered TimingMiddleware")
    except ImportError:
        logger.debug("TimingMiddleware not available in this FastMCP version")

    try:
        from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware

        mcp.add_middleware(ResponseLimitingMiddleware(max_size=1_000_000))
        logger.debug("registered ResponseLimitingMiddleware(max_size=1MB)")
    except ImportError:
        logger.debug("ResponseLimitingMiddleware not available in this FastMCP version")

    mcp.add_middleware(AuditIdentityMiddleware())
    logger.debug("registered AuditIdentityMiddleware")
