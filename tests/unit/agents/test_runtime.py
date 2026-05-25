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

"""Unit tests for the pluggable runtime contracts (Gap 3B) and middleware (Gap 3C)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from kubeflow_mcp.agents.runtime.contracts import (
    TurnContext,
    TurnResult,
    build_chain,
)
from kubeflow_mcp.agents.observability.middleware import UsageMiddleware


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ctx(text: str = "hello") -> TurnContext:
    return TurnContext(
        user_input=text,
        session_id="test-session",
        model="ollama/qwen3:8b",
        tool_mode="full",
    )


class _EchoRunner:
    """Minimal TurnRunner that echoes the input as output."""

    def run_turn(self, ctx: TurnContext) -> TurnResult:
        return TurnResult(text=f"echo: {ctx.user_input}")

    def rebuild(self, *, model=None, tool_mode=None) -> None:
        pass


# ── build_chain ───────────────────────────────────────────────────────────────


def test_build_chain_no_middleware():
    runner = _EchoRunner()
    chain = build_chain(runner, [])
    result = chain(_make_ctx("ping"))
    assert result.text == "echo: ping"


def test_build_chain_middleware_order():
    """Verify outermost middleware wraps inner ones (order index 0 is outermost)."""
    log: list[str] = []

    class _LogMW:
        def __init__(self, tag: str) -> None:
            self._tag = tag

        def __call__(self, ctx, next):
            log.append(f"enter:{self._tag}")
            r = next(ctx)
            log.append(f"exit:{self._tag}")
            return r

    runner = _EchoRunner()
    chain = build_chain(runner, [_LogMW("A"), _LogMW("B")])
    chain(_make_ctx())

    assert log == ["enter:A", "enter:B", "exit:B", "exit:A"]


# ── UsageMiddleware ───────────────────────────────────────────────────────────


def test_usage_middleware_accumulates():
    class _CostRunner:
        def run_turn(self, ctx: TurnContext) -> TurnResult:
            return TurnResult(
                text="ok",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_cost": 0.001},
            )

        def rebuild(self, **_: Any) -> None:
            pass

    mw = UsageMiddleware()
    chain = build_chain(_CostRunner(), [mw])
    chain(_make_ctx())
    chain(_make_ctx())

    assert mw.totals["prompt_tokens"] == 20.0
    assert mw.totals["completion_tokens"] == 10.0
    assert mw.totals["turns"] == 2.0


# ── schema helpers ────────────────────────────────────────────────────────────


def test_build_tool_schema_basic():
    from kubeflow_mcp.agents.core.schema import build_tool_schema

    def my_tool(name: str, count: int = 1) -> dict:
        """Return something."""
        ...

    schema = build_tool_schema(my_tool)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "my_tool"
    assert "name" in fn["parameters"]["required"]
    assert "count" not in fn["parameters"].get("required", [])
    assert fn["parameters"]["properties"]["name"]["type"] == "string"
    assert fn["parameters"]["properties"]["count"]["type"] == "integer"


# ── confirm shim backward compat ──────────────────────────────────────────────


def test_confirm_shim_imports():
    """Ensure frameworks/_confirm.py shim re-exports correctly."""
    from kubeflow_mcp.agents.frameworks._confirm import set_confirm_handler, wrap_with_confirm

    assert callable(set_confirm_handler)
    assert callable(wrap_with_confirm)


def test_tools_shim_imports():
    """Ensure frameworks/_tools.py shim re-exports correctly."""
    from kubeflow_mcp.agents.frameworks._tools import audit_wrap, get_system_prompt, load_tools

    assert callable(audit_wrap)
    assert callable(get_system_prompt)
    assert callable(load_tools)
