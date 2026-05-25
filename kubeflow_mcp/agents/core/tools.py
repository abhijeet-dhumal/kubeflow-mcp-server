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

"""Framework-agnostic tool loading, system prompt, and audit wrapping.

Replaces frameworks/_tools.py which had a backward dependency on litellm_agent.
All tool-loading logic now lives here; litellm_agent imports from this module.

HTTP mode: when KUBEFLOW_MCP_HTTP_URL is set, tools are loaded from the remote
MCP server instead of in-process (Gap 4 — placeholder; in-process path active).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

VALID_MODES: frozenset[str] = frozenset({"full", "progressive", "semantic"})


# ── In-process tool loading ───────────────────────────────────────────────────


def _get_full_mode_tools() -> tuple[list[Callable[..., Any]], dict[str, str]]:
    """All trainer + health tools with descriptions (full mode)."""
    from kubeflow_mcp.common.tool_metadata import ALL_TOOL_DESCRIPTIONS
    from kubeflow_mcp.core.health import HEALTH_TOOLS

    try:
        from kubeflow_mcp.trainer import TOOLS as TRAINER_TOOLS
    except ImportError:
        TRAINER_TOOLS = []  # type: ignore[assignment]

    return list(TRAINER_TOOLS) + list(HEALTH_TOOLS), ALL_TOOL_DESCRIPTIONS


def _get_meta_mode_tools(mode: str) -> tuple[list[Callable[..., Any]], dict[str, str]]:
    """Meta-tools (progressive or semantic) routed via execute_tool / find_tools.

    Uses lazy registration so the 400ms kubeflow.trainer SDK import is deferred
    until the first execute_tool() call rather than at agent startup.
    """
    from kubeflow_mcp.common.tool_metadata import ALL_TOOL_DESCRIPTIONS
    from kubeflow_mcp.core.dynamic_tools import (
        PROGRESSIVE_TOOLS,
        SEMANTIC_TOOLS,
        TOOL_REGISTRY,
        init_dynamic_tools_lazy,
    )

    if not TOOL_REGISTRY:
        init_dynamic_tools_lazy(ALL_TOOL_DESCRIPTIONS)

    tools = PROGRESSIVE_TOOLS if mode == "progressive" else SEMANTIC_TOOLS
    descs = {fn.__name__: (fn.__doc__ or "").strip().split("\n")[0] for fn in tools}
    return list(tools), descs


def load_tools(tool_mode: str) -> tuple[list[Callable[..., Any]], dict[str, str]]:
    """Return (tool callables, descriptions) for the given mode.

    When ``KUBEFLOW_MCP_HTTP_URL`` is set the HTTP loader will be used (Gap 4).
    Falls back to in-process loading for stdio / local dev.
    """
    if os.environ.get("KUBEFLOW_MCP_HTTP_URL"):
        warnings.warn(
            "KUBEFLOW_MCP_HTTP_URL is set but HTTP tool loading (Gap 4) is not yet implemented. "
            "Falling back to in-process tool loading. "
            "See docs/design/production-gaps-lld.md Gap 4 for the roadmap.",
            stacklevel=2,
        )

    if tool_mode == "full":
        return _get_full_mode_tools()
    if tool_mode in ("progressive", "semantic"):
        return _get_meta_mode_tools(tool_mode)
    raise ValueError(f"Invalid tool_mode {tool_mode!r}. Choose: {', '.join(sorted(VALID_MODES))}")


# ── System prompt ─────────────────────────────────────────────────────────────


def get_system_prompt(
    *,
    persona: str = "readonly",
    instruction_tier: str = "full",
) -> str:
    """Return the server instruction text, or a minimal fallback."""
    try:
        from kubeflow_mcp.core.server import build_agent_instruction_text

        return build_agent_instruction_text(persona=persona, instruction_tier=instruction_tier)
    except ImportError:
        return "You are a Kubeflow training assistant. Help users manage ML training jobs."


# ── Audit wrapping ────────────────────────────────────────────────────────────


def audit_wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Apply server-side _audit_wrap (rate limit + circuit breaker + OTel span).

    Falls back to the raw function when the MCP server package is not installed
    (e.g. lightweight agent-only deployments).
    """
    try:
        from kubeflow_mcp.core.server import _audit_wrap

        return _audit_wrap(fn)
    except (ImportError, AttributeError):
        return fn
