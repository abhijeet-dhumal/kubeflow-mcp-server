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

"""Shared tool loading for framework adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kubeflow_mcp.agents.litellm_agent import _get_full_mode_tools, _get_meta_mode_tools


def load_tools(tool_mode: str) -> tuple[list[Callable[..., Any]], dict[str, str]]:
    """Return (tool callables, descriptions) for the given mode."""
    if tool_mode == "full":
        return _get_full_mode_tools()
    if tool_mode in ("progressive", "semantic"):
        return _get_meta_mode_tools(tool_mode)
    msg = f"Invalid tool_mode {tool_mode!r}. Choose: full, progressive, semantic"
    raise ValueError(msg)


def get_system_prompt() -> str:
    try:
        from kubeflow_mcp.core.server import build_agent_instruction_text

        return build_agent_instruction_text()
    except ImportError:
        return "You are a Kubeflow training assistant. Help users manage ML training jobs."


def audit_wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    try:
        from kubeflow_mcp.core.server import _audit_wrap

        return _audit_wrap(fn)
    except (ImportError, AttributeError):
        return fn
