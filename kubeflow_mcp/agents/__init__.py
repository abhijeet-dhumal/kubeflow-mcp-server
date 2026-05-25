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

"""Pluggable CLI agents for Kubeflow MCP.

Heavy providers (Ollama, LiteLLM) are loaded lazily via :func:`__getattr__`
so ``import kubeflow_mcp.agents`` does not require optional dependencies.
"""

from kubeflow_mcp.agents.base import AgentProvider

__all__ = [
    "AgentProvider",
    "LiteLLMAgent",
    "LiteLLMProvider",
    "LoopState",
    "build_tool_schema",
]


def __getattr__(name: str):
    if name == "LiteLLMProvider":
        from kubeflow_mcp.agents.litellm_provider import LiteLLMProvider as _LiteLLMProvider

        return _LiteLLMProvider
    if name == "LiteLLMAgent":
        from kubeflow_mcp.agents.litellm_agent import LiteLLMAgent as _LiteLLMAgent

        return _LiteLLMAgent
    if name == "LoopState":
        from kubeflow_mcp.agents.litellm_agent import LoopState as _LoopState

        return _LoopState
    if name == "build_tool_schema":
        from kubeflow_mcp.agents.core.schema import build_tool_schema as _build_tool_schema

        return _build_tool_schema
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
