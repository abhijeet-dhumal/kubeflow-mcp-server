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

"""Extended thinking / reasoning mode for LiteLLM-backed framework agents."""

from __future__ import annotations

from typing import Any

from kubeflow_mcp.agents.frameworks._observability import is_local_ollama_model


def extract_thinking_delta(token: Any, chunk: Any = None) -> str | None:
    """Pull a reasoning/thinking text delta from a LangChain streaming callback."""
    message = getattr(chunk, "message", None)
    if message is not None:
        additional = getattr(message, "additional_kwargs", None) or {}
        rc = additional.get("reasoning_content")
        if rc:
            return str(rc)

    if isinstance(token, list):
        parts: list[str] = []
        for item in token:
            if isinstance(item, dict) and item.get("type") == "thinking":
                parts.append(str(item.get("thinking") or ""))
        joined = "".join(parts)
        return joined or None

    return None


def is_answer_content_token(token: Any, chunk: Any = None) -> bool:
    """True when a stream chunk carries normal answer text (not reasoning)."""
    if extract_thinking_delta(token, chunk):
        return False
    if isinstance(token, str) and token:
        return True
    message = getattr(chunk, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return isinstance(content, str) and bool(content)


def thinking_completion_kwargs(*, enabled: bool, model: str) -> dict[str, Any]:
    """Extra kwargs for litellm.completion and LiteLLM wrappers."""
    if not enabled:
        return {}
    # Ollama uses ``think``; Anthropic/OpenAI-style providers use ``thinking``.
    if is_local_ollama_model(model):
        return {"think": True}
    return {"thinking": {"type": "enabled", "budget_tokens": 8192}}


def apply_thinking_to_chat_litellm(llm_kwargs: dict[str, Any], *, enabled: bool, model: str) -> None:
    """Merge thinking kwargs into ChatLiteLLM model_kwargs."""
    extra = thinking_completion_kwargs(enabled=enabled, model=model)
    if extra:
        llm_kwargs.setdefault("model_kwargs", {}).update(extra)


def apply_thinking_to_litellm_model(model_kwargs: dict[str, Any], *, enabled: bool, model: str) -> None:
    """Merge thinking kwargs into smolagents LiteLLMModel kwargs."""
    model_kwargs.update(thinking_completion_kwargs(enabled=enabled, model=model))


def apply_thinking_to_llamaindex(llm_kwargs: dict[str, Any], *, enabled: bool, model: str) -> None:
    """Merge thinking kwargs into llama-index LiteLLM additional_kwargs."""
    extra = thinking_completion_kwargs(enabled=enabled, model=model)
    if extra:
        llm_kwargs.setdefault("additional_kwargs", {}).update(extra)
        llm_kwargs.update(extra)
