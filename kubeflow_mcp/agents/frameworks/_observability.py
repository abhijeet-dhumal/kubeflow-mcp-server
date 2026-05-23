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

"""Shared observability helpers for framework adapters."""

from __future__ import annotations

import os


def setup_langsmith(*, langfuse: bool = False) -> str | None:
    """Enable LangSmith when CLI --langfuse or env is set."""
    if langfuse:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
    api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
    if not tracing:
        return None
    if not api_key:
        return "LangSmith tracing on (set LANGCHAIN_API_KEY)"
    project = (
        os.environ.get("LANGCHAIN_PROJECT")
        or os.environ.get("LANGSMITH_PROJECT")
        or "default"
    )
    endpoint = (
        os.environ.get("LANGCHAIN_ENDPOINT")
        or os.environ.get("LANGSMITH_ENDPOINT")
        or "https://smith.langchain.com"
    )
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    return f"LangSmith → {project} @ {host}"


def is_local_ollama_model(model: str) -> bool:
    lower = model.lower()
    return lower.startswith("ollama/") or lower.startswith("ollama_chat/")
