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

"""LiteLLM provider — pluggable CLI agent backend for kubeflow-mcp.

Registered as the ``litellm`` entry-point in ``pyproject.toml``::

    [project.entry-points."kubeflow_mcp.providers"]
    litellm = "kubeflow_mcp.agents.litellm_provider:LiteLLMProvider"

CLI invocation (via ``kubeflow-mcp agent --provider litellm``)::

    # Local Ollama
    kubeflow-mcp agent --provider litellm --model ollama/gemma4:e4b

    # On-prem LiteLLM Proxy
    kubeflow-mcp agent --provider litellm --model openai/my-model \\
        --base-url http://litellm-proxy.svc:4000

    # Cloud
    kubeflow-mcp agent --provider litellm --model gpt-4.1
"""

from __future__ import annotations

import os
import warnings
from typing import Any


class LiteLLMProvider:
    """Provider that delegates to :class:`~kubeflow_mcp.agents.litellm_repl.run_litellm_chat`.

    Attributes:
        name: Identifies this provider in the entry-point registry.
        default_model: Used when the caller passes no ``--model`` flag.
        requires: Python packages that must be importable at runtime.
    """

    name = "litellm"
    default_model = "ollama/gemma4:e4b"
    requires = ["litellm", "rich"]

    def run(
        self,
        model: str,
        mode: str = "full",
        framework: str = "langchain",
        thinking: bool = True,
        base_url: str | None = None,
        fallback_model: str | None = None,
        num_retries: int = 3,
        cache: bool = False,
        langfuse: bool = False,
        mlflow_uri: str | None = None,
        otel_endpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start the interactive REPL.

        Args:
            model: LiteLLM model string. Use ``ollama/<tag>`` for local Ollama,
                ``openai/<name>`` for OpenAI-compatible proxies, or bare provider
                names like ``gpt-4.1``.
            mode: Tool scope — ``"full"`` (all tools), ``"progressive"`` (meta-tools
                with category listing), or ``"semantic"`` (natural-language tool search).
            framework: Agent loop backend — ``langchain`` (default ReAct),
                ``smolagents``, ``llamaindex``, or ``litellm`` (legacy native loop).
            base_url: Override the LiteLLM base URL for on-prem endpoints or a
                self-hosted LiteLLM Proxy.
            fallback_model: Secondary model attempted when the primary fails.
            num_retries: LiteLLM built-in retry count (default 3).
            cache: Enable LiteLLM semantic caching (requires Redis or in-memory
                cache configured via env vars).
            langfuse: Enable Langfuse callback for LLM-level tracing.
            mlflow_uri: If set, enable MLflow callback and point to this URI.
        """
        self._configure_observability(
            langfuse=langfuse,
            mlflow_uri=mlflow_uri,
            otel_endpoint=otel_endpoint,
        )
        # Expose model to _session.py OTel spans via env var
        os.environ["KUBEFLOW_MCP_MODEL"] = model

        # Disable SSL verification for custom base_url endpoints (RHOAI/OpenShift
        # clusters use internal CAs not trusted by the system keychain).
        # Controlled by KUBEFLOW_MCP_SSL_VERIFY=true to re-enable.
        if base_url and os.environ.get("KUBEFLOW_MCP_SSL_VERIFY", "false").lower() not in ("1", "true", "yes"):
            os.environ.setdefault("CURL_CA_BUNDLE", "")
            os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
            try:
                import litellm as _ll
                _ll.ssl_verify = False
            except ImportError:
                pass
        if cache:
            self._configure_cache()

        # Resolve base_url from env when not passed explicitly
        if base_url is None:
            base_url = os.environ.get("LITELLM_BASE_URL") or None

        _skip = {"thinking", "framework", "cache", "langfuse", "mlflow_uri", "otel_endpoint", "url"}
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "tool_mode": mode,
            "base_url": base_url,
            "langfuse": langfuse,
            "thinking": thinking,
            "num_retries": num_retries,
            **{k: v for k, v in kwargs.items() if k not in _skip},
        }

        if framework == "langchain":
            try:
                from kubeflow_mcp.agents.frameworks.langchain_agent import run_langchain_chat
            except ImportError as exc:
                msg = "Install optional deps: uv sync --extra agents-langchain"
                raise RuntimeError(msg) from exc
            run_langchain_chat(**chat_kwargs)
            return

        if framework == "smolagents":
            try:
                from kubeflow_mcp.agents.frameworks.smolagents_agent import run_smolagents_chat
            except ImportError as exc:
                msg = "Install optional deps: uv sync --extra agents-smolagents"
                raise RuntimeError(msg) from exc
            run_smolagents_chat(**chat_kwargs)
            return

        if framework == "llamaindex":
            try:
                from kubeflow_mcp.agents.frameworks.llamaindex_agent import run_llamaindex_chat
            except ImportError as exc:
                msg = "Install optional deps: uv sync --extra agents-llamaindex"
                raise RuntimeError(msg) from exc
            run_llamaindex_chat(**chat_kwargs)
            return

        if framework != "litellm":
            msg = f"Unknown framework {framework!r}"
            raise ValueError(msg)

        warnings.warn(
            "framework='litellm' is the legacy native loop — prefer --framework langchain",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            from kubeflow_mcp.agents.litellm_repl import run_litellm_chat
        except ImportError as exc:
            msg = (
                "Install optional deps: uv sync --extra agents-litellm  "
                "(or --extra agents for all backends)"
            )
            raise RuntimeError(msg) from exc

        run_litellm_chat(
            model=model,
            tool_mode=mode,
            base_url=base_url,
            fallback_model=fallback_model,
            thinking=thinking,
            num_retries=num_retries,
        )

    # ─── Observability wiring ─────────────────────────────────────────────────

    @staticmethod
    def _configure_observability(
        *,
        langfuse: bool = False,
        mlflow_uri: str | None = None,
        otel_endpoint: str | None = None,
    ) -> None:
        """Attach LiteLLM callbacks and configure OTel tracer."""
        # OTel — must run before LiteLLM callbacks so the tracer is ready.
        from kubeflow_mcp.agents.observability._otel import setup_otel_tracer

        setup_otel_tracer(endpoint=otel_endpoint)

        try:
            import litellm
        except ImportError:
            return

        callbacks: list[str] = []

        if langfuse or os.environ.get("LANGFUSE_SECRET_KEY"):
            callbacks.append("langfuse")

        if mlflow_uri:
            os.environ.setdefault("MLFLOW_TRACKING_URI", mlflow_uri)
            callbacks.append("mlflow")
        elif os.environ.get("MLFLOW_TRACKING_URI"):
            callbacks.append("mlflow")

        if callbacks:
            litellm.success_callback = callbacks
            litellm.failure_callback = callbacks

    @staticmethod
    def _configure_cache() -> None:
        """Enable LiteLLM in-memory semantic cache (Redis if env configured)."""
        try:
            import litellm

            redis_url = os.environ.get("REDIS_URL")
            if redis_url:
                litellm.cache = litellm.Cache(type="redis", url=redis_url)
            else:
                litellm.cache = litellm.Cache(type="local")
        except Exception:
            pass  # Cache is best-effort; never block startup
