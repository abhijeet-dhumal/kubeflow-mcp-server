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

"""Framework-agnostic MLflow session logger for agent REPLs."""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ._context import trace_mode, trace_text, trim_preview, update_trace_context
from ._otel import _SESSION_ID_VAR, agent_turn_span


def _sanitize_mlflow_metric_key(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip().lower()).strip("_.-")
    return cleaned or "unknown"


def _session_messages_from_transcript(
    transcript: list[dict[str, Any]],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in transcript:
        user = turn.get("user")
        assistant = turn.get("assistant")
        if isinstance(user, str) and user:
            messages.append({"role": "user", "content": user})
        if isinstance(assistant, str) and assistant:
            messages.append({"role": "assistant", "content": assistant})
    return messages


class MlflowSessionLogger:
    """Best-effort MLflow session logger usable by all agent framework adapters.

    Usage::

        logger = MlflowSessionLogger(model="ollama/qwen3:8b", tool_mode="full", framework="smolagents")
        try:
            # inside REPL loop:
            logger.log_turn(user_input=line, assistant_output=answer, ...)
        finally:
            logger.close()
    """

    def __init__(self, *, model: str, tool_mode: str, framework: str) -> None:
        self.enabled = False
        self.run_id: str | None = None
        self.session_id = (
            os.environ.get("MLFLOW_SESSION_ID") or f"session-{uuid4().hex[:12]}"
        )
        self.framework = framework
        self._owns_run = False
        self._step = 0
        self._session_started_at = datetime.now(timezone.utc).isoformat()
        self._session_tool_calls = 0
        self._session_llm_calls = 0
        self._session_duration_s = 0.0
        self._trace_user = (
            os.environ.get("MLFLOW_TRACE_USER") or os.environ.get("USER") or "unknown"
        )
        self._trace_mode = trace_mode()
        self._transcript: list[dict[str, Any]] = []
        self._mlflow = None

        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        try:
            import mlflow
        except ImportError:
            return

        self._mlflow = mlflow
        # Ensure the tracking URI is set explicitly — env var alone may not
        # propagate to the GenAI tracing subsystem which initialises lazily.
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        try:
            exp_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
            exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
            if exp_id:
                mlflow.set_experiment(experiment_id=exp_id)
            elif exp_name:
                mlflow.set_experiment(experiment_name=exp_name)

            # Enable automatic LangChain tracing (MLflow 3.x GenAI Traces tab).
            # Only for langchain; other frameworks use manual start_span() below.
            if framework == "langchain":
                try:
                    mlflow.langchain.autolog(log_traces=True, disable=False)
                except Exception:
                    pass

            active = mlflow.active_run()
            if active is None:
                run = mlflow.start_run(run_name=f"kubeflow-mcp-{framework}-repl")
                self._owns_run = True
                self.run_id = run.info.run_id
            else:
                self.run_id = active.info.run_id

            mlflow.set_tags(
                {
                    "agent.framework": framework,
                    "agent.model": model,
                    "agent.tool_mode": tool_mode,
                    "agent.session.started_at": self._session_started_at,
                    "agent.session_id": self.session_id,
                    "agent.trace_mode": self._trace_mode,
                }
            )
            mlflow.log_params(
                {
                    "agent.framework": framework,
                    "agent.model": model,
                    "agent.tool_mode": tool_mode,
                    "agent.session_id": self.session_id,
                    "agent.trace_mode": self._trace_mode,
                }
            )
            os.environ.setdefault("KUBEFLOW_MCP_SESSION_ID", self.session_id)
            self.enabled = True
            self._log_session_start_trace()
        except Exception:
            self.enabled = False

    def _log_session_start_trace(self) -> None:
        if not self.enabled or self._mlflow is None:
            return
        try:
            from mlflow.entities import SpanType

            with self._mlflow.start_span(
                name="chat.session.start",
                span_type=SpanType.CHAT_MODEL,
                attributes={
                    "agent.framework": self.framework,
                    "agent.session_id": self.session_id,
                    "mlflow.traceSessionId": self.session_id,
                    "agent.trace_mode": self._trace_mode,
                },
            ) as span:
                span.set_inputs(
                    {
                        "session": {
                            "id": self.session_id,
                            "model": os.environ.get("KUBEFLOW_MCP_MODEL", ""),
                            "started_at": self._session_started_at,
                        }
                    }
                )
                span.set_outputs({"status": "started"})
                update_trace_context(
                    session_id=self.session_id,
                    framework=self.framework,
                    request_preview="session_start",
                    response_preview="session_started",
                )
        except Exception:
            return

    def log_turn(  # noqa: C901
        self,
        *,
        user_input: str,
        assistant_output: str,
        tool_names: list[str] | None = None,
        tool_call_count: int = 0,
        llm_call_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_s: float = 0.0,
        otel_span: Any | None = None,
    ) -> None:
        """Log one completed REPL turn to MLflow.

        Pass ``otel_span`` (the active OpenTelemetry span from ``OTelMiddleware``)
        to annotate the span with turn-level stats.
        """
        # Ensure session_id is propagated for tool spans in the next turn.
        _SESSION_ID_VAR.set(self.session_id)

        if otel_span is not None:
            try:
                otel_span.set_attribute("agent.tool_call_count", tool_call_count)
                otel_span.set_attribute("agent.llm_call_count", llm_call_count)
                otel_span.set_attribute("agent.duration_ms", round(duration_s * 1000))
            except Exception:
                pass

        self._log_turn_inner(
            user_input=user_input,
            assistant_output=assistant_output,
            tool_names=tool_names,
            tool_call_count=tool_call_count,
            llm_call_count=llm_call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_s=duration_s,
        )

    def _log_turn_inner(  # noqa: C901
        self,
        *,
        user_input: str,
        assistant_output: str,
        tool_names: list[str] | None = None,
        tool_call_count: int = 0,
        llm_call_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_s: float = 0.0,
    ) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._step += 1
        self._session_tool_calls += tool_call_count
        self._session_llm_calls += llm_call_count
        self._session_duration_s += duration_s

        _tool_names = tool_names or []
        traced_user = trace_text(user_input)
        traced_assistant = trace_text(assistant_output)

        self._transcript.append(
            {
                "turn": self._step,
                "user": traced_user,
                "assistant": traced_assistant,
                "tools": _tool_names,
                "tokens": {"input": input_tokens, "output": output_tokens},
                "duration_s": round(duration_s, 3),
            }
        )
        try:
            metrics: dict[str, float] = {
                "agent.turn.llm_calls": float(llm_call_count),
                "agent.turn.tool_calls": float(tool_call_count),
                "agent.turn.duration_s": float(duration_s),
                "agent.turn.input_tokens": float(input_tokens),
                "agent.turn.output_tokens": float(output_tokens),
            }
            for tool_name, count in Counter(_tool_names).items():
                key = _sanitize_mlflow_metric_key(tool_name)
                metrics[f"agent.turn.tool.{key}"] = float(count)
            self._mlflow.log_metrics(metrics, step=self._step)
            self._mlflow.log_dict(
                {
                    "turn": self._step,
                    "user_input": traced_user,
                    "assistant_preview": traced_assistant[:400],
                    "session_id": self.session_id,
                    "tool_calls": _tool_names,
                    "tool_call_count": tool_call_count,
                    "llm_calls": llm_call_count,
                    "duration_s": round(duration_s, 3),
                    "tokens": {"input": input_tokens, "output": output_tokens},
                },
                artifact_file=f"agent_turns/turn-{self._step:04d}.json",
            )
            try:
                from mlflow.entities import SpanType

                with self._mlflow.start_span(
                    name=f"chat.turn.{self._step:04d}",
                    span_type=SpanType.CHAT_MODEL,
                    attributes={
                        "agent.framework": self.framework,
                        "agent.session_id": self.session_id,
                        "mlflow.traceSessionId": self.session_id,
                        "agent.turn": str(self._step),
                        "agent.trace_mode": self._trace_mode,
                    },
                ) as span:
                    span.set_inputs(
                        {"messages": [{"role": "user", "content": traced_user}]}
                    )
                    span.set_outputs(
                        {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": trim_preview(
                                        traced_assistant, max_chars=4000
                                    ),
                                }
                            ],
                            "tool_calls": _tool_names,
                            "tool_call_count": tool_call_count,
                        }
                    )
                    update_trace_context(
                        session_id=self.session_id,
                        framework=self.framework,
                        request_preview=trim_preview(traced_user),
                        response_preview=trim_preview(traced_assistant),
                    )
            except Exception:
                pass
        except Exception:
            return

    def close(self) -> None:
        if not self.enabled or self._mlflow is None:
            return
        traced_messages = _session_messages_from_transcript(self._transcript)
        try:
            self._mlflow.log_metrics(
                {
                    "agent.session.turns": float(self._step),
                    "agent.session.tool_calls": float(self._session_tool_calls),
                    "agent.session.llm_calls": float(self._session_llm_calls),
                    "agent.session.duration_s": float(self._session_duration_s),
                }
            )
            self._mlflow.log_dict(
                {
                    "turns": self._step,
                    "tool_calls": self._session_tool_calls,
                    "llm_calls": self._session_llm_calls,
                    "duration_s": round(self._session_duration_s, 3),
                    "started_at": self._session_started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "session_id": self.session_id,
                },
                artifact_file="agent_session_summary.json",
            )
            self._mlflow.log_dict(
                {
                    "session_id": self.session_id,
                    "user": self._trace_user,
                    "started_at": self._session_started_at,
                    "messages": traced_messages,
                    "turns": self._transcript,
                },
                artifact_file="chat_session.json",
            )
            try:
                from mlflow.entities import SpanType

                with self._mlflow.start_span(
                    name="chat.session.snapshot",
                    span_type=SpanType.CHAT_MODEL,
                    attributes={
                        "agent.framework": self.framework,
                        "agent.session_id": self.session_id,
                        "mlflow.traceSessionId": self.session_id,
                        "agent.trace_mode": self._trace_mode,
                    },
                ) as span:
                    span.set_inputs(
                        {
                            "session": {
                                "id": self.session_id,
                                "started_at": self._session_started_at,
                                "turn_count": len(self._transcript),
                            }
                        }
                    )
                    span.set_outputs({"messages": traced_messages})
                    update_trace_context(
                        session_id=self.session_id,
                        framework=self.framework,
                        request_preview="session_snapshot",
                        response_preview=f"session_snapshot:{len(traced_messages)}_messages",
                    )
            except Exception:
                pass
        except Exception:
            pass

        if not self._owns_run:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            return
