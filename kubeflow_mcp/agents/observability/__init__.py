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

"""Framework-agnostic MLflow observability layer for Kubeflow MCP agents.

All public symbols are safe to import unconditionally — MLflow is only
accessed at runtime when ``MLFLOW_TRACKING_URI`` is set and the ``mlflow``
package is installed.
"""

from ._context import trace_mode, trace_text, trim_preview, update_trace_context
from ._otel import (
    _SESSION_ID_VAR,
    agent_turn_span,
    get_tracer,
    setup_otel_tracer,
    tool_call_span,
)
from ._session import MlflowSessionLogger
from ._spans import invoke_with_mlflow_span

__all__ = [
    "MlflowSessionLogger",
    "invoke_with_mlflow_span",
    "setup_otel_tracer",
    "get_tracer",
    "agent_turn_span",
    "tool_call_span",
    "_SESSION_ID_VAR",
    "trace_mode",
    "trace_text",
    "trim_preview",
    "update_trace_context",
]
