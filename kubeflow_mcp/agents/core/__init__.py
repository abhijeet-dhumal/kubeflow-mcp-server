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

"""Framework-agnostic agent core: tool loading, schema, confirm gate."""

from kubeflow_mcp.agents.core.confirm import (
    ConfirmMiddleware,
    ConfirmHandler,
    make_console_confirm_handler,
    set_confirm_handler,
    wrap_with_confirm,
)
from kubeflow_mcp.agents.core.schema import (
    _annotation_to_json_schema,
    build_tool_schema,
)
from kubeflow_mcp.agents.core.tools import (
    VALID_MODES,
    audit_wrap,
    get_system_prompt,
    load_tools,
)

__all__ = [
    # confirm
    "ConfirmHandler",
    "ConfirmMiddleware",
    "make_console_confirm_handler",
    "set_confirm_handler",
    "wrap_with_confirm",
    # schema
    "_annotation_to_json_schema",
    "build_tool_schema",
    # tools
    "VALID_MODES",
    "audit_wrap",
    "get_system_prompt",
    "load_tools",
]
