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

"""Shared types for kubeflow-mcp tools."""

import traceback
from typing import Any, Literal

from pydantic import BaseModel


def exception_details(exc: Exception) -> dict[str, Any]:
    """Extract structured details from an exception for ToolError.details.

    Includes the cause chain so downstream error handlers (e.g. the Ollama
    agent's _format_friendly_error) can surface HTTP status codes that the SDK
    wraps inside a generic message.
    """
    details: dict[str, Any] = {"exception": type(exc).__name__, "message": str(exc)}
    cause = exc.__cause__ or exc.__context__
    if cause:
        details["cause"] = f"{type(cause).__name__}: {cause}"
    # Include traceback for SDK errors that bury K8s HTTP details
    tb = traceback.format_exc()
    if tb and tb.strip() != "NoneType: None":
        details["traceback"] = tb
    return details


class ToolResponse(BaseModel):
    """Standard success response."""

    success: Literal[True] = True
    data: dict[str, Any]


class ToolError(BaseModel):
    """Standard error response."""

    success: Literal[False] = False
    error: str
    error_code: str | None = None
    details: dict[str, Any] | None = None
    hint: str | None = None  # Suggest relevant MCP prompt for recovery


class PreviewResponse(BaseModel):
    """Response for two-phase confirmation pattern."""

    status: Literal["preview"] = "preview"
    message: str = "Set confirmed=True to execute"
    config: dict[str, Any]


ToolResult = ToolResponse | ToolError | PreviewResponse
