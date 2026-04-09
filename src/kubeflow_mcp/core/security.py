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

"""Input validation and security checks."""

import ast
import re
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError

K8S_NAME_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
MAX_NAME_LENGTH = 63


def validate_k8s_name(name: str, field: str = "name") -> ToolError | None:
    """Validate Kubernetes resource name.

    Returns ToolError if invalid, None if valid.
    """
    if not name:
        return ToolError(
            error=f"{field} cannot be empty",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    if len(name) > MAX_NAME_LENGTH:
        return ToolError(
            error=f"{field} too long (max {MAX_NAME_LENGTH})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    if not K8S_NAME_PATTERN.match(name):
        return ToolError(
            error=f"{field} must be lowercase alphanumeric with hyphens",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"value": name, "pattern": K8S_NAME_PATTERN.pattern},
        )

    return None


def validate_namespace(namespace: str) -> ToolError | None:
    """Validate namespace name format."""
    return validate_k8s_name(namespace, "namespace")


def check_namespace_allowed(namespace: str | None) -> ToolError | None:
    """Check if namespace is allowed by policy.

    Args:
        namespace: Namespace to check (None uses default, always allowed)

    Returns:
        ToolError if namespace is restricted, None if allowed
    """
    if namespace is None:
        return None

    # Import here to avoid circular import
    from kubeflow_mcp.core.policy import get_allowed_namespaces

    allowed = get_allowed_namespaces()
    if allowed is None:
        # No restrictions
        return None

    if namespace not in allowed:
        return ToolError(
            error=f"Namespace '{namespace}' not allowed by policy",
            error_code=ErrorCode.PERMISSION_DENIED,
            details={"allowed_namespaces": allowed},
        )

    return None


def is_safe_python_code(code: str) -> tuple[bool, str]:
    """Heuristic scan for obviously risky patterns (not a sandbox).

    This does **not** execute or confine code; it only flags common dangerous
    constructs before submission. Do not treat a ``True`` result as security
    approval for running untrusted code.

    Returns:
        ``(is_safe, reason)`` where reason explains the first match or ``"OK"``.
    """
    dangerous_patterns = [
        "import os",
        "import subprocess",
        "import sys",
        "__import__",
        "eval(",
        "exec(",
        "open(",
        "compile(",
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return False, f"Dangerous pattern: {pattern}"

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for alias in getattr(node, "names", []):
                    name = alias.name if hasattr(alias, "name") else str(alias)
                    if name in ("os", "subprocess", "sys", "shutil"):
                        return False, f"Dangerous import: {name}"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, "OK"


def sanitize_log_output(output: str, max_length: int = 10000) -> str:
    """Sanitize and truncate log output."""
    if len(output) > max_length:
        output = output[:max_length] + f"\n... (truncated, {len(output)} total chars)"
    return output


def validate_resource_limits(
    cpu: str | None,
    memory: str | None,
    gpu: int | None,
) -> ToolError | None:
    """Validate resource limit specifications."""
    if cpu:
        if not re.match(r"^\d+m?$", cpu):
            return ToolError(
                error="Invalid CPU format (use '100m' or '1')",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

    if memory:
        if not re.match(r"^\d+(Ki|Mi|Gi|Ti)?$", memory):
            return ToolError(
                error="Invalid memory format (use '256Mi' or '1Gi')",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

    if gpu is not None and gpu < 0:
        return ToolError(
            error="GPU count cannot be negative",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    return None


def mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    """Mask sensitive fields in data for logging."""
    sensitive_keys = {"token", "password", "secret", "key", "credential"}
    result: dict[str, Any] = {}

    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys):
            result[k] = "***"
        elif isinstance(v, dict):
            result[k] = mask_sensitive_data(v)
        else:
            result[k] = v

    return result
