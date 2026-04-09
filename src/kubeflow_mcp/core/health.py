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

"""Health check tools for server monitoring."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.core.logging import get_log_buffer

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


@dataclass
class HealthManager:
    """Manages health check tools."""

    mcp: "FastMCP"
    _start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def register_health_tools(self) -> None:
        """Register health check tools with the MCP server."""

        @self.mcp.tool()
        def health_check() -> dict[str, Any]:
            """Check server health and connectivity.

            Returns server status, uptime, and K8s connectivity.
            Call periodically to verify server is operational.

            Returns:
                JSON with {status: str, uptime_seconds: int, kubernetes: bool}
            """
            return self._health_check()

        @self.mcp.tool()
        def get_server_logs(
            level: str = "INFO",
            limit: int = 100,
        ) -> dict[str, Any]:
            """Get recent server logs for debugging.

            Retrieves logs from in-memory buffer. Use with MCP-Analyzer
            for advanced log analysis.

            Args:
                level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
                limit: Maximum number of log entries to return

            Returns:
                JSON with {logs: [{timestamp, level, message}]}
            """
            return self._get_logs(level, limit)

    def _health_check(self) -> dict[str, Any]:
        """Internal health check implementation."""
        from kubeflow_mcp.common.utils import K8S_TIMEOUT, get_core_v1_api

        now = datetime.now(timezone.utc)
        uptime = (now - self._start_time).total_seconds()

        k8s_ok = False
        try:
            v1 = get_core_v1_api()
            v1.list_namespace(limit=1, _request_timeout=K8S_TIMEOUT)
            k8s_ok = True
        except Exception as e:
            logger.warning(f"K8s health check failed: {e}")

        return ToolResponse(
            data={
                "status": "healthy" if k8s_ok else "degraded",
                "uptime_seconds": int(uptime),
                "kubernetes": k8s_ok,
                "timestamp": now.isoformat(),
            }
        ).model_dump()

    def _get_logs(self, level: str, limit: int) -> dict[str, Any]:
        """Internal log retrieval implementation."""
        try:
            level_order = {
                "DEBUG": 0,
                "INFO": 1,
                "WARNING": 2,
                "ERROR": 3,
                "CRITICAL": 4,
            }
            min_level = level_order.get(level.upper(), 1)

            all_logs = get_log_buffer()
            filtered = [
                log
                for log in all_logs
                if level_order.get(log.get("level", "INFO"), 1) >= min_level
            ]

            return ToolResponse(
                data={
                    "logs": filtered[-limit:],
                    "total": len(filtered),
                    "buffer_size": len(all_logs),
                }
            ).model_dump()

        except Exception as e:
            return ToolError(
                error=str(e),
                error_code=ErrorCode.SDK_ERROR,
            ).model_dump()
