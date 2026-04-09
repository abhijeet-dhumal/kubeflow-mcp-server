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

"""Discovery tools for training jobs and runtimes.

Maps to TrainerClient methods (SDK 0.4.0):
- list_training_jobs() → TrainerClient.list_jobs()
- get_training_job() → TrainerClient.get_job()
- list_runtimes() → TrainerClient.list_runtimes()
- get_runtime() → TrainerClient.get_runtime()
"""

from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details
from kubeflow_mcp.common.utils import get_trainer_client

# Legacy filter/docs used "Succeeded" for finished TrainJobs; SDK uses "Complete".
_JOB_STATUS_FILTER_ALIASES: dict[str, str] = {
    "Succeeded": "Complete",
}


def _trainjob_runtime_to_mcp(runtime: object | None) -> dict[str, str] | None:
    """Serialize SDK Runtime for MCP JSON (avoid passing raw objects)."""
    if runtime is None:
        return None
    name = getattr(runtime, "name", None)
    if isinstance(name, str) and name:
        return {"name": name}
    return None


def list_training_jobs(
    runtime: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List training jobs in the current namespace.

    Args:
        runtime: Filter by ClusterTrainingRuntime name (e.g., ``torch-tune``).
        status: Filter by TrainJob status: ``Created``, ``Running``, ``Complete``,
            ``Failed``, ``Suspended``. ``Succeeded`` is accepted as an alias for ``Complete``.
        limit: Maximum jobs to return. Defaults to 50.

    Returns:
        dict: Response containing:

        - ``jobs`` (list): List of jobs with ``name``, ``status``, ``runtime``
        - ``total`` (int): Total matching jobs

    Example:
        >>> list_training_jobs(status="Running")
        {"data": {"jobs": [{"name": "fine-tune-abc", "status": "Running"}], "total": 1}}
    """
    try:
        client = get_trainer_client()
        if runtime:
            rt_obj = client.get_runtime(name=runtime)
            jobs = client.list_jobs(runtime=rt_obj)
        else:
            jobs = client.list_jobs()

        job_list = []
        for job in jobs:
            jr = job.runtime if hasattr(job, "runtime") else None
            job_data = {
                "name": job.name,
                "status": job.status if hasattr(job, "status") else "Unknown",
                "runtime": _trainjob_runtime_to_mcp(jr),
            }
            job_list.append(job_data)

        if status:
            want = _JOB_STATUS_FILTER_ALIASES.get(status, status)
            job_list = [j for j in job_list if j.get("status") == want]

        return ToolResponse(
            data={"jobs": job_list[:limit], "total": len(job_list)}
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_training_job(name: str) -> dict[str, Any]:
    """Get details of a specific training job.

    Args:
        name: The TrainJob name.

    Returns:
        dict: Response containing ``name``, ``status``, ``runtime``.

    Raises:
        ToolError: If job not found (``RESOURCE_NOT_FOUND``).
    """
    try:
        client = get_trainer_client()
        job = client.get_job(name=name)

        jr = job.runtime if hasattr(job, "runtime") else None
        return ToolResponse(
            data={
                "name": job.name,
                "status": job.status if hasattr(job, "status") else "Unknown",
                "runtime": _trainjob_runtime_to_mcp(jr),
            }
        ).model_dump()

    except Exception as e:
        if "not found" in str(e).lower():
            return ToolError(
                error=f"Training job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def list_runtimes() -> dict[str, Any]:
    """List available ClusterTrainingRuntimes.

    Call this if ``fine_tune()`` fails with "runtime not found" to see
    what runtimes are installed in the cluster.

    Returns:
        dict: Response containing:

        - ``runtimes`` (list): Available runtimes with ``name``
        - ``total`` (int): Runtime count

    Example:
        >>> list_runtimes()
        {"data": {"runtimes": [{"name": "torch-tune"}, {"name": "torch-distributed"}], "total": 2}}
    """
    try:
        client = get_trainer_client()
        runtimes = client.list_runtimes()

        runtime_list = []
        for rt in runtimes:
            runtime_list.append(
                {
                    "name": rt.name if hasattr(rt, "name") else str(rt),
                }
            )

        return ToolResponse(
            data={"runtimes": runtime_list, "total": len(runtime_list)}
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_runtime(name: str) -> dict[str, Any]:
    """Get ClusterTrainingRuntime configuration.

    Args:
        name: Runtime name (e.g., ``torch-tune``).

    Returns:
        dict: Response containing runtime ``name`` and configuration.

    Raises:
        ToolError: If runtime not found (``RESOURCE_NOT_FOUND``).
    """
    try:
        client = get_trainer_client()
        rt = client.get_runtime(name=name)

        rt_name = rt.name if hasattr(rt, "name") else name

        # Serialise as much of the runtime spec as the SDK exposes.
        # Fields are optional — we use getattr with None fallbacks so the
        # response is forward-compatible with future SDK additions.
        data: dict[str, Any] = {"name": rt_name}

        spec = getattr(rt, "spec", None)
        if spec is not None:
            ml_policy = getattr(spec, "ml_policy", None)
            data["framework"] = getattr(ml_policy, "torch", None) or getattr(
                ml_policy, "mpi", None
            )
            template = getattr(spec, "template", None)
            if template is not None:
                replicated_jobs = getattr(
                    getattr(template, "spec", None), "replicated_jobs", None
                )
                if replicated_jobs:
                    data["replicated_jobs"] = [
                        {
                            "name": getattr(rj, "name", None),
                            "replicas": getattr(
                                getattr(getattr(rj, "template", None), "spec", None),
                                "completions",
                                None,
                            ),
                        }
                        for rj in replicated_jobs
                    ]

        return ToolResponse(data=data).model_dump()

    except Exception as e:
        if "not found" in str(e).lower():
            return ToolError(
                error=f"Runtime '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_runtime_packages(name: str) -> dict[str, Any]:
    """Get pip packages installed in a runtime container.

    Executes ``pip list`` inside the runtime's container image.

    Args:
        name: Runtime name (e.g., ``torch-tune``).

    Returns:
        dict: Response containing ``runtime`` name and ``packages`` list.

    Note:
        SDK may print output to stdout instead of returning it.
    """
    try:
        client = get_trainer_client()
        runtime = client.get_runtime(name=name)
        result = client.get_runtime_packages(runtime=runtime)

        if result is None:
            return ToolResponse(
                data={
                    "runtime": name,
                    "message": "Packages printed to stdout. Check logs for details.",
                }
            ).model_dump()

        return ToolResponse(data={"runtime": name, "packages": result}).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
