# Copyright 2026 The Kubeflow Authors
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

"""Kubeflow TrainerClient MCP tools."""

from kubeflow_mcp.trainer.api.discovery import (
    get_runtime,
    get_runtime_packages,
    get_training_job,
    list_runtimes,
    list_training_jobs,
)
from kubeflow_mcp.trainer.api.lifecycle import (
    delete_training_job,
    resume_training_job,
    suspend_training_job,
)
from kubeflow_mcp.trainer.api.monitoring import (
    get_training_events,
    get_training_logs,
    wait_for_training,
)
from kubeflow_mcp.trainer.api.planning import estimate_resources, get_cluster_resources
from kubeflow_mcp.trainer.api.training import (
    fine_tune,
    run_container_training,
    run_custom_training,
)

TOOLS = [
    get_cluster_resources,
    estimate_resources,
    list_training_jobs,
    get_training_job,
    list_runtimes,
    get_runtime,
    get_runtime_packages,
    fine_tune,
    run_custom_training,
    run_container_training,
    get_training_logs,
    get_training_events,
    wait_for_training,
    delete_training_job,
    suspend_training_job,
    resume_training_job,
]

MODULE_INFO = {
    "name": "trainer",
    "status": "implemented",
    "description": "Kubeflow TrainerClient tools for fine-tuning and training job management",
    "tool_count": len(TOOLS),
}
