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

"""TrainerClient integration with MCP tools and skills.

Structure mirrors kubeflow/trainer/:
├── api/           # Tool implementations
├── types/         # Type definitions
└── constants/     # Constants
"""

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

MODULE_INFO = {
    "name": "trainer",
    "description": "Distributed training and LLM fine-tuning (wraps TrainerClient)",
    "sdk_client": "kubeflow.trainer.TrainerClient",
    "status": "implemented",
}

# Tool categories for progressive loading
TOOL_CATEGORIES = {
    "core": [
        # Essential tools - always loaded (~5 tools, ~500 tokens)
        get_cluster_resources,
        fine_tune,
        list_training_jobs,
        get_training_logs,
        list_runtimes,
    ],
    "planning": [
        get_cluster_resources,
        estimate_resources,
    ],
    "training": [
        fine_tune,
        run_custom_training,
        run_container_training,
    ],
    "discovery": [
        list_training_jobs,
        get_training_job,
        list_runtimes,
        get_runtime,
        get_runtime_packages,
    ],
    "monitoring": [
        get_training_logs,
        get_training_events,
        wait_for_training,
    ],
    "lifecycle": [
        delete_training_job,
        suspend_training_job,
        resume_training_job,
    ],
}

# All tools (for full mode)
TOOLS = [
    # planning.py
    get_cluster_resources,
    estimate_resources,
    # training.py
    fine_tune,
    run_custom_training,
    run_container_training,
    # discovery.py
    list_training_jobs,
    get_training_job,
    list_runtimes,
    get_runtime,
    get_runtime_packages,
    # monitoring.py
    get_training_logs,
    get_training_events,
    wait_for_training,
    # lifecycle.py
    delete_training_job,
    suspend_training_job,
    resume_training_job,
]


def get_tools(categories: list[str] | None = None) -> list:
    """Get tools by category for progressive loading.

    Args:
        categories: List of categories to load. Options:
            - "core": Essential tools only (~5 tools)
            - "planning": Resource planning tools
            - "training": Training submission tools
            - "discovery": Job/runtime discovery tools
            - "monitoring": Logs and events tools
            - "lifecycle": Job management tools
            - None: All tools (default)

    Returns:
        List of tool functions
    """
    if categories is None:
        return TOOLS

    tools = []
    seen = set()
    for cat in categories:
        for tool in TOOL_CATEGORIES.get(cat, []):  # type: ignore[attr-defined]
            if tool.__name__ not in seen:
                tools.append(tool)
                seen.add(tool.__name__)
    return tools


__all__ = [
    "MODULE_INFO",
    "TOOLS",
    "TOOL_CATEGORIES",
    "get_tools",
    *[t.__name__ for t in TOOLS],
]
