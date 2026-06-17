# Copyright The Kubeflow Authors
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

"""Trainer tool metadata — names, descriptions, annotations.

Intentionally zero heavy imports so this module loads in <5ms.
Used by the lazy tool registry to serve progressive/semantic mode
without triggering the 400ms kubeflow.trainer SDK import chain.
"""

from __future__ import annotations

# Tool names in workflow order (mirrors TOOLS list in __init__.py).
TOOL_NAMES: list[str] = [
    "pre_flight",
    "check_compatibility",
    "get_cluster_resources",
    "estimate_resources",
    "fine_tune",
    "run_custom_training",
    "run_container_training",
    "list_training_jobs",
    "get_training_job",
    "list_runtimes",
    "get_runtime",
    "get_training_logs",
    "get_training_events",
    "wait_for_training",
    "delete_training_job",
    "update_training_job",
    "inspect_crd",
    "inspect_controller",
    "patch_runtime",
    "create_runtime",
    "delete_runtime",
]

CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "pre_flight": (
        "One-shot: compatibility + cluster resources + model estimate + runtimes. "
        "Call FIRST. Pass model= for GPU sizing."
    ),
    "check_compatibility": "Verify K8s version, Trainer CRD, installed packages, and platform. Takes NO arguments. Prefer pre_flight() which runs this plus resource checks in one call.",
    "get_cluster_resources": "Check cluster GPU/CPU/memory availability. Takes NO arguments. Prefer pre_flight() which runs this plus compatibility checks in one call.",
    "estimate_resources": "Estimate GPU memory needed for a HuggingFace model. Use pre_flight(model=...) instead.",
    "list_training_jobs": "List training jobs. Filter by runtime, status, or namespace.",
    "get_training_job": "Get details of a specific training job. Supports optional namespace.",
    "list_runtimes": "List available ClusterTrainingRuntimes.",
    "get_runtime": "Get runtime config. Pass include_packages=True to fetch pip list (slow: creates a Pod).",
    "fine_tune": (
        "Fine-tune HuggingFace model with LoRA. Run list_runtimes() first to find the "
        "correct runtime name. Optional name= for custom job name. Set confirmed=True to submit."
    ),
    "run_custom_training": (
        "Run Python training script on the cluster. Pass runtime= for runtime selection. "
        "Set confirmed=True to submit."
    ),
    "run_container_training": (
        "Run training with custom container image. Pass runtime= and command= to "
        "override runtime and entrypoint. Set confirmed=True to submit."
    ),
    "get_training_logs": "Get pod logs from a training job. Supports optional namespace.",
    "get_training_events": "Get K8s events for debugging pending/failed jobs. Supports optional namespace.",
    "wait_for_training": "Block until job reaches target status (Complete/Failed). Supports optional namespace.",
    "delete_training_job": "[DESTRUCTIVE] Delete a training job permanently. Set confirmed=True to execute.",
    "update_training_job": "Suspend or resume a training job. Pass action='suspend' or 'resume'.",
    "inspect_crd": "List Trainer CRDs or get details for a specific one. Pass name= for details.",
    "inspect_controller": "Inspect controller pod. Pass view='logs' or 'events'. Auto-discovers namespace.",
    "patch_runtime": "Strategic merge patch on a ClusterTrainingRuntime. Set confirmed=True to apply.",
    "create_runtime": "Create a new ClusterTrainingRuntime. Set confirmed=True to create.",
    "delete_runtime": "[DESTRUCTIVE] Delete a ClusterTrainingRuntime. Lists dependent jobs first. Set confirmed=True.",
}

HEALTH_TOOL_DESCRIPTIONS: dict[str, str] = {
    "health_check": "Check server health and K8s connectivity.",
    "get_server_logs": "Get recent server logs for debugging. Filter by level.",
}

ALL_TOOL_DESCRIPTIONS: dict[str, str] = {
    **CLIENT_TOOL_DESCRIPTIONS,
    **HEALTH_TOOL_DESCRIPTIONS,
}
