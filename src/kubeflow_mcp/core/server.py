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

"""MCP server factory with dynamic client loading.

Designed for extensibility:
- Phase 1: trainer only
- Phase 2+: Contributors add optimizer, hub
"""

import importlib
import logging

from fastmcp import FastMCP

from kubeflow_mcp.common.constants import PROMPT_METADATA, RESOURCE_METADATA
from kubeflow_mcp.core.health import HealthManager
from kubeflow_mcp.core.policy import apply_policy_filters, get_allowed_tools
from kubeflow_mcp.core.prompts import register_prompts
from kubeflow_mcp.core.resources import register_resources

logger = logging.getLogger(__name__)

CLIENT_MODULES = {
    "trainer": "kubeflow_mcp.trainer",
    "optimizer": "kubeflow_mcp.optimizer",
    "hub": "kubeflow_mcp.hub",
}

# Concise tool descriptions for MCP clients (token-efficient)
# Full docstrings remain in tool functions for API documentation
TOOL_DESCRIPTIONS: dict[str, str] = {
    # Planning
    "get_cluster_resources": "Check cluster GPU/CPU availability. Call FIRST before training.",
    "estimate_resources": "Estimate GPU memory needed for a HuggingFace model.",
    # Discovery
    "list_training_jobs": "List training jobs. Filter by runtime or status.",
    "get_training_job": "Get details of a specific training job.",
    "list_runtimes": "List available ClusterTrainingRuntimes.",
    "get_runtime": "Get runtime configuration details.",
    "get_runtime_packages": "List pip packages in a runtime container.",
    # Training (require confirmed=True to submit)
    "fine_tune": "Fine-tune HuggingFace model with LoRA. Set confirmed=True to submit.",
    "run_custom_training": "Run Python training script. Set confirmed=True to submit.",
    "run_container_training": "Run training with custom container. Set confirmed=True to submit.",
    # Monitoring
    "get_training_logs": "Get pod logs from a training job.",
    "get_training_events": "Get K8s events for debugging pending/failed jobs.",
    "wait_for_training": "Block until job reaches target status (Complete/Failed).",
    # Lifecycle
    "delete_training_job": "Delete a training job permanently.",
    "suspend_training_job": "Pause a running job. Resume with resume_training_job.",
    "resume_training_job": "Resume a suspended training job.",
}

# Tool annotations for MCP clients
# See: https://spec.modelcontextprotocol.io/specification/2025-03-26/server/tools/#annotations
# Tags enable phase-based tool discovery (planning, discovery, training, monitoring, lifecycle)
TOOL_ANNOTATIONS: dict[str, dict] = {
    # Phase 1: Planning tools (call FIRST before any training)
    "get_cluster_resources": {
        "title": "Get Cluster Resources",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["planning", "cluster", "gpu"],
    },
    "estimate_resources": {
        "title": "Estimate Training Resources",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["planning", "resources", "estimation"],
    },
    # Phase 2: Discovery tools (explore available resources)
    "list_training_jobs": {
        "title": "List Training Jobs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "jobs"],
    },
    "get_training_job": {
        "title": "Get Training Job Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "monitoring", "jobs"],
    },
    "list_runtimes": {
        "title": "List Training Runtimes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "runtimes"],
    },
    "get_runtime": {
        "title": "Get Runtime Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "runtimes"],
    },
    "get_runtime_packages": {
        "title": "Get Runtime Packages",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "runtimes"],
    },
    # Phase 3: Training tools (create resources, require confirmation)
    "fine_tune": {
        "title": "Fine-tune Model",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["training", "fine-tuning", "llm"],
    },
    "run_custom_training": {
        "title": "Run Custom Training Script",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["training", "custom", "script"],
    },
    "run_container_training": {
        "title": "Run Container Training",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["training", "container"],
    },
    # Phase 4: Monitoring tools (track progress and debug)
    "get_training_logs": {
        "title": "Get Training Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "logs", "debug"],
    },
    "get_training_events": {
        "title": "Get Training Events",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "events", "debug"],
    },
    "wait_for_training": {
        "title": "Wait for Training Completion",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "blocking"],
    },
    # Lifecycle tools (manage job state)
    "delete_training_job": {
        "title": "Delete Training Job",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "cleanup"],
    },
    "suspend_training_job": {
        "title": "Suspend Training Job",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "pause"],
    },
    "resume_training_job": {
        "title": "Resume Training Job",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "resume"],
    },
}


def _build_server_instructions() -> str:
    """Build server instructions dynamically from metadata."""
    prompts_section = "\n".join(
        f"- {name} → {desc}" for name, desc in PROMPT_METADATA.items()
    )
    resources_section = "\n".join(
        f"- {uri} → {desc}" for uri, desc in RESOURCE_METADATA.items()
    )

    return f"""Kubeflow MCP Server - AI Model Training on Kubernetes

CRITICAL WORKFLOW - Follow these steps IN ORDER:

1. PLANNING (always do first):
   - get_cluster_resources() → Check GPU availability
   - estimate_resources(model) → Check memory requirements
   - If gpu_total=0 or insufficient memory, STOP and inform user

2. DISCOVERY (if needed):
   - list_runtimes() → Find available training runtimes
   - list_training_jobs() → See existing jobs

3. TRAINING (requires user confirmation):
   - fine_tune(..., confirmed=False) → Preview config first
   - Show preview to user, ask for confirmation
   - fine_tune(..., confirmed=True) → Submit job only after approval

4. MONITORING:
   - get_training_job(job_id) → Check status
   - get_training_logs(job_id) → View output/errors
   - get_training_events(job_id) → Debug scheduling issues

TOOL SELECTION:
- HuggingFace model fine-tuning → fine_tune()
- Custom Python training script → run_custom_training()
- Pre-built container image → run_container_training()

IMPORTANT:
- NEVER skip planning steps before training
- ALWAYS preview before submitting (confirmed=False first)
- Training jobs consume GPU resources - be conservative
- Use get_training_events() to debug stuck/failed jobs

PROMPTS (use for detailed guidance):
{prompts_section}

RESOURCES (read-only reference data):
{resources_section}
"""


SERVER_INSTRUCTIONS = _build_server_instructions()


def create_server(
    clients: list[str] | None = None,
    persona: str = "ml-engineer",
) -> FastMCP:
    """Create MCP server with dynamic client loading.

    Args:
        clients: List of client modules to load (default: ["trainer"])
                 Options: "trainer", "optimizer", "hub"
        persona: User role for tool filtering

    Returns:
        Configured FastMCP server instance

    Tool filtering is applied in two stages:
    1. Persona filtering: Built-in or custom persona defines base tool set
    2. Policy filtering: ~/.kf-mcp-policy.yaml can further restrict via allow/deny lists
    """
    mcp: FastMCP = FastMCP("kubeflow-mcp", instructions=SERVER_INSTRUCTIONS)

    # Stage 1: Get tools allowed by persona
    allowed_tools = get_allowed_tools(persona)

    if clients is None:
        clients = ["trainer"]

    # Collect all available tool names first (for policy filtering)
    all_tool_names: set[str] = set()
    for client_name in clients:
        if client_name not in CLIENT_MODULES:
            continue
        try:
            module = importlib.import_module(CLIENT_MODULES[client_name])
            if hasattr(module, "TOOLS") and module.TOOLS:
                for tool_func in module.TOOLS:
                    all_tool_names.add(tool_func.__name__)
        except ImportError:
            pass

    # Stage 2: Apply policy filters (allow/deny lists from ~/.kf-mcp-policy.yaml)
    if allowed_tools is not None:
        # Start with persona-allowed tools
        final_allowed = allowed_tools
    else:
        # platform-admin: start with all tools
        final_allowed = all_tool_names

    # Apply policy file restrictions
    final_allowed = apply_policy_filters(final_allowed)
    logger.debug(f"Final allowed tools after policy: {len(final_allowed)}")

    for client_name in clients:
        if client_name not in CLIENT_MODULES:
            logger.warning(f"Unknown client: {client_name}, skipping")
            continue

        try:
            module = importlib.import_module(CLIENT_MODULES[client_name])

            if not hasattr(module, "TOOLS") or not module.TOOLS:
                logger.info(f"Client '{client_name}' has no tools (stub module)")
                continue

            registered = 0
            for tool_func in module.TOOLS:
                tool_name = tool_func.__name__

                if tool_name in final_allowed:
                    # Get annotations and description for this tool
                    annotations = TOOL_ANNOTATIONS.get(tool_name)
                    description = TOOL_DESCRIPTIONS.get(tool_name)

                    # Register with description (overrides docstring for MCP)
                    # and annotations (MCP hints)
                    if annotations and description:
                        mcp.tool(description=description, annotations=annotations)(
                            tool_func
                        )
                    elif description:
                        mcp.tool(description=description)(tool_func)
                    elif annotations:
                        mcp.tool(annotations=annotations)(tool_func)
                    else:
                        mcp.tool()(tool_func)
                    registered += 1
                    logger.debug(f"Registered tool: {tool_name}")

            logger.info(f"Loaded client '{client_name}' with {registered} tools")

        except ImportError as e:
            logger.error(f"Failed to load client '{client_name}': {e}")

    # Register MCP prompts for structured workflows
    register_prompts(mcp)

    # Register MCP resources for read-only reference data
    register_resources(mcp)

    health = HealthManager(mcp)
    health.register_health_tools()

    return mcp
