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

"""Planning tools for resource estimation and cluster inspection.

Maps to cluster inspection functionality:
- get_cluster_resources() → K8s node inspection
- estimate_resources() → Model-based resource estimation (via HuggingFace API)
"""

from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details


def _get_model_info_from_hf(model: str) -> dict[str, Any] | None:
    """Fetch model info from HuggingFace Hub."""
    try:
        from huggingface_hub import model_info

        info = model_info(model)

        # Get parameter count from safetensors metadata
        params = None
        if info.safetensors:
            params = info.safetensors.total

        # Try card_data for parameter count
        if not params and info.card_data:
            params = getattr(info.card_data, "num_parameters", None)

        return {
            "model_id": info.id,
            "params": params,
            "library": getattr(info, "library_name", None),
            "pipeline": getattr(info, "pipeline_tag", None),
        }

    except Exception as e:
        return {"error": str(e)}


def _estimate_from_params(params: float, batch_size: int = 4) -> dict[str, Any]:
    """Estimate resources based on parameter count.

    Rules of thumb for LoRA fine-tuning with bf16:
    - GPU memory ≈ params * 2 bytes (weights) + activations + overhead
    - Each billion params ≈ 4-6GB GPU memory for LoRA training
    - Full fine-tuning needs 4-6x more (gradients + optimizer states)
    """
    params_b = params / 1e9  # Convert to billions

    # GPU memory estimation (training with LoRA, bf16)
    # Base: ~4GB per billion params + 2GB overhead
    gpu_memory_gb = int(params_b * 4 + 2)

    # Adjust for batch size (each +1 batch adds ~10% memory)
    gpu_memory_gb = int(gpu_memory_gb * (1 + (batch_size - 1) * 0.1))

    # Determine GPU count and type needed
    if gpu_memory_gb <= 8:
        gpu_count = 1
        gpu_type = "8GB (RTX 3070/4070)"
    elif gpu_memory_gb <= 16:
        gpu_count = 1
        gpu_type = "16GB (T4/RTX 4080)"
    elif gpu_memory_gb <= 24:
        gpu_count = 1
        gpu_type = "24GB (A10/RTX 3090)"
    elif gpu_memory_gb <= 40:
        gpu_count = 1
        gpu_type = "40GB (A100-40GB)"
    elif gpu_memory_gb <= 80:
        gpu_count = 1
        gpu_type = "80GB (A100-80GB/H100)"
    else:
        gpu_count = max(2, (gpu_memory_gb + 79) // 80)
        gpu_type = f"{gpu_count}x 80GB GPUs"

    # System memory (CPU RAM) - roughly 2x GPU memory for data loading
    system_memory = max(16, gpu_memory_gb * 2)

    return {
        "gpu_count": gpu_count,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_type": gpu_type,
        "system_memory_gi": system_memory,
        "params_billions": round(params_b, 2),
    }


def get_cluster_resources() -> dict[str, Any]:
    """Check cluster GPU and compute availability.

    Call this before submitting training jobs to verify resources exist.

    Returns:
        dict: Response containing:

        - ``gpu_total`` (int): Total GPUs across all nodes
        - ``nodes_with_gpu`` (int): Number of nodes with GPUs
        - ``node_count`` (int): Total node count
        - ``nodes`` (list): Per-node details (name, memory, cpu, gpus)

    Example:
        >>> get_cluster_resources()
        {"data": {"gpu_total": 4, "nodes_with_gpu": 2, ...}}

    Note:
        Returns ``gpu_total=0`` if no GPUs available - LLM fine-tuning requires GPUs.
    """
    try:
        from kubeflow_mcp.common.utils import K8S_TIMEOUT, get_core_v1_api

        v1 = get_core_v1_api()
        nodes = v1.list_node(_request_timeout=K8S_TIMEOUT)
        gpu_total = 0
        node_info = []

        for node in nodes.items:
            alloc = node.status.allocatable or {}
            gpu = int(alloc.get("nvidia.com/gpu", 0))
            gpu_total += gpu

            node_data = {
                "name": node.metadata.name,
                "memory": alloc.get("memory"),
                "cpu": alloc.get("cpu"),
            }

            if gpu > 0:
                node_data["gpus"] = gpu

            node_info.append(node_data)

        return ToolResponse(
            data={
                "gpu_total": gpu_total,
                "nodes_with_gpu": sum(1 for n in node_info if n.get("gpus", 0) > 0),
                "node_count": len(node_info),
                "nodes": node_info,
            }
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()


def estimate_resources(
    model: str,
    num_workers: int = 1,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Estimate GPU and memory requirements for training a model.

    Fetches model metadata from HuggingFace Hub and calculates resource
    requirements based on parameter count for LoRA fine-tuning with bf16.

    Args:
        model: HuggingFace model ID. Accepts ``google/gemma-2b`` or ``hf://google/gemma-2b``.
        num_workers: Number of distributed workers. Defaults to 1.
        batch_size: Per-GPU batch size. Defaults to 4.

    Returns:
        dict: Response containing:

        - ``params_billions`` (float): Model size in billions of parameters
        - ``gpu_per_worker`` (int): GPUs needed per worker
        - ``gpu_memory_required`` (str): Estimated GPU memory (e.g., "16GB")
        - ``gpu_type_recommended`` (str): Suggested GPU type
        - ``total_gpu`` (int): Total GPUs needed (gpu_per_worker * num_workers)
        - ``recommendation`` (str): Human-readable suggestion

    Example:
        >>> estimate_resources("google/gemma-2b", batch_size=2)
        {"data": {"params_billions": 2.0, "gpu_memory_required": "16GB", ...}}
    """
    try:
        # Strip hf:// prefix if present (fine_tune uses hf://, but HF API needs raw ID)
        model_id = model.removeprefix("hf://")

        # Fetch model info from HuggingFace
        hf_info = _get_model_info_from_hf(model_id)

        if not hf_info or "error" in hf_info:
            error_msg = (
                hf_info.get("error", "Unknown error") if hf_info else "API failed"
            )
            return ToolError(
                error=f"Could not fetch model info from HuggingFace: {error_msg}",
                error_code=ErrorCode.SDK_ERROR,
                details={
                    "hint": "Ensure model path is correct (e.g., 'meta-llama/Llama-3.2-1B')"
                },
            ).model_dump()

        params = hf_info.get("params")
        if not params:
            return ToolError(
                error="Model found but parameter count not available in HuggingFace metadata",
                error_code=ErrorCode.SDK_ERROR,
                details={
                    "model_id": hf_info.get("model_id"),
                    "hint": "Try a different model or check HuggingFace model card",
                },
            ).model_dump()

        # Calculate estimates from parameter count
        estimates = _estimate_from_params(params, batch_size)

        gpu_per_worker = estimates["gpu_count"]
        total_gpu = gpu_per_worker * num_workers

        return ToolResponse(
            data={
                "model": model,
                "params_billions": estimates["params_billions"],
                "gpu_per_worker": gpu_per_worker,
                "gpu_memory_required": f"{estimates['gpu_memory_gb']}GB",
                "gpu_type_recommended": estimates["gpu_type"],
                "memory_per_worker": f"{estimates['system_memory_gi']}Gi",
                "total_gpu": total_gpu,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "training_type": "LoRA (bf16)",
                "recommendation": f"Request {total_gpu} GPU(s) - {estimates['gpu_type']}",
            }
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
