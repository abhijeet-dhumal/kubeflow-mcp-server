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

"""MCP resources for read-only cluster and model information.

Resources provide cacheable, read-only data that clients can fetch
without consuming tool call quota. Use resources for:
- Static configuration (supported models, runtimes)
- Cluster status snapshots
- Documentation and guides

NOTE: Resource URIs defined here should match RESOURCE_METADATA in server.py
to ensure SERVER_INSTRUCTIONS stays in sync.

NOTE: Model configurations and GPU requirements are example data based on
common configurations. Actual requirements vary by model architecture,
quantization, and training approach.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register_resources(mcp: "FastMCP") -> None:
    """Register MCP resources for cluster context and model info.

    Resources are read-only and cacheable by clients, reducing
    the need for repeated tool calls.
    """

    @mcp.resource("trainer://models/supported")
    def supported_models() -> str:
        """List of tested model configurations with resource requirements."""
        return """# Supported Models

Models tested with kubeflow-mcp fine-tuning:

## Small Models (8GB GPU)

| Model | Parameters | GPU Memory | Batch Size |
|-------|------------|------------|------------|
| google/gemma-2b | 2B | 8GB | 2-4 |
| meta-llama/Llama-3.2-1B | 1B | 6GB | 4-8 |
| Qwen/Qwen2.5-1.5B | 1.5B | 8GB | 2-4 |

## Medium Models (16-24GB GPU)

| Model | Parameters | GPU Memory | Batch Size |
|-------|------------|------------|------------|
| meta-llama/Llama-3.2-3B | 3B | 16GB | 2-4 |
| mistralai/Mistral-7B-v0.1 | 7B | 24GB | 1-2 |
| meta-llama/Llama-3.1-8B | 8B | 24GB | 1-2 |
| Qwen/Qwen2.5-7B-Instruct | 7B | 24GB | 1-2 |

## Large Models (40GB+ GPU)

| Model | Parameters | GPU Memory | Batch Size |
|-------|------------|------------|------------|
| meta-llama/Llama-3.1-70B | 70B | 140GB (2xA100-80GB) | 1 |

## Dataset Compatibility

Tested datasets for instruction tuning:
- `tatsu-lab/alpaca` - General instructions
- `databricks/dolly-15k` - Diverse tasks
- `OpenAssistant/oasst1` - Conversational
- `squad` - Question answering

## Model ID Formats

- `estimate_resources()`: Use bare ID like `google/gemma-2b`
- `fine_tune()`: Use `hf://` prefix like `hf://google/gemma-2b`
"""

    @mcp.resource("trainer://runtimes/info")
    def runtime_info() -> str:
        """Information about training runtimes."""
        return """# Training Runtimes

## torch-tune (Default)

The default runtime for LLM fine-tuning using TorchTune.

**Supports:**
- LoRA and QLoRA fine-tuning
- HuggingFace models and datasets
- Multi-GPU and multi-node training

**Use with:** `fine_tune()` tool

## torch-distributed

Generic PyTorch distributed training runtime.

**Supports:**
- Custom training scripts
- DDP (DistributedDataParallel)
- DeepSpeed integration

**Use with:** `run_custom_training()` or `run_container_training()`

## Checking Available Runtimes

```
list_runtimes()
```

Returns all ClusterTrainingRuntimes installed in your cluster.

## Runtime Patches

Customize runtime behavior when submitting jobs:

```python
fine_tune(
    model="hf://google/gemma-2b",
    dataset="hf://tatsu-lab/alpaca",
    # Target specific GPU nodes
    node_selector={"node-type": "gpu-a100"},
    # Schedule on tainted GPU nodes
    tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
    # Add environment variables
    env=[{"name": "NCCL_DEBUG", "value": "INFO"}],
    confirmed=True
)
```
"""

    @mcp.resource("trainer://guides/quickstart")
    def quickstart_guide() -> str:
        """Quick start guide for new users."""
        return """# Quick Start Guide

## 1. Check Your Cluster

Before training, always check available resources:

```
get_cluster_resources()
```

Look for:
- `gpu_total > 0` - You need GPUs for LLM training
- `nodes_with_gpu` - How many nodes have GPUs

## 2. Estimate Requirements

Check what your model needs:

```
estimate_resources(model="google/gemma-2b")
```

Compare `gpu_memory_required` with your available GPUs.

## 3. Fine-Tune a Model

### Preview first (always!)

```
fine_tune(
    model="hf://google/gemma-2b",
    dataset="hf://tatsu-lab/alpaca",
    batch_size=2,
    epochs=1,
    confirmed=False  # Preview only
)
```

### Submit after reviewing

```
fine_tune(
    model="hf://google/gemma-2b",
    dataset="hf://tatsu-lab/alpaca",
    batch_size=2,
    epochs=1,
    confirmed=True  # Actually submit
)
```

## 4. Monitor Progress

```
get_training_logs(name="your-job-name")
get_training_events(name="your-job-name")
```

## Common Issues

| Problem | Solution |
|---------|----------|
| No GPUs | Request GPU nodes from admin |
| OOMKilled | Reduce `batch_size` |
| Job stuck Pending | Check `get_training_events()` |
| Image pull failed | Verify image name |
"""

    @mcp.resource("trainer://guides/troubleshooting")
    def troubleshooting_quick_ref() -> str:
        """Quick troubleshooting reference."""
        return """# Troubleshooting Quick Reference

## Diagnostic Commands

```
get_training_job(name="job-name")      # Check status
get_training_events(name="job-name")   # K8s events
get_training_logs(name="job-name")     # Container logs
```

## Status Meanings

| Status | Action |
|--------|--------|
| Created | Check events for scheduling issues |
| Running | Check logs for progress |
| Failed | Check logs for error |
| Complete | Done! |

## Common Errors

### OOMKilled
- Reduce `batch_size`
- Use gradient checkpointing
- Try smaller model

### FailedScheduling
- No GPUs available - wait or reduce request
- Node selector too restrictive

### ImagePullBackOff
- Check image name spelling
- Verify registry access

### NCCL Timeout (multi-node)
- Network issues between nodes
- Try `env=[{"name": "NCCL_TIMEOUT", "value": "1800"}]`

## Recovery Commands

```
delete_training_job(name="failed-job")   # Clean up
suspend_training_job(name="running-job") # Pause
resume_training_job(name="paused-job")   # Continue
```
"""
