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

"""MCP prompts for structured workflows and guides.

Prompts provide reusable, parameterized templates for common tasks.
They are exposed via MCP protocol and can be used by any MCP client.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register_prompts(mcp: "FastMCP") -> None:
    """Register MCP prompts for training workflows.

    These prompts provide structured guidance for common tasks:
    - fine_tuning_workflow: Step-by-step fine-tuning guide
    - custom_training_workflow: Custom script/container guide
    - troubleshooting_guide: Diagnose and fix job failures
    - resource_planning: Estimate resources before training
    """

    @mcp.prompt(
        name="fine_tuning_workflow",
        description="Step-by-step workflow for fine-tuning LLMs with LoRA",
    )
    def fine_tuning_workflow(
        model: str = "",
        dataset: str = "",
    ) -> str:
        """Generate a fine-tuning workflow guide."""
        model_display = model or "your-model"
        dataset_display = dataset or "your-dataset"

        return f"""# Fine-Tuning Workflow

Fine-tune {model_display} on {dataset_display}.

## Step 1: Check Resources (REQUIRED)

```
get_cluster_resources()
```
- Verify gpu_total > 0
- If no GPUs, request GPU nodes or use CPU training

## Step 2: Estimate Requirements

```
estimate_resources(model="{model or "meta-llama/Llama-3.2-3B"}")
```
- Compare with available GPUs from Step 1
- Adjust batch_size if memory is limited

## Step 3: Check Runtimes

```
list_runtimes()
```
- Verify torch-tune runtime is available
- Note the runtime name for fine_tune()

## Step 4: Preview Configuration (ALWAYS DO THIS)

```
fine_tune(
    model="hf://{model_display}",
    dataset="hf://{dataset_display}",
    batch_size=4,
    epochs=1,
    confirmed=False  # Preview only
)
```
- Review the config before submitting
- Check resource requests match cluster capacity

## Step 5: Submit Job (After User Approval)

```
fine_tune(
    model="hf://{model_display}",
    dataset="hf://{dataset_display}",
    batch_size=4,
    epochs=1,
    confirmed=True  # Actually submit
)
```

## Step 6: Monitor Progress

```
get_training_logs(name="<job-name>")
get_training_events(name="<job-name>")
wait_for_training(name="<job-name>", timeout_seconds=3600)
```

## Key Parameters

| Parameter | Purpose |
|-----------|---------|
| batch_size | Reduce if OOM, increase if fast |
| epochs | 1-3 for most datasets |
| num_nodes | Multi-node for large models |
| lora_rank | Higher = more capacity |
| hf_token | Required for gated models |

## Common Issues

- OOMKilled → Reduce batch_size
- No GPUs → Check get_cluster_resources()
- Runtime not found → Check list_runtimes()
"""

    @mcp.prompt(
        name="custom_training_workflow",
        description="Guide for running custom Python scripts or containers",
    )
    def custom_training_workflow(
        training_type: str = "script",
    ) -> str:
        """Generate a custom training workflow guide."""
        if training_type == "container":
            return """# Container Training Workflow

Run a pre-built container image on the cluster.

## Step 1: Check Resources

```
get_cluster_resources()
```

## Step 2: Preview Configuration

```
run_container_training(
    image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    num_nodes=1,
    gpu_per_node=1,
    env={"LEARNING_RATE": "0.001"},
    confirmed=False  # Preview
)
```

## Step 3: Submit (After Approval)

```
run_container_training(
    image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    num_nodes=1,
    gpu_per_node=1,
    env={"LEARNING_RATE": "0.001"},
    confirmed=True
)
```

## Advanced Options

### Mount Data Volume
```python
volumes=[{
    "name": "data",
    "persistentVolumeClaim": {"claimName": "training-data"}
}],
volume_mounts=[{
    "name": "data",
    "mountPath": "/data"
}]
```

### Target GPU Type
```python
node_selector={"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"},
tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}]
```
"""
        else:
            return """# Custom Script Training Workflow

Run your Python training script on the cluster.

## Step 1: Check Resources

```
get_cluster_resources()
```

## Step 2: Write Your Script

```python
script = '''
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()

model = torch.nn.Linear(10, 10).cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    loss = model(torch.randn(32, 10).cuda()).sum()
    loss.backward()
    optimizer.step()
    if rank == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
'''
```

## Step 3: Preview

```
run_custom_training(
    script=script,
    num_nodes=1,
    gpu_per_node=1,
    packages=["torch"],
    confirmed=False
)
```

## Step 4: Submit

```
run_custom_training(
    script=script,
    num_nodes=1,
    gpu_per_node=1,
    packages=["torch"],
    confirmed=True
)
```

## Security Restrictions

NOT allowed in scripts:
- import os, subprocess, sys
- eval(), exec(), compile()
- open(), __import__()

Use run_container_training() if you need system access.
"""

    @mcp.prompt(
        name="troubleshooting_guide",
        description="Diagnose and fix common training job failures",
    )
    def troubleshooting_guide(
        error_type: str = "",
    ) -> str:
        """Generate a troubleshooting guide."""
        base_guide = """# Troubleshooting Training Jobs

## Diagnostic Workflow

1. Check job status:
   ```
   get_training_job(name="<job-name>")
   ```

2. Check K8s events:
   ```
   get_training_events(name="<job-name>")
   ```

3. Check container logs:
   ```
   get_training_logs(name="<job-name>")
   ```

## Job Status Reference

| Status | Meaning | Action |
|--------|---------|--------|
| Created | Job created / starting | Check events |
| Running | Training in progress | Check logs |
| Complete | TrainJob finished successfully | Done |
| Failed | Crashed | Check logs |
| Suspended | Paused | resume_training_job() |

"""
        error_guides = {
            "oom": """## OOMKilled (Out of Memory)

**Symptoms**: Job fails, events show OOMKilled

**Solutions**:
1. Reduce batch_size: `fine_tune(..., batch_size=2, ...)`
2. Use gradient checkpointing (custom training)
3. Enable quantization (QLoRA)
4. Use multi-node to distribute memory

""",
            "pending": """## Job Stuck in Pending

**Symptoms**: Job never starts running

**Diagnosis**:
```
get_training_events(name="<job>")
# Look for: FailedScheduling, Insufficient nvidia.com/gpu
```

**Solutions**:
1. Check `get_cluster_resources()` for available GPUs
2. Reduce gpu_per_node request
3. Add tolerations for GPU nodes
4. Wait for resources to free up

""",
            "image": """## Image Pull Failed

**Symptoms**: ErrImagePull, ImagePullBackOff

**Solutions**:
1. Verify image name is correct
2. Check imagePullSecrets in namespace
3. Verify registry credentials
4. Check Docker Hub rate limits

""",
            "nccl": """## NCCL Timeout (Multi-Node)

**Symptoms**: Job hangs, then fails with NCCL timeout

**Solutions**:
1. Check network between nodes
2. Increase timeout:
   ```
   fine_tune(..., env=[{"name": "NCCL_TIMEOUT", "value": "1800"}])
   ```
3. Use single node if network issues persist

""",
        }

        if error_type.lower() in error_guides:
            return base_guide + error_guides[error_type.lower()]

        return (
            base_guide
            + """## Common Issues

### OOMKilled
- Reduce batch_size
- Enable gradient checkpointing
- Use quantization

### Pending Forever
- Check get_cluster_resources()
- Add GPU tolerations
- Reduce resource requests

### Image Pull Failed
- Verify image name
- Check registry auth

### NCCL Timeout
- Check network
- Increase NCCL_TIMEOUT
- Try single node

### Permission Denied (Gated Models)
- Accept license on HuggingFace
- Provide hf_token parameter
"""
        )

    @mcp.prompt(
        name="resource_planning",
        description="Plan resources before training a model",
    )
    def resource_planning(
        model: str = "",
    ) -> str:
        """Generate resource planning guide."""
        return f"""# Resource Planning Guide

Plan resources for training {model or "your model"}.

## Step 1: Check Cluster Capacity

```
get_cluster_resources()
```

Key metrics:
- gpu_total: Total GPUs available
- nodes_with_gpu: Nodes that have GPUs
- gpu_types: Types of GPUs (A100, V100, etc.)

## Step 2: Estimate Model Requirements

```
estimate_resources(model="{model or "meta-llama/Llama-3.2-3B"}")
```

Returns:
- gpu_memory_required: Minimum VRAM needed
- gpu_type_recommended: Suitable GPU type
- batch_size_recommendation: Based on memory

## GPU Memory Reference

| Model Size | Min GPU Memory | Recommended |
|------------|----------------|-------------|
| 1-3B | 8GB | T4, RTX 3080 |
| 7-8B | 24GB | A10, RTX 4090 |
| 13B | 40GB | A100-40GB |
| 70B | 80GB+ | A100-80GB x2 |

## Batch Size Guide

| GPU Memory | batch_size |
|------------|------------|
| 8GB | 1-2 |
| 16GB | 2-4 |
| 24GB | 4-8 |
| 40GB+ | 8-16 |

## Multi-Node Considerations

Use multi-node (num_nodes > 1) when:
- Model doesn't fit on single GPU
- Training is too slow on single node
- You need more total memory

Requirements:
- Same GPU type across nodes
- Good network bandwidth
- NCCL-compatible setup
"""

    @mcp.prompt(
        name="monitoring_workflow",
        description="Monitor running training jobs and debug issues",
    )
    def monitoring_workflow(
        job_name: str = "",
    ) -> str:
        """Generate monitoring workflow guide."""
        job_display = job_name or "<job-name>"

        return f"""# Monitoring Workflow

Monitor and debug training job: {job_display}

## Step 1: Check Job Status

```
get_training_job(name="{job_display}")
```

Key fields:
- status: Current state (Created, Running, Complete, Failed)
- start_time: When job started
- completion_time: When job finished (if done)

## Step 2: Watch Progress (Running Jobs)

```
get_training_logs(name="{job_display}")
```

Look for:
- Epoch progress: "Epoch 1/3: loss=2.34"
- GPU utilization messages
- Checkpoint saves

## Step 3: Check Events (Stuck/Failed Jobs)

```
get_training_events(name="{job_display}")
```

Common events:
| Event | Meaning |
|-------|---------|
| Scheduled | Pod assigned to node |
| Pulling | Downloading container image |
| Started | Container running |
| FailedScheduling | No suitable node (check resources) |
| OOMKilled | Out of memory (reduce batch_size) |
| BackOff | Container crash loop |

## Step 4: Wait for Completion

```
wait_for_training(name="{job_display}", timeout_seconds=3600)
```

Returns when job succeeds, fails, or timeout.

## Quick Diagnosis Checklist

1. **Job stuck in Pending?**
   - Run `get_cluster_resources()` - are GPUs available?
   - Check events for `FailedScheduling`

2. **Job failed immediately?**
   - Check events for `ImagePullBackOff` (wrong image)
   - Check logs for import errors (missing packages)

3. **Job failed during training?**
   - Check logs for `OOMKilled` (reduce batch_size)
   - Check logs for CUDA errors (driver issues)
   - Check logs for NCCL timeout (network issues)

4. **Job running but no progress?**
   - Check logs for hanging at data loading
   - Check if learning rate is too low (loss not decreasing)

## Recovery Actions

```
# Delete failed job and retry
delete_training_job(name="{job_display}")

# Suspend running job (pause)
suspend_training_job(name="{job_display}")

# Resume suspended job
resume_training_job(name="{job_display}")
```
"""
