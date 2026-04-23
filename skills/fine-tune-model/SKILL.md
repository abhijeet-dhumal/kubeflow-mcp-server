# Fine-Tune a Model on Kubeflow

Use the kubeflow-mcp MCP server tools to fine-tune a HuggingFace model on a Kubernetes cluster.

## Steps

1. **Check cluster resources** — call `get_cluster_resources()` and verify `gpu_total > 0`
2. **Estimate memory** — call `estimate_resources(model="<bare-model-id>")` (e.g. `google/gemma-2b`) and compare `gpu_memory_required` with available GPUs
3. **Preview the job** — call `fine_tune(model="hf://<model>", dataset="hf://<dataset>", confirmed=false)` and show the config to the user
4. **Submit after approval** — call `fine_tune(...)` again with `confirmed=true`
5. **Monitor** — call `get_training_logs(name="<job-name>")` and `get_training_events(name="<job-name>")` to track progress

## Key Details

- Model URIs use `hf://` prefix for fine_tune (e.g. `hf://google/gemma-2b`) but bare IDs for estimate_resources (e.g. `google/gemma-2b`)
- If `gpu_total=0`, inform the user — GPU nodes are required for LLM fine-tuning
- For gated models (Llama, Mistral), pass `hf_token`
- For QLoRA (lower memory), set `quantize_base=true`
- Always preview before submitting — training jobs consume GPU resources
