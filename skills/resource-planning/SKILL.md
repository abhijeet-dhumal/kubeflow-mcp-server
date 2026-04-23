# Plan Resources for Training

Use the kubeflow-mcp MCP server tools to assess cluster capacity and estimate resource requirements before submitting jobs.

## Steps

1. **Check cluster state** — call `get_cluster_resources()` to get `gpu_total`, `nodes_with_gpu`, `node_count`, and per-node details (`memory`, `cpu`, `gpus`)
2. **Estimate for target model** — call `estimate_resources(model="<bare-model-id>", quantization="bf16")` to get `gpu_memory_required`, `gpu_per_worker`, `total_gpu`, and `breakdown`
3. **Compare quantization options** — call `estimate_resources` again with `quantization="int4"` or `"int8"` to see the memory savings from QLoRA
4. **Multi-node planning** — set `num_workers` > 1 in `estimate_resources` to see distributed training requirements
5. **Check runtimes** — call `list_runtimes()` to see what training runtimes are available, then `get_runtime(name="<runtime>")` for detailed config

## Key Details

- `estimate_resources` uses bare model IDs (e.g. `google/gemma-2b`), not `hf://` prefixed URIs
- The `quantization` parameter accepts: `bf16`, `fp16`, `fp32`, `int4`, `int8`
- `int4` corresponds to QLoRA — typically cuts memory by ~4x vs bf16
- `get_cluster_resources` returns `gpu_total` across all nodes — check individual node `gpus` in the `nodes` list for per-node capacity
- If `gpu_total=0`, the cluster has no GPU nodes — inform the user before they attempt training
- To check if GPUs are in use, list running jobs with `list_training_jobs(status="Running")` and compare their GPU requests against `gpu_total`
- For gated models (Llama, Mistral), the HF Hub lookup in `estimate_resources` may require authentication
