# Troubleshoot a Failed Training Job

Use the kubeflow-mcp MCP server tools to diagnose and recover from training failures.

## Steps

1. **Get job status** — call `get_training_job(name="<job-name>")` and check the `status` field
2. **Pull logs** — call `get_training_logs(name="<job-name>")` — the response includes a `failure_hint` field with auto-detected root cause (OOM, missing module, NCCL timeout, image pull, etc.)
3. **Check events** — call `get_training_events(name="<job-name>")` for K8s-level issues (scheduling failures, quota exceeded, node pressure, image pull backoff)
4. **Cross-reference with cluster** — call `get_cluster_resources()` to verify GPU/memory availability vs. what the job requested
5. **Verify runtime** — if the hint mentions missing packages, call `get_runtime_packages(name="<runtime>")` to see what's installed
6. **Fix and retry** — delete the failed job with `delete_training_job(name="<job-name>", confirmed=true)`, then resubmit with corrected parameters

## Common Failure Patterns

- **OOM** → reduce `batch_size`, enable `quantize_base=true` (QLoRA), or request more GPU memory via `resources_per_node`
- **NCCL timeout** → check node count, network policy, and GPU driver compatibility
- **Missing module** → add the package to `packages` in `run_custom_training`, or switch to a runtime that includes it
- **ImagePullBackOff** → wrong image tag, missing `image_pull_secrets`, or registry auth issue
- **Quota exceeded** → job requests more resources than the namespace `ResourceQuota` allows
