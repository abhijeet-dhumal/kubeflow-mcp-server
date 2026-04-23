# Run Custom Training on Kubeflow

Use the kubeflow-mcp MCP server tools to submit a custom Python training script or a pre-built container image to the cluster.

## Option A: Python Script (`run_custom_training`)

1. **Write the script** — prepare a Python code string. The script runs inside a training pod with full privileges. Use `os.environ` to read hyperparameters so you can iterate without rewriting the script.
2. **Check resources** — call `get_cluster_resources()` and `estimate_resources(...)` if using a known model
3. **Preview** — call `run_custom_training(script="...", num_nodes=1, gpu_per_node=1, packages=["torch","transformers"], confirmed=false)` — the server runs a heuristic safety scan and returns a preview
4. **Submit** — call again with `confirmed=true`
5. **Monitor** — use `get_training_logs(name="<job-name>")` and `wait_for_training(name="<job-name>")`

### Iterative tuning via `env`

Keep the script constant, vary hyperparameters through `env`:

```python
run_custom_training(
    script="import os; lr=float(os.environ['LR']); ...",
    env={"LR": "3e-4", "EPOCHS": "5"},
    confirmed=false
)
```

Re-submit with different `env` values without re-validating the script.

## Option B: Container Image (`run_container_training`)

1. **Preview** — call `run_container_training(image="my-registry/my-trainer:v1", num_nodes=2, gpu_per_node=4, confirmed=false)`
2. **Submit** — call again with `confirmed=true`
3. **Monitor** — same as above

### When to use which

- **`run_custom_training`** — quick prototyping, ad-hoc scripts, no Docker build needed
- **`run_container_training`** — production images, complex dependencies, reproducible environments
