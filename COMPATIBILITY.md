# Compatibility Matrix

## Version Support

| MCP Server | Kubeflow SDK | Python      | Kubernetes |
|------------|-------------|-------------|------------|
| 0.1.x      | ≥ 0.4.0     | 3.10 – 3.12 | ≥ 1.27     |

The `kubeflow` package bundles all SDK clients (`TrainerClient`, `OptimizerClient`, etc.).
Each MCP client extra installs it:

```toml
[project.optional-dependencies]
trainer   = ["kubeflow>=0.4.0"]
optimizer = ["kubeflow>=0.4.0"]
hub       = ["kubeflow>=0.4.0", "model-registry>=0.3.6"]
```

---

## TrainerClient API Coverage

### SDK Types Used

| SDK Type | Used In | Notes |
|----------|---------|-------|
| `BuiltinTrainer` | `fine_tune` | LoRA via `TorchTuneConfig` or HuggingFace `LoraConfig` |
| `CustomTrainer` | `run_custom_training` | Inline Python script, serialized via `cloudpickle` |
| `CustomTrainerContainer` | `run_container_training` | Pre-built OCI image |
| `Initializer` | `fine_tune` | Wraps model + dataset initializers |
| `HuggingFaceModelInitializer` | `fine_tune` | Auto-selected for `hf://` URIs |
| `S3ModelInitializer` | `fine_tune` | Auto-selected for `s3://` model URIs |
| `S3DatasetInitializer` | `fine_tune` | Auto-selected for `s3://` dataset URIs |
| `LoraConfig` | `fine_tune` | Full field coverage: rank, alpha, dropout, DoRA, QLoRA |
| `TorchTuneConfig` | `fine_tune` | Config path + override args |
| `DataType` | `fine_tune` | `bf16` / `fp32` via `dtype` parameter |
| `RuntimePatch` | all training tools | Node selector, tolerations, affinity |
| `ContainerPatch` | all training tools | Env vars, resource overrides |
| `PodSpecPatch` | all training tools | Service account, image pull secrets |
| `Labels` | all training tools | Job-level label injection |
| `Annotations` | all training tools | Job-level annotation injection |

### SDK Types Not Exposed

| SDK Type | Reason |
|----------|--------|
| `JobSetTemplatePatch` | Low-level K8s construct; use `RuntimePatch` instead |
| `ReplicatedJobPatch` | Same as above |

### TrainerClient Methods

| SDK Method | MCP Tool | Notes |
|-----------|----------|-------|
| `client.train(trainer=BuiltinTrainer(...))` | `fine_tune` | |
| `client.train(trainer=CustomTrainer(...))` | `run_custom_training` | |
| `client.train(trainer=CustomTrainerContainer(...))` | `run_container_training` | |
| `client.get_job()` | `get_training_job` | |
| `client.list_jobs()` | `list_training_jobs` | Supports namespace + runtime + status filters |
| `client.delete_job()` | `delete_training_job` | |
| `client.suspend_job()` | `suspend_training_job` | |
| `client.resume_job()` | `resume_training_job` | |
| `client.get_job_logs(follow=False)` | `get_training_logs` | Streaming (`follow=True`) not exposed |
| `client.wait_for_job_status()` | `wait_for_training` | Accepts single status or list |
| `client.list_runtimes()` | `list_runtimes` | |
| `client.get_runtime()` | `get_runtime` | Returns framework + replicated job details |

### Kubernetes API (via `CoreV1Api`)

| K8s API | MCP Tool | Notes |
|---------|----------|-------|
| `list_node()` | `get_cluster_resources` | GPU/CPU capacity aggregation |
| `list_namespaced_event()` | `get_training_events` | Filtered to TrainJob pods |

### Planning Tools (no SDK call)

| MCP Tool | Source |
|----------|--------|
| `estimate_resources` | HuggingFace Hub metadata + local heuristics |
| `get_cluster_resources` | Kubernetes node capacity via `CoreV1Api` |

---

## OptimizerClient API Coverage

Status: **Stub** — tools not yet implemented.

Target SDK client: `kubeflow.katib.KatibClient`

Planned tools: `create_optimization_job`, `list_optimization_jobs`,
`get_optimization_job`, `wait_for_optimization`, `delete_optimization_job`.

---

## ModelRegistry (Hub) API Coverage

Status: **Stub** — tools not yet implemented.

Target SDK: `model_registry` package (`model-registry>=0.3.6`).

Planned tools: `register_model`, `list_models`, `get_model`,
`list_model_versions`, `get_model_version`.

---

## PipelinesClient API Coverage

Status: **Planned** — not yet started.

Target SDK client: `kubeflow.pipelines.Client`

---

## SparkClient API Coverage

Status: **Planned** — not yet started.

Target SDK client: `kubeflow.spark.SparkClient`

---

## FeastClient API Coverage

Status: **Planned** — not yet started.

Target SDK client: `kubeflow.feast.FeastClient`
