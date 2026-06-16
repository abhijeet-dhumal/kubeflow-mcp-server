# Platform Fixes

Actionable JSON and commands for platform-specific issues.

---

## Platform Detection

| Node label | Platform |
|------------|----------|
| `node.openshift.io/os_id` | OpenShift |
| `eks.amazonaws.com/nodegroup` | Amazon EKS |
| `cloud.google.com/gke-nodepool` | Google GKE |
| None of above | Vanilla K8s / bare-metal |

Error-based detection (from `get_training_logs()`):

| Error pattern | Platform | Fix |
|---------------|----------|-----|
| "Read-only file system" | OpenShift (restricted SCC) | Add emptyDir volumes below |
| "Permission denied" on /.local or /.cache | OpenShift (restricted SCC) | Add emptyDir volumes below |

## OpenShift emptyDir Volumes

ALWAYS pass when `platform=openshift`. Copy-paste ready — pass directly as `volumes` parameter to any training tool:

```json
"volumes": [
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp", "mount_path": "/tmp", "empty_dir": {}},
  {"name": "home", "mount_path": "/home", "empty_dir": {}}
]
```

**Rules**:
- `fine_tune()`: Do NOT add workspace emptyDir — `/workspace` comes from the runtime PVC
- `run_custom_training()`: **MUST always pass volumes** — this triggers the workspace emptyDir auto-injection at `/workspace`, which is required for the script entrypoint to write the training script file (the entrypoint writes to the container CWD; without volumes the CWD is read-only on OpenShift)
- `run_container_training()`: add workspace emptyDir only if your image writes to `/workspace`

## run_custom_training() on OpenShift — Known Gaps

### Gap 1: `packages` parameter fails (Permission denied on `/.local`)

The pre-script pip install step uses `pip install --user` which writes to `/.local`. On OpenShift, the emptyDir volumes defined via the `volumes` parameter are **not injected as volumeMounts** into the training container — only the auto-injected `/workspace` emptyDir is mounted. This means `/.local` remains read-only even when a `dot-local` volume is defined.

**Workaround — do NOT use `packages`. Install inside the script:**

```python
import subprocess, sys, os

lib_dir = '/workspace/lib'
os.makedirs(lib_dir, exist_ok=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    '--target', lib_dir, '--quiet',
    'transformers', 'peft', 'trl', 'datasets', 'accelerate'
], check=True)
sys.path.insert(0, lib_dir)
```

This writes to `/workspace/lib` (the auto-injected writable PVC) and bypasses `/.local` entirely. Requires `KUBEFLOW_MCP_UNSAFE_SCRIPTS=true` in the server env (add to `.mcp.json` / `claude_desktop_config.json`).

### Gap 2: HuggingFace cache writes fail in `fine_tune()` initializer and node pods

**Status: RESOLVED in the MCP layer** — no workaround needed.

The MCP server automatically injects `HF_HOME=/workspace/.hf` into both the initializer pods (via `_HFModelInitializerWithHFHome` / `_HFDatasetInitializerWithHFHome` subclasses) and the training node pod (via `spec.trainer.env`). The `/workspace` PVC is always writable under OpenShift restricted SCC.

You do NOT need to set `HF_HOME` manually or fall back to `run_custom_training()`.

### Gap 3: `alpaca_cleaned_dataset` HF URI error with local dataset files

**Status: RESOLVED in the MCP layer** — no workaround needed.

The Kubeflow Trainer SDK generates `dataset.data_dir=/workspace/dataset/.` for top-level HF dataset URIs. The torchtune `alpaca_cleaned_dataset` passes this as a path component inside HF URIs, causing:

```
HfUriError: Invalid HF URI 'hf://datasets/yahma/alpaca-cleaned@rev//workspace/dataset/...'
```

The MCP server monkey-patches `get_trainer_cr_from_builtin_trainer` to:
1. Strip the trailing `/.` from `dataset.data_dir`
2. Append `dataset.source=<local_path>` — overrides the hardcoded HF source in `alpaca_cleaned_dataset`, making `load_dataset()` use the local PVC directory
3. Append `dataset.data_dir=null` — clears the `data_dir` so the datasets library doesn't raise "data_dir must be relative to a dataset directory's root"

These patches are transparent — `fine_tune()` works as-is on OpenShift with a torchtune runtime.

---

## `fine_tune()` on OpenShift — Verified Working Example

The following call works end-to-end on OpenShift with the `torchtune-*` runtimes. Copy-paste ready:

```json
{
  "runtime": "torchtune-qwen2.5-1.5b",
  "model": "hf://Qwen/Qwen2.5-1.5B-Instruct",
  "dataset": "hf://yahma/alpaca-cleaned",
  "namespace": "oss-summit-demo",
  "epochs": 1,
  "batch_size": 4,
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_attn_modules": ["q_proj", "v_proj", "output_proj"],
  "apply_lora_to_mlp": true,
  "volumes": [
    {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
    {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
    {"name": "tmp-vol", "mount_path": "/tmp", "empty_dir": {}}
  ],
  "confirmed": true
}
```

**What the MCP layer handles automatically (no action needed):**
- `HF_HOME=/workspace/.hf` on all pods (initializers + node)
- `dataset.source` / `dataset.data_dir` OmegaConf overrides to load from local PVC

**What you must always provide:**
- `volumes` with `/.local`, `/.cache`, `/.tmp` emptyDirs — required by OpenShift restricted SCC for non-HF temp writes
- A `torchtune-*` runtime (not `torch-distributed`) for `fine_tune()`
- `namespace` matching your project (default: `oss-summit-demo` via `KUBEFLOW_MCP_DEFAULT_NAMESPACE`)

**To cap training time** (e.g. smoke test in ~5 min):

Add to the call above:
```json
"dataset_preprocess_config": {"max_steps_per_epoch": 10}
```

Or use `max_steps_per_epoch` if the tool schema exposes it directly.

## OpenShift Non-Root UID

Random UIDs (e.g. 1000660000). Do NOT assume root. Implications:
- HOME may not be writable — use `/tmp` for outputs
- `/.local` emptyDir fixes most pip issues
- Avoid `chmod` / `chown` in training scripts

## GPU Tolerations

Pass as a direct `tolerations` parameter to any training tool:

```json
"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
```

Or shorthand: `gpu_per_node=1` (auto-maps to `nvidia.com/gpu`).

## NCCL Environment (Multi-Node GPU)

For `run_custom_training()` or `run_container_training()`, pass `env` directly:

```json
"env": {
  "NCCL_DEBUG": "INFO",
  "NCCL_P2P_DISABLE": "1",
  "NCCL_TIMEOUT": "1800"
}
```

For `fine_tune()`: the `env` parameter is NOT supported. If you need NCCL tuning, use `run_custom_training()` with a LoRA script and pass env vars there (see trainer://guides/training-patterns).

## OpenShift SCC Workarounds

SCC (Security Context Constraints) cannot be changed via MCP tools. Instead of escalating privileges, use tool parameters to work around the `restricted` SCC:

**1. Read-only filesystem** — pass writable emptyDir volumes to the training tool:

```json
"volumes": [
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp", "mount_path": "/tmp", "empty_dir": {}},
  {"name": "home", "mount_path": "/home", "empty_dir": {}}
]
```

For `fine_tune()`, these volumes apply to ALL replicated jobs (node, dataset-initializer, model-initializer).

**2. Non-root UID issues** — avoid commands that assume root. In scripts:
- Write outputs to `/tmp` instead of `/root` or `/home`
- Do NOT use `chmod`, `chown`, or write to `/etc`
- Use `/.local` for pip user installs (already covered by emptyDir above)

**3. Network restrictions** — some SCCs block host networking. For multi-node training, pass env vars to `run_custom_training()` or `run_container_training()`:

```json
"env": {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
```

**4. If emptyDirs are not enough** — escalate to cluster admin:

```bash
oc adm policy add-scc-to-user anyuid -z <service-account> -n <namespace>
```

This is a cluster-level change and cannot be done through MCP tools. Inform the user.
