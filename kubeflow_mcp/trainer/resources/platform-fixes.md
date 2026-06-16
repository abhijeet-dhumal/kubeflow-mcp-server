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

### Gap 2: HuggingFace downloads fail in `fine_tune()` initializer pods

The `fine_tune()` initializer pods (`model-initializer`, `dataset-initializer`) also have the volumeMount gap — `hf://` downloads via `xet_get` fail with Permission denied because `~/.cache/huggingface` is not writable.

**Workaround — use `run_custom_training()` with a LoRA script instead of `fine_tune()` on OpenShift**, and set `HF_HOME=/workspace/.hf` via `env`:

```json
"env": {"HF_HOME": "/workspace/.hf"}
```

### Complete working template for OpenShift LoRA fine-tuning

```python
# run_custom_training() script — OpenShift safe
import subprocess, sys, os

lib_dir = '/workspace/lib'
os.makedirs(lib_dir, exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install',
    '--target', lib_dir, '--quiet',
    'transformers', 'peft', 'trl', 'datasets', 'accelerate'], check=True)
sys.path.insert(0, lib_dir)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

os.makedirs(os.environ['HF_HOME'], exist_ok=True)

model_name = 'YOUR_MODEL'  # e.g. 'Qwen/Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_cuda = torch.cuda.is_available()
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    trust_remote_code=True)

model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05, bias='none'))
model.print_trainable_parameters()

dataset = load_dataset('YOUR_DATASET', split='train')

SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir='/workspace/checkpoints', num_train_epochs=1,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=50,
        bf16=use_cuda, fp16=False, max_seq_length=512, report_to='none'),
    train_dataset=dataset, processing_class=tokenizer,
).train()
print('Training complete!')
```

Call with:
```json
{
  "runtime": "torch-distributed",
  "env": {"HF_HOME": "/workspace/.hf", "NCCL_P2P_DISABLE": "1"},
  "volumes": [
    {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
    {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
    {"name": "tmp", "mount_path": "/tmp", "empty_dir": {}},
    {"name": "home", "mount_path": "/home", "empty_dir": {}}
  ]
}
```

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
