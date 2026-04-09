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

"""Training tools for LLM fine-tuning and custom training.

Maps to TrainerClient.train() (SDK 0.4.0) with different configurations:
- fine_tune() → HuggingFace/S3 model fine-tuning with BuiltinTrainer or CustomTrainer
- run_custom_training() → User-provided training script with CustomTrainer
- run_container_training() → Pre-built container image with CustomTrainerContainer
"""

from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.common.utils import get_trainer_client
from kubeflow_mcp.core.security import is_safe_python_code, validate_k8s_name

# Import Kubeflow SDK types at module level to avoid import deadlocks
# when tools are called in rapid succession
try:
    from kubeflow.trainer.options import (  # type: ignore[attr-defined]
        Annotations,
        ContainerPatch,
        JobSetSpecPatch,
        JobSetTemplatePatch,
        JobSpecPatch,
        JobTemplatePatch,
        Labels,
        PodSpecPatch,
        PodTemplatePatch,
        ReplicatedJobPatch,
        RuntimePatch,
        TrainingRuntimeSpecPatch,
    )
    from kubeflow.trainer.types.types import (
        BuiltinTrainer,
        CustomTrainer,
        CustomTrainerContainer,
        DataType,
        HuggingFaceDatasetInitializer,
        HuggingFaceModelInitializer,
        Initializer,
        LoraConfig,
        S3DatasetInitializer,
        S3ModelInitializer,
        TorchTuneConfig,
    )

    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def _sdk_error(e: Exception, hint: str | None = None) -> dict[str, Any]:
    """Convert an exception into a ToolError dict with optional K8s response detail."""
    details: dict[str, Any] | None = None
    if e.__cause__:
        details = {"cause": str(e.__cause__)}
    elif hasattr(e, "response"):
        try:
            details = {"response": e.response.text}  # type: ignore[union-attr]
        except Exception:
            pass
    return ToolError(
        error=str(e),
        error_code=ErrorCode.SDK_ERROR,
        details=details,
        hint=hint,
    ).model_dump()


def _build_initializer(
    model: str,
    dataset: str,
    hf_token: str | None = None,
    s3_endpoint: str | None = None,
    s3_access_key_id: str | None = None,
    s3_secret_access_key: str | None = None,
    s3_region: str | None = None,
    s3_role_arn: str | None = None,
) -> "Initializer":
    """Build an Initializer from model/dataset URIs.

    Auto-detects the storage backend from the URI prefix:
    - ``hf://`` → HuggingFace initializers (access_token used for gated models)
    - ``s3://`` → S3 initializers (endpoint/key/secret/region/role_arn used)
    """
    s3_kwargs: dict[str, Any] = {}
    if s3_endpoint:
        s3_kwargs["endpoint"] = s3_endpoint
    if s3_access_key_id:
        s3_kwargs["access_key_id"] = s3_access_key_id
    if s3_secret_access_key:
        s3_kwargs["secret_access_key"] = s3_secret_access_key
    if s3_region:
        s3_kwargs["region"] = s3_region
    if s3_role_arn:
        s3_kwargs["role_arn"] = s3_role_arn

    if model.startswith("s3://"):
        model_init: Any = S3ModelInitializer(storage_uri=model, **s3_kwargs)
    else:
        model_init = HuggingFaceModelInitializer(
            storage_uri=model, access_token=hf_token
        )

    if dataset.startswith("s3://"):
        dataset_init: Any = S3DatasetInitializer(storage_uri=dataset, **s3_kwargs)
    else:
        dataset_init = HuggingFaceDatasetInitializer(
            storage_uri=dataset, access_token=hf_token
        )

    return Initializer(model=model_init, dataset=dataset_init)


def _build_runtime_patch(
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    env: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
) -> list[Any]:
    """Build runtime patch options for the SDK.

    Returns a list of options to pass to ``client.train(options=...)``.
    Returns an empty list if no patches are specified.

    Top-level options (``Labels``, ``Annotations``) are appended alongside
    the ``RuntimePatch`` in the returned list so they are applied independently.
    """
    has_pod_patch = any(
        [
            node_selector,
            tolerations,
            env,
            volumes,
            volume_mounts,
            affinity,
            service_account_name,
            image_pull_secrets,
        ]
    )
    has_meta = labels or annotations

    if not has_pod_patch and not has_meta:
        return []

    if not _SDK_AVAILABLE:
        return []

    options: list[Any] = []

    if has_pod_patch:
        containers = None
        if env or volume_mounts:
            containers = [
                ContainerPatch(
                    name="trainer",
                    env=env,
                    volume_mounts=volume_mounts,
                )
            ]

        pod_spec = PodSpecPatch(
            node_selector=node_selector,
            tolerations=tolerations,
            volumes=volumes,
            containers=containers,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
        )

        options.append(
            RuntimePatch(
                training_runtime_spec=TrainingRuntimeSpecPatch(
                    template=JobSetTemplatePatch(
                        spec=JobSetSpecPatch(
                            replicated_jobs=[
                                ReplicatedJobPatch(
                                    name="node",
                                    template=JobTemplatePatch(
                                        spec=JobSpecPatch(
                                            template=PodTemplatePatch(spec=pod_spec),
                                        ),
                                    ),
                                ),
                            ],
                        ),
                    ),
                ),
            )
        )

    if labels:
        options.append(Labels(labels=labels))
    if annotations:
        options.append(Annotations(annotations=annotations))

    return options


def fine_tune(
    model: str,
    dataset: str,
    runtime: str = "torch-tune",
    hf_token: str | None = None,
    batch_size: int = 4,
    epochs: int = 1,
    num_nodes: int = 1,
    dtype: str | None = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float | None = None,
    use_dora: bool | None = None,
    quantize_base: bool | None = None,
    s3_endpoint: str | None = None,
    s3_access_key_id: str | None = None,
    s3_secret_access_key: str | None = None,
    s3_region: str | None = None,
    s3_role_arn: str | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    env: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Fine-tune a model using LoRA/QLoRA with torchtune.

    Supports HuggingFace (``hf://``) and S3 (``s3://``) model/dataset sources.
    Requires ``confirmed=True`` to submit. First call returns a preview.

    Args:
        model: Model URI. Use ``hf://`` prefix for HuggingFace (e.g.,
            ``hf://google/gemma-2b``) or ``s3://`` prefix for S3.
        dataset: Dataset URI. Same prefix rules as ``model``.
        runtime: ClusterTrainingRuntime name. Defaults to ``torch-tune``.
        hf_token: HuggingFace access token for gated models (Llama, Mistral).
        batch_size: Per-GPU batch size. Defaults to 4.
        epochs: Number of training epochs. Defaults to 1.
        num_nodes: Distributed training nodes. Defaults to 1.
        dtype: Training precision — ``"bf16"`` or ``"fp32"``. Uses runtime
            default if not specified.
        lora_rank: LoRA rank. Defaults to 8.
        lora_alpha: LoRA alpha scaling. Defaults to 16.
        lora_dropout: LoRA dropout probability (0.0–1.0). SDK default if omitted.
        use_dora: Enable DoRA (weight-decomposed LoRA). SDK default if omitted.
        quantize_base: Quantize base model weights for QLoRA. SDK default if omitted.
        s3_endpoint: S3-compatible endpoint URL (MinIO, Ceph, etc.).
        s3_access_key_id: S3 access key ID.
        s3_secret_access_key: S3 secret access key.
        s3_region: S3 region.
        s3_role_arn: IAM role ARN for S3 access.
        node_selector: K8s node selector (e.g., ``{"gpu-type": "a100"}``).
        tolerations: K8s tolerations for tainted nodes.
        env: Additional pod environment variables as list of K8s env var dicts.
        volumes: K8s volume definitions.
        volume_mounts: K8s volume mounts.
        affinity: K8s pod affinity/anti-affinity rules.
        service_account_name: K8s service account for the training pod.
        image_pull_secrets: K8s image pull secrets.
        labels: Extra labels to apply to the TrainJob.
        annotations: Extra annotations to apply to the TrainJob.
        confirmed: Set ``True`` to submit job. ``False`` returns preview only.

    Returns:
        dict: If ``confirmed=False``: preview with ``config`` dict.
            If ``confirmed=True``: ``job_name``, ``status``, ``message``.

    Example:
        >>> fine_tune("hf://google/gemma-2b", "hf://tatsu-lab/alpaca", confirmed=True)
        {"data": {"job_name": "train-gemma-abc", "status": "Created"}}

    Note:
        Call ``get_cluster_resources()`` first to verify GPU availability.
        For QLoRA set ``quantize_base=True``. For DoRA set ``use_dora=True``.
    """
    try:
        if not _SDK_AVAILABLE:
            return ToolError(
                error="Kubeflow SDK not available",
                error_code=ErrorCode.SDK_ERROR,
            ).model_dump()

        if dtype and dtype not in ("bf16", "fp32"):
            return ToolError(
                error=f"Invalid dtype '{dtype}'. Must be 'bf16' or 'fp32'.",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()

        config: dict[str, Any] = {
            "model": model,
            "dataset": dataset,
            "runtime": runtime,
            "hf_token": "***" if hf_token else None,
            "batch_size": batch_size,
            "epochs": epochs,
            "num_nodes": num_nodes,
            "dtype": dtype,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "use_dora": use_dora,
            "quantize_base": quantize_base,
        }
        if s3_access_key_id:
            config["s3_access_key_id"] = "***"
        if s3_secret_access_key:
            config["s3_secret_access_key"] = "***"
        for k, v in [
            ("s3_endpoint", s3_endpoint),
            ("s3_region", s3_region),
            ("s3_role_arn", s3_role_arn),
            ("node_selector", node_selector),
            ("tolerations", tolerations),
            ("env", env),
            ("volumes", volumes),
            ("volume_mounts", volume_mounts),
            ("affinity", affinity),
            ("service_account_name", service_account_name),
            ("image_pull_secrets", image_pull_secrets),
            ("labels", labels),
            ("annotations", annotations),
        ]:
            if v:
                config[k] = v

        hf_compatible_runtimes = ["torch-tune", "torchtune"]
        use_initializer_pattern = runtime in hf_compatible_runtimes

        if use_initializer_pattern:
            config["mode"] = "builtin_trainer"
            config["note"] = "Using BuiltinTrainer with TorchTuneConfig + Initializers"
        else:
            config["mode"] = "custom_trainer"
            config["note"] = (
                f"Runtime '{runtime}' doesn't have initializer jobs. "
                "Using CustomTrainer with packages_to_install instead."
            )

        if not confirmed:
            return PreviewResponse(
                message="Review config and set confirmed=True to submit job",
                config=config,
            ).model_dump()

        options = _build_runtime_patch(
            node_selector=node_selector,
            tolerations=tolerations,
            env=env,
            volumes=volumes,
            volume_mounts=volume_mounts,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
            labels=labels,
            annotations=annotations,
        )

        client = get_trainer_client()

        if use_initializer_pattern:
            initializer = _build_initializer(
                model=model,
                dataset=dataset,
                hf_token=hf_token,
                s3_endpoint=s3_endpoint,
                s3_access_key_id=s3_access_key_id,
                s3_secret_access_key=s3_secret_access_key,
                s3_region=s3_region,
                s3_role_arn=s3_role_arn,
            )

            lora_cfg = LoraConfig(
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
                quantize_base=quantize_base,
            )
            torch_cfg = TorchTuneConfig(
                batch_size=batch_size,
                epochs=epochs,
                num_nodes=num_nodes,
                dtype=DataType(dtype) if dtype else None,
                peft_config=lora_cfg,
            )

            trainer = BuiltinTrainer(config=torch_cfg)
            job_name = client.train(
                runtime=runtime,
                initializer=initializer,
                trainer=trainer,
                options=options if options else None,
            )
        else:
            model_id = model.removeprefix("hf://")
            dataset_id = dataset.removeprefix("hf://")
            _batch_size = batch_size
            _epochs = epochs
            _lora_rank = lora_rank
            _lora_alpha = lora_alpha

            def train_func() -> None:
                import os
                import subprocess

                from huggingface_hub import login, snapshot_download

                hf_tok = os.environ.get("HF_TOKEN")
                if hf_tok:
                    login(token=hf_tok)

                print(f"Downloading model: {model_id}")
                snapshot_download(model_id, local_dir="/workspace/model")

                print(f"Downloading dataset: {dataset_id}")
                from datasets import load_dataset

                ds = load_dataset(dataset_id)
                ds.save_to_disk("/workspace/dataset")

                print(
                    f"Starting fine-tuning — batch_size={_batch_size}, "
                    f"epochs={_epochs}, lora_rank={_lora_rank}, lora_alpha={_lora_alpha}"
                )
                subprocess.run(  # noqa: S603
                    [
                        "tune",
                        "run",
                        "lora_finetune_single_device",
                        "--config",
                        "llama3_2/1B_lora_single_device",
                        "model.path=/workspace/model",
                        "dataset.source=/workspace/dataset",
                        f"batch_size={_batch_size}",
                        f"epochs={_epochs}",
                        f"lora_rank={_lora_rank}",
                        f"lora_alpha={_lora_alpha}",
                    ],
                    check=True,
                )

            trainer = CustomTrainer(  # type: ignore[assignment]
                func=train_func,
                packages_to_install=[
                    "torchtune",
                    "transformers",
                    "datasets",
                    "huggingface_hub",
                    "accelerate",
                ],
                num_nodes=num_nodes,
                env={"HF_TOKEN": hf_token} if hf_token else None,
            )
            job_name = client.train(
                runtime=runtime,
                trainer=trainer,
                options=options if options else None,
            )

        return ToolResponse(
            data={
                "job_name": job_name,
                "status": "Created",
                "message": f"Training job '{job_name}' submitted successfully",
            }
        ).model_dump()

    except Exception as e:
        return _sdk_error(
            e,
            hint="Use troubleshooting_guide prompt for diagnosis, or resource_planning to check requirements",
        )


def run_custom_training(
    script: str,
    name: str | None = None,
    num_nodes: int = 1,
    gpu_per_node: int = 1,
    packages: list[str] | None = None,
    image: str | None = None,
    pip_index_urls: list[str] | None = None,
    resources_per_node: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    pod_env: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Run a custom Python training script on the cluster.

    Script is validated for security before execution. The script runs inside
    a remote worker pod via ``exec()`` — the function is serialised by the SDK.

    Args:
        script: Python code string. Validated against dangerous operations.
        name: TrainJob name. Auto-generated if not provided.
        num_nodes: Distributed training nodes. Defaults to 1.
        gpu_per_node: GPUs per node. Set 0 for CPU-only. Defaults to 1.
        packages: Pip packages to install (e.g., ``["torch", "transformers"]``).
        image: Custom base container image for the training pod. Uses runtime
            default if omitted.
        pip_index_urls: Custom PyPI mirror URLs (e.g., internal Nexus/Artifactory).
        resources_per_node: Full resource dict for the training pod (e.g.,
            ``{"cpu": "8", "memory": "32Gi", "gpu": 2}``). When provided,
            overrides ``gpu_per_node``. When omitted, defaults to
            ``{"gpu": gpu_per_node}`` (or no resource constraints if
            ``gpu_per_node=0``).
        env: Environment variables injected directly into the training function
            (serialised into the closure by cloudpickle). Dict format:
            ``{"KEY": "VALUE"}``.
        node_selector: K8s node selector applied to the pod.
        tolerations: K8s tolerations for tainted nodes.
        pod_env: Additional pod-level environment variables as a list of K8s
            env var dicts (e.g., ``[{"name": "XYZ", "value": "1"}]``).
            Distinct from ``env`` which is inlined into the training closure.
        volumes: K8s volume definitions.
        volume_mounts: K8s volume mounts.
        affinity: K8s pod affinity/anti-affinity rules.
        service_account_name: K8s service account for the training pod.
        image_pull_secrets: K8s image pull secrets.
        labels: Extra labels to apply to the TrainJob.
        annotations: Extra annotations to apply to the TrainJob.
        confirmed: Set ``True`` to submit. ``False`` returns preview.

    Returns:
        dict: If ``confirmed=False``: preview with truncated script.
            If ``confirmed=True``: ``job_name``, ``status``, ``message``.

    Note:
        Use ``run_container_training()`` for unrestricted script execution.
    """
    try:
        if not _SDK_AVAILABLE:
            return ToolError(
                error="Kubeflow SDK not available",
                error_code=ErrorCode.SDK_ERROR,
            ).model_dump()

        safe, reason = is_safe_python_code(script)
        if not safe:
            return ToolError(
                error=f"Script validation failed: {reason}",
                error_code=ErrorCode.VALIDATION_ERROR,
                hint="Use custom_training_workflow prompt for secure script guidelines, or run_container_training for unrestricted access",
            ).model_dump()

        if name:
            err = validate_k8s_name(name)
            if err:
                return err.model_dump()

        effective_resources = resources_per_node or (
            {"gpu": gpu_per_node} if gpu_per_node > 0 else None
        )

        config: dict[str, Any] = {
            "script": script[:200] + "..." if len(script) > 200 else script,
            "name": name,
            "num_nodes": num_nodes,
            "gpu_per_node": gpu_per_node,
            "packages": packages or [],
            "image": image,
            "pip_index_urls": pip_index_urls or [],
            "resources_per_node": effective_resources,
        }
        for k, v in [
            ("env", env),
            ("node_selector", node_selector),
            ("tolerations", tolerations),
            ("pod_env", pod_env),
            ("volumes", volumes),
            ("volume_mounts", volume_mounts),
            ("affinity", affinity),
            ("service_account_name", service_account_name),
            ("image_pull_secrets", image_pull_secrets),
            ("labels", labels),
            ("annotations", annotations),
        ]:
            if v:
                config[k] = v

        if not confirmed:
            return PreviewResponse(
                message="Review config and set confirmed=True to submit job",
                config=config,
            ).model_dump()

        _script = script

        def train_func() -> None:
            # Runs inside the remote training pod. exec() is intentional:
            # the script was already validated by is_safe_python_code() above.
            exec(compile(_script, "<training_script>", "exec"), {})  # noqa: S102

        trainer = CustomTrainer(
            func=train_func,
            packages_to_install=packages,
            image=image,
            pip_index_urls=pip_index_urls or [],
            num_nodes=num_nodes,
            resources_per_node=effective_resources,
            env=env,
        )

        options = _build_runtime_patch(
            node_selector=node_selector,
            tolerations=tolerations,
            env=pod_env,
            volumes=volumes,
            volume_mounts=volume_mounts,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
            labels=labels,
            annotations=annotations,
        )

        client = get_trainer_client()
        job_name = client.train(trainer=trainer, options=options if options else None)

        return ToolResponse(
            data={
                "job_name": job_name,
                "status": "Created",
                "message": f"Custom training job '{job_name}' submitted",
            }
        ).model_dump()

    except Exception as e:
        return _sdk_error(e, hint="Use troubleshooting_guide prompt for diagnosis")


def run_container_training(
    image: str,
    command: list[str] | None = None,
    num_nodes: int = 1,
    gpu_per_node: int = 1,
    resources_per_node: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Run training with a pre-built container image.

    No script validation — full control via container ENTRYPOINT/CMD.

    Args:
        image: Container image (e.g., ``pytorch/pytorch:2.0-cuda11.8``).
        command: Override container command (baked into image if omitted).
        num_nodes: Distributed training nodes. Defaults to 1.
        gpu_per_node: GPUs per node. Set 0 for CPU-only. Defaults to 1.
        resources_per_node: Full resource dict (e.g.,
            ``{"cpu": "8", "memory": "32Gi", "gpu": 2}``). Overrides
            ``gpu_per_node`` when provided.
        env: Environment variables as dict (e.g., ``{"HF_TOKEN": "xxx"}``).
        node_selector: K8s node selector.
        tolerations: K8s tolerations.
        volumes: K8s volume definitions.
        volume_mounts: K8s volume mounts.
        affinity: K8s pod affinity/anti-affinity rules.
        service_account_name: K8s service account for the training pod.
        image_pull_secrets: K8s image pull secrets.
        labels: Extra labels to apply to the TrainJob.
        annotations: Extra annotations to apply to the TrainJob.
        confirmed: Set ``True`` to submit. ``False`` returns preview.

    Returns:
        dict: If ``confirmed=False``: preview with config.
            If ``confirmed=True``: ``job_name``, ``status``, ``message``.
    """
    try:
        if not _SDK_AVAILABLE:
            return ToolError(
                error="Kubeflow SDK not available",
                error_code=ErrorCode.SDK_ERROR,
            ).model_dump()

        effective_resources = resources_per_node or (
            {"gpu": gpu_per_node} if gpu_per_node > 0 else None
        )

        config: dict[str, Any] = {
            "image": image,
            "command": command,
            "num_nodes": num_nodes,
            "gpu_per_node": gpu_per_node,
            "resources_per_node": effective_resources,
            "env": env,
        }
        for k, v in [
            ("node_selector", node_selector),
            ("tolerations", tolerations),
            ("volumes", volumes),
            ("volume_mounts", volume_mounts),
            ("affinity", affinity),
            ("service_account_name", service_account_name),
            ("image_pull_secrets", image_pull_secrets),
            ("labels", labels),
            ("annotations", annotations),
        ]:
            if v:
                config[k] = v

        if not confirmed:
            return PreviewResponse(
                message="Review config and set confirmed=True to submit job",
                config=config,
            ).model_dump()

        # Note: SDK 0.4.0 CustomTrainerContainer doesn't support custom command;
        # it must be baked into the container image's ENTRYPOINT/CMD.
        trainer = CustomTrainerContainer(
            image=image,
            num_nodes=num_nodes,
            resources_per_node=effective_resources,
            env=env,
        )

        # env is already on CustomTrainerContainer; don't repeat it via RuntimePatch
        # (would duplicate env vars in the pod spec).
        options = _build_runtime_patch(
            node_selector=node_selector,
            tolerations=tolerations,
            volumes=volumes,
            volume_mounts=volume_mounts,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
            labels=labels,
            annotations=annotations,
        )

        client = get_trainer_client()
        job_name = client.train(trainer=trainer, options=options if options else None)

        return ToolResponse(
            data={
                "job_name": job_name,
                "status": "Created",
                "message": f"Container training job '{job_name}' submitted",
            }
        ).model_dump()

    except Exception as e:
        return _sdk_error(e, hint="Use troubleshooting_guide prompt for diagnosis")
