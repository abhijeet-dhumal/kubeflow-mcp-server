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

"""SDK client factories with caching and timeout configuration.

Mirrors kubeflow SDK's client structure:
- TrainerClient from kubeflow.trainer
"""

from functools import lru_cache
from typing import TYPE_CHECKING

# Import at module level to avoid import deadlocks when tools are called rapidly
from kubeflow.trainer import TrainerClient

if TYPE_CHECKING:
    from kubernetes import client as k8s_client

K8S_TIMEOUT = 5


@lru_cache(maxsize=1)
def _get_api_client() -> "k8s_client.ApiClient":
    """Create and cache a single ApiClient with strict timeouts.

    Cached so all higher-level API objects share one connection pool.
    Call ``reset_clients()`` to force re-creation (e.g. after kubeconfig change).
    """
    from kubernetes import client, config

    config.load_config()
    configuration = client.Configuration.get_default_copy()
    configuration.retries = 1
    configuration.socket_options = None  # rely on OS defaults
    api_client = client.ApiClient(configuration)
    api_client.rest_client.pool_manager.connection_pool_kw["timeout"] = K8S_TIMEOUT
    return api_client


def get_core_v1_api() -> "k8s_client.CoreV1Api":
    """Get CoreV1Api backed by the shared, timeout-configured ApiClient."""
    from kubernetes import client

    return client.CoreV1Api(_get_api_client())


def get_custom_objects_api() -> "k8s_client.CustomObjectsApi":
    """Get CustomObjectsApi backed by the shared, timeout-configured ApiClient."""
    from kubernetes import client

    return client.CustomObjectsApi(_get_api_client())


def get_trainer_effective_namespace(namespace: str | None = None) -> str:
    """Namespace for TrainJob operations: explicit arg, then SDK backend, else ``default``.

    Aligns direct CustomObjects calls with :class:`TrainerClient` (Kubernetes backend).
    """
    if namespace:
        return namespace
    client = get_trainer_client()
    backend = client.backend
    ns = getattr(backend, "namespace", None)
    if ns is not None:
        return str(ns)
    return "default"


def get_trainer_custom_objects_api() -> "k8s_client.CustomObjectsApi":
    """CustomObjectsApi from the SDK Kubernetes backend when available.

    Falls back to :func:`get_custom_objects_api` for non-Kubernetes backends
    (e.g. local process), so suspend/resume still use a configured client.
    """
    client = get_trainer_client()
    backend = client.backend
    custom = getattr(backend, "custom_api", None)
    if custom is not None:
        return custom
    return get_custom_objects_api()


@lru_cache(maxsize=1)
def get_trainer_client() -> TrainerClient:
    """Get or create TrainerClient singleton.

    Uses default KubernetesBackendConfig with current kubeconfig context.
    """
    return TrainerClient()


def reset_clients() -> None:
    """Reset all cached clients (for testing or kubeconfig rotation)."""
    get_trainer_client.cache_clear()
    _get_api_client.cache_clear()
