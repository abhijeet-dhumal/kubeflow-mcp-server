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

"""Kubernetes authentication interface.

NOTE: This is an interface placeholder for Stage 1.
Actual implementation deferred to production hardening.
"""

from dataclasses import dataclass


@dataclass
class AuthContext:
    """Authentication context for requests."""

    user: str | None = None
    groups: list[str] | None = None
    impersonate: str | None = None


def get_auth_context() -> AuthContext:
    """Get current authentication context.

    Returns default context. Full implementation requires:
    - K8s service account token validation
    - User impersonation support
    - RBAC integration
    """
    return AuthContext()
