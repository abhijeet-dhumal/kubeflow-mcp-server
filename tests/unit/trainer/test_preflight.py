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

from unittest.mock import patch

from kubeflow_mcp.trainer.api.planning import estimate_resources, pre_flight


def test_preflight_surfaces_estimate_error_fields():
    with (
        patch(
            "kubeflow_mcp.trainer.api.planning.check_compatibility",
            return_value={"data": {"platform": "kubernetes", "blockers": []}},
        ),
        patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources",
            return_value={"data": {"gpu_total": 4}},
        ),
        patch(
            "kubeflow_mcp.trainer.api.planning.estimate_resources",
            return_value={
                "success": False,
                "error": "Invalid HuggingFace model ID format: 'gemma4:e4b'",
                "hint": "Use 'google/gemma-2-2b'",
            },
        ),
        patch(
            "kubeflow_mcp.trainer.api.discovery.list_runtimes",
            return_value={"data": {"runtimes": [{"name": "torch-default-gpu"}]}},
        ),
    ):
        result = pre_flight(model="gemma4:e4b")

    assert result["success"] is True
    data = result["data"]
    assert data["estimate_status"] == "error"
    assert "Invalid HuggingFace model ID format" in data["estimate_error"]
    assert data["estimate_hint"] == "Use 'google/gemma-2-2b'"
    assert any("Estimate failed:" in step for step in data["next_steps"])


def test_estimate_resources_invalid_id_includes_suggestions():
    with patch(
        "kubeflow_mcp.trainer.api.planning._suggest_hf_model_ids",
        return_value=["google/gemma-2-2b"],
    ):
        result = estimate_resources("hf://gemma4:e4b")

    assert result["success"] is False
    assert "Invalid HuggingFace model ID format" in result["error"]
    assert "google/gemma-2-2b" in (result.get("hint") or "")
