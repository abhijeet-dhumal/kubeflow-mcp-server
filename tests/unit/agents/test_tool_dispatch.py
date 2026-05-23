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

from kubeflow_mcp.agents.core.tool_dispatch import (
    compact_execute_tool_result,
    normalize_execute_tool_args,
)


def test_normalize_execute_tool_decodes_string_arguments():
    normalized = normalize_execute_tool_args(
        {
            "tool_name": "pre_flight",
            "arguments": '{"model":"google/gemma-2-2b"}',
            "batch_size": 2,
        }
    )

    assert normalized == {
        "tool_name": "pre_flight",
        "arguments": {"model": "google/gemma-2-2b", "batch_size": 2},
    }


def test_normalize_execute_tool_extracts_function_form():
    normalized = normalize_execute_tool_args({"function": "pre_flight()", "model": "foo/bar"})
    assert normalized == {"tool_name": "pre_flight", "arguments": {"model": "foo/bar"}}


def test_compact_execute_tool_result_keeps_non_execute_tool_results():
    result = {"ok": True, "items": list(range(20))}
    assert compact_execute_tool_result("list_runtimes", result) == result

