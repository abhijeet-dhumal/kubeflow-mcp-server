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

from kubeflow_mcp.agents.frameworks._thinking import (
    extract_thinking_delta,
    is_answer_content_token,
    thinking_completion_kwargs,
)


def test_thinking_disabled_returns_empty():
    assert thinking_completion_kwargs(enabled=False, model="ollama/qwen3:8b") == {}


def test_thinking_ollama_uses_think_only():
    kwargs = thinking_completion_kwargs(enabled=True, model="ollama/qwen3:8b")
    assert kwargs == {"think": True}
    assert "thinking" not in kwargs


def test_thinking_cloud_uses_thinking_dict():
    kwargs = thinking_completion_kwargs(enabled=True, model="gpt-4.1")
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "think" not in kwargs


def test_extract_thinking_delta_from_additional_kwargs():
    class Msg:
        additional_kwargs = {"reasoning_content": "hmm"}

    class Chunk:
        message = Msg()

    assert extract_thinking_delta("", Chunk()) == "hmm"


def test_extract_thinking_delta_from_list_token():
    token = [{"type": "thinking", "thinking": "step"}]
    assert extract_thinking_delta(token) == "step"


def test_is_answer_content_token():
    assert is_answer_content_token("Final Answer:")
    assert not is_answer_content_token([{"type": "thinking", "thinking": "x"}])
