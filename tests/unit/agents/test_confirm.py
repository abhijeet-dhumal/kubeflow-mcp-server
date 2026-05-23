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

from kubeflow_mcp.agents.frameworks._confirm import set_confirm_handler, wrap_with_confirm
from kubeflow_mcp.agents.frameworks._observability import is_local_ollama_model


def test_wrap_with_confirm_approves_and_sets_flag():
    calls: list[tuple[str, dict]] = []

    def approve(name: str, args: dict) -> bool:
        calls.append((name, args))
        return True

    set_confirm_handler(approve)

    def mutating(*, confirmed: bool = False) -> dict:
        return {"confirmed": confirmed}

    wrapped = wrap_with_confirm(mutating)
    result = wrapped(confirmed=False)

    assert result == {"confirmed": True}
    assert calls == [("mutating", {"confirmed": False})]


def test_wrap_with_confirm_declines():
    set_confirm_handler(lambda _n, _a: False)

    def mutating(*, confirmed: bool = False) -> dict:
        return {"confirmed": confirmed}

    result = wrap_with_confirm(mutating)(confirmed=False)
    assert result["cancelled"] is True


def test_is_local_ollama_model():
    assert is_local_ollama_model("ollama/qwen3:8b")
    assert not is_local_ollama_model("gpt-4.1")


def test_wrap_with_confirm_execute_tool_accepts_string_arguments():
    set_confirm_handler(lambda _n, _a: True)
    calls: list[tuple[str, dict]] = []

    def execute_tool(*, tool_name: str, arguments: dict | None = None) -> dict:
        calls.append((tool_name, arguments or {}))
        return {"ok": True}

    wrapped = wrap_with_confirm(execute_tool)
    result = wrapped(tool_name="pre_flight", arguments='{"model": "google/gemma-2-2b"}')

    assert result == {"ok": True}
    assert calls == [("pre_flight", {"model": "google/gemma-2-2b"})]
