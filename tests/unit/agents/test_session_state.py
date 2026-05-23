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

import json

from kubeflow_mcp.agents.runtime.session_state import (
    build_session_snapshot,
    export_session_snapshot,
    import_session_snapshot,
    reset_token_totals,
)


def test_build_session_snapshot_includes_common_fields():
    payload = build_session_snapshot(
        model="ollama/qwen3:8b",
        framework="langchain",
        tool_mode="full",
        token_input=12,
        token_output=34,
        extra={"chat_history": {"messages": []}},
    )
    assert payload["model"] == "ollama/qwen3:8b"
    assert payload["framework"] == "langchain"
    assert payload["tool_mode"] == "full"
    assert payload["tokens"] == {"input": 12, "output": 34}
    assert payload["chat_history"] == {"messages": []}


def test_export_session_snapshot_writes_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = build_session_snapshot(
        model="test-model",
        framework="llamaindex",
        tool_mode="semantic",
        token_input=1,
        token_output=2,
    )
    output_path = export_session_snapshot(payload)
    exported = json.loads((tmp_path / output_path).read_text())
    assert exported == payload


def test_reset_token_totals_resets_common_fields():
    class Tracker:
        session_input = 77
        session_output = 88

    tracker = Tracker()
    reset_token_totals(tracker)
    assert tracker.session_input == 0
    assert tracker.session_output == 0


def test_import_session_snapshot_reads_payload(tmp_path):
    path = tmp_path / "session.json"
    path.write_text('{"model":"m1","messages":[]}')
    payload = import_session_snapshot(str(path))
    assert payload["model"] == "m1"

