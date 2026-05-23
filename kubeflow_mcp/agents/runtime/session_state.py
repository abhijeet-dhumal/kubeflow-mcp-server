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

"""Shared helpers for REPL session export and reset."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_session_snapshot(
    *,
    model: str,
    framework: str,
    tool_mode: str,
    token_input: int,
    token_output: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard export payload for interactive sessions."""
    payload: dict[str, Any] = {
        "model": model,
        "framework": framework,
        "tool_mode": tool_mode,
        "tokens": {"input": token_input, "output": token_output},
    }
    if extra:
        payload.update(extra)
    return payload


def export_session_snapshot(payload: dict[str, Any], *, prefix: str = "session") -> str:
    """Write session payload to a timestamped JSON file and return path."""
    output_path = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as file_handle:
        json.dump(payload, file_handle, indent=2, default=str)
    return output_path


def reset_token_totals(tracker: Any) -> None:
    """Reset common session token counters when present."""
    if hasattr(tracker, "session_input"):
        tracker.session_input = 0
    if hasattr(tracker, "session_output"):
        tracker.session_output = 0


def import_session_snapshot(path: str) -> dict[str, Any]:
    """Load and validate a session payload from disk."""
    source = Path(path).expanduser()
    with source.open() as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in session file: {source}"
        raise ValueError(msg)
    return payload

