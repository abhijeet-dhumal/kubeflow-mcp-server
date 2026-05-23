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

"""Shared execute_tool argument normalization and result compaction."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_call_name(value: str) -> str | None:
    """Parse tool names from strings like ``list_runtimes()`` or ``list_runtimes``."""
    raw = value.strip()
    match = re.match(r"^(\w+)\s*\(", raw)
    if match:
        return match.group(1)
    if re.fullmatch(r"\w+", raw):
        return raw
    return None


def normalize_execute_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize execute_tool payloads to ``{tool_name, arguments}``."""
    parsed_args = dict(args)
    raw_arguments = parsed_args.get("arguments")
    if isinstance(raw_arguments, str):
        try:
            decoded = json.loads(raw_arguments)
            parsed_args["arguments"] = decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            parsed_args["arguments"] = {}

    if "tool_name" in parsed_args and isinstance(parsed_args["tool_name"], str):
        normalized = dict(parsed_args)
        extras = {
            k: v for k, v in normalized.items() if k not in ("tool_name", "arguments")
        }
        if extras:
            merged_args: dict[str, Any] = {}
            if isinstance(normalized.get("arguments"), dict):
                merged_args.update(normalized["arguments"])
            merged_args.update(extras)
            normalized["arguments"] = merged_args
            for key in extras:
                normalized.pop(key, None)
        return normalized

    for key in ("tool_input", "input", "tool", "name", "function"):
        val = parsed_args.get(key)
        if isinstance(val, str):
            extracted = extract_call_name(val)
            if extracted:
                normalized: dict[str, Any] = {"tool_name": extracted}
                merged_args: dict[str, Any] = {}
                if isinstance(parsed_args.get("arguments"), dict):
                    merged_args.update(parsed_args["arguments"])
                extras = {
                    k: v for k, v in parsed_args.items() if k not in ("arguments", key)
                }
                if extras:
                    merged_args.update(extras)
                if merged_args:
                    normalized["arguments"] = merged_args
                return normalized

    return parsed_args


def _truncate_for_reasoning(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "<truncated>"

    if isinstance(value, dict):
        return {
            key: _truncate_for_reasoning(item, depth=depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, list):
        limit = 8
        trimmed = [_truncate_for_reasoning(item, depth=depth + 1) for item in value[:limit]]
        if len(value) > limit:
            trimmed.append(f"... (+{len(value) - limit} more)")
        return trimmed

    if isinstance(value, str) and len(value) > 500:
        return f"{value[:500]}... (+{len(value) - 500} chars)"

    return value


def compact_execute_tool_result(tool_name: str, result: Any) -> Any:
    """Compact high-volume execute_tool output before reasoning pass."""
    if tool_name != "execute_tool" or not isinstance(result, dict):
        return result
    compact = _truncate_for_reasoning(result)
    if isinstance(compact, dict):
        compact.setdefault(
            "_note",
            "Output compacted for reasoning. Use /tools + execute_tool for full details.",
        )
    return compact

