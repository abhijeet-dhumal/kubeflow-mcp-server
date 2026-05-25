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

"""Tool schema helpers: Python callables → OpenAI JSON schema.

Extracted from litellm_agent.py so all framework adapters share one
implementation without importing the full LiteLLM agent module.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin, get_type_hints

_PRIMITIVE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type hint to an OpenAI JSON schema fragment."""
    origin = get_origin(annotation)

    is_union = origin is Union
    if not is_union and hasattr(types, "UnionType"):
        is_union = isinstance(annotation, types.UnionType)
    if is_union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _annotation_to_json_schema(non_none[0]) if non_none else {"type": "string"}

    if origin is list:
        item_args = get_args(annotation)
        item_schema = _annotation_to_json_schema(item_args[0]) if item_args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    if annotation in _PRIMITIVE_MAP:
        return {"type": _PRIMITIVE_MAP[annotation]}

    return {"type": "string"}


def build_tool_schema(fn: Callable[..., Any], description: str = "") -> dict[str, Any]:
    """Build an OpenAI function-calling tool schema from a Python callable.

    Args:
        fn: The function to describe.
        description: Override description (falls back to first docstring line).

    Returns:
        OpenAI ``{"type": "function", "function": {...}}`` schema dict.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name in ("_meta", "ctx", "self"):
            continue
        prop = _annotation_to_json_schema(hints.get(param_name, str))
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    doc = (fn.__doc__ or "").strip()
    desc = description or (doc.split("\n")[0][:200] if doc else fn.__name__)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
