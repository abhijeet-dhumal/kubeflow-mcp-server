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

"""Shared slash-command dispatcher for framework REPLs."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

REPL_EXIT_COMMANDS = frozenset({"exit", "quit", "q"})


@dataclass(frozen=True)
class CommonReplHandlers:
    """Callbacks for framework-agnostic slash commands."""

    on_tools: Callable[[], None]
    on_think: Callable[[], None]
    on_export: Callable[[], None]
    on_import: Callable[[str], None]
    on_clear: Callable[[], None]
    on_unknown: Callable[[str], None]


def handle_common_repl_command(line: str, handlers: CommonReplHandlers) -> bool:
    """Handle shared slash commands. Returns True when consumed."""
    if line == "/tools":
        handlers.on_tools()
        return True
    if line == "/think":
        handlers.on_think()
        return True
    if line == "/export":
        handlers.on_export()
        return True
    if line.startswith("/import"):
        path = line[len("/import") :].strip()
        if path:
            handlers.on_import(path)
        else:
            handlers.on_unknown("/import")
        return True
    if line == "/clear":
        handlers.on_clear()
        return True
    if line.startswith("/"):
        handlers.on_unknown(line)
        return True
    return False


def dispatch_slash_command(
    line: str,
    *,
    handlers: CommonReplHandlers,
    custom_dispatch: Callable[[str, list[str]], bool] | None = None,
    exit_commands: Collection[str] = REPL_EXIT_COMMANDS,
) -> bool:
    """Dispatch slash commands with custom handlers then common handlers.

    Returns True only when the caller should exit the REPL.
    """
    parts = line.split()
    if not parts:
        return False
    command = parts[0].lower()
    args = parts[1:]
    if command in exit_commands:
        return True
    if custom_dispatch and custom_dispatch(command, args):
        return False
    handle_common_repl_command(command, handlers)
    return False

