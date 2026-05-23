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

from kubeflow_mcp.agents.runtime.repl_commands import (
    REPL_EXIT_COMMANDS,
    CommonReplHandlers,
    dispatch_slash_command,
    handle_common_repl_command,
)


def test_common_commands_dispatch_expected_handler():
    calls: list[str] = []
    handlers = CommonReplHandlers(
        on_tools=lambda: calls.append("tools"),
        on_think=lambda: calls.append("think"),
        on_export=lambda: calls.append("export"),
        on_import=lambda path: calls.append(f"import:{path}"),
        on_clear=lambda: calls.append("clear"),
        on_unknown=lambda line: calls.append(f"unknown:{line}"),
    )

    assert handle_common_repl_command("/tools", handlers)
    assert handle_common_repl_command("/think", handlers)
    assert handle_common_repl_command("/export", handlers)
    assert handle_common_repl_command("/import some-file.json", handlers)
    assert handle_common_repl_command("/import", handlers)
    assert handle_common_repl_command("/clear", handlers)
    assert handle_common_repl_command("/not-a-command", handlers)
    assert calls == [
        "tools",
        "think",
        "export",
        "import:some-file.json",
        "unknown:/import",
        "clear",
        "unknown:/not-a-command",
    ]


def test_non_command_not_consumed():
    handlers = CommonReplHandlers(
        on_tools=lambda: None,
        on_think=lambda: None,
        on_export=lambda: None,
        on_import=lambda _path: None,
        on_clear=lambda: None,
        on_unknown=lambda _line: None,
    )
    assert not handle_common_repl_command("hello", handlers)


def test_dispatch_slash_command_prioritizes_custom_handler():
    calls: list[str] = []
    handlers = CommonReplHandlers(
        on_tools=lambda: calls.append("tools"),
        on_think=lambda: calls.append("think"),
        on_export=lambda: calls.append("export"),
        on_import=lambda path: calls.append(f"import:{path}"),
        on_clear=lambda: calls.append("clear"),
        on_unknown=lambda line: calls.append(f"unknown:{line}"),
    )

    handled = dispatch_slash_command(
        "/custom 1 2",
        handlers=handlers,
        custom_dispatch=lambda command, args: command == "/custom" and calls.append(f"custom:{args}") is None,
        exit_commands=REPL_EXIT_COMMANDS,
    )
    assert not handled
    assert calls == ["custom:['1', '2']"]


def test_dispatch_slash_command_handles_exit_and_fallback():
    calls: list[str] = []
    handlers = CommonReplHandlers(
        on_tools=lambda: calls.append("tools"),
        on_think=lambda: calls.append("think"),
        on_export=lambda: calls.append("export"),
        on_import=lambda path: calls.append(f"import:{path}"),
        on_clear=lambda: calls.append("clear"),
        on_unknown=lambda line: calls.append(f"unknown:{line}"),
    )

    assert dispatch_slash_command("exit", handlers=handlers)
    assert not dispatch_slash_command("/tools", handlers=handlers)
    assert calls == ["tools"]

