from __future__ import annotations

from src.executor.policy import resolve_executor_tools
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


def test_edit_with_preload_only_write_tools() -> None:
    subtask = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.EDIT,
        description="fix main.py",
        allowed_tools=["context_search", "edit_file", "write_file"],
        context_files=["main.py"],
    )
    tools = resolve_executor_tools(subtask, preloaded_paths=frozenset({"main.py"}))
    assert tools == frozenset({"edit_file", "write_file", "context_search"})
    assert "grep_search" not in tools
    assert "read_file" not in tools


def test_edit_paths_only_mode_allows_grep_and_read() -> None:
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="fix",
        allowed_tools=["context_search", "edit_file", "write_file"],
        context_files=["main.py"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset(),
        truncated_paths=frozenset({"main.py"}),
    )
    assert "context_search" in tools
    assert "grep_search" not in tools
    assert "read_file" not in tools
    assert "edit_file" in tools


def test_edit_scoped_explore_grants_context_search() -> None:
    """Harness grants context_search for scoped edit discovery."""
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="fix sql",
        allowed_tools=["edit_file", "write_file"],
        context_files=["app.py", "main.py"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset(),
        truncated_paths=frozenset({"app.py", "main.py"}),
    )
    assert "context_search" in tools
    assert "read_files" not in tools


def test_edit_truncated_preload_keeps_grep_and_read() -> None:
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="fix main.py",
        allowed_tools=["context_search", "edit_file", "write_file"],
        context_files=["main.py", "app.py"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset({"main.py", "app.py"}),
        truncated_paths=frozenset({"main.py", "app.py"}),
    )
    assert tools == frozenset({
        "edit_file",
        "write_file",
        "context_search",
    })


def test_edit_without_preload_keeps_read_grep() -> None:
    subtask = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.EDIT,
        description="fix",
        allowed_tools=["edit_file"],
    )
    tools = resolve_executor_tools(subtask, preloaded_paths=frozenset())
    assert "context_search" in tools
    assert "read_file" not in tools


def test_diagnose_grants_context_search_even_when_planner_omits() -> None:
    subtask = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="find view",
        allowed_tools=[],
        context_files=[],
    )
    tools = resolve_executor_tools(subtask, preloaded_paths=frozenset())
    assert "context_search" in tools
    assert "map_search" not in tools


def test_edit_splice_mode_tools() -> None:
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="fix",
        allowed_tools=["context_search", "edit_file"],
        context_files=["app.py"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset({"app.py"}),
        splice_edit=True,
    )
    assert tools == frozenset({"replace_symbol", "write_file"})
    assert "edit_file" not in tools


def test_edit_read_fallback_grants_read_on_full_preload() -> None:
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="fix",
        allowed_tools=["context_search", "edit_file"],
        context_files=["app.py"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset({"app.py"}),
        edit_read_fallback=True,
    )
    assert tools == frozenset({"edit_file", "read_file", "read_files"})


def test_edit_restrict_explore_with_preload_is_write_only() -> None:
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="switch query to view",
        allowed_tools=["context_search", "edit_file"],
        context_files=["app.py", "db/init/init.sql"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset({"app.py", "db/init/init.sql"}),
        explore_restricted=True,
    )
    assert tools == frozenset({"edit_file"})


def test_verify_with_preload_shell_only() -> None:
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.VERIFY,
        description="run test",
        allowed_tools=["shell_exec", "context_search"],
        context_files=["test_api.py"],
    )
    tools = resolve_executor_tools(subtask, preloaded_paths=frozenset({"test_api.py"}))
    assert tools == frozenset({"shell_exec"})


def test_verify_without_preload_can_use_read_grep_map() -> None:
    subtask = SubTaskNode(
        id="st-3",
        kind=SubTaskKind.VERIFY,
        description="inspect and run tests",
        allowed_tools=["shell_exec", "context_search"],
    )
    tools = resolve_executor_tools(subtask, preloaded_paths=frozenset())
    assert tools == frozenset({
        "shell_exec",
        "context_search",
    })


def test_coordinate_handoff_restricts_non_edit_explore_tools() -> None:
    subtask = SubTaskNode(
        id="st-3",
        kind=SubTaskKind.SHELL,
        description="run focused command",
        allowed_tools=["shell_exec", "context_search"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset({"app.py"}),
        explore_restricted=True,
    )
    assert tools == frozenset({"shell_exec"})


def test_coordinate_handoff_restricts_diagnose_explore_tools() -> None:
    subtask = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.DIAGNOSE,
        description="check cited target",
        allowed_tools=["context_search"],
    )
    tools = resolve_executor_tools(
        subtask,
        preloaded_paths=frozenset({"app.py"}),
        explore_restricted=True,
    )
    assert tools == frozenset()
