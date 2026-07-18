from __future__ import annotations

from pathlib import Path

from src.tools.arg_normalize import (
    normalize_grep_search_args,
    normalize_shell_exec_args,
    unwrap_raw_tool_arguments,
)


class MockSubtask:
    def __init__(self) -> None:
        self.description = "Fix registration transaction commit/rollback in main.py"
        self.acceptance_criteria = "sp_register_passenger has try/except"
        self.context_files = ["main.py"]


def test_normalize_grep_fills_pattern_from_subtask() -> None:
    subtask = MockSubtask()
    args = normalize_grep_search_args({"path": "main.py"}, subtask=subtask)
    assert "pattern" in args or "patterns" in args
    joined = " ".join(args.get("patterns") or [args.get("pattern", "")]).casefold()
    assert "register" in joined
    assert args["path"] == "main.py"


def test_normalize_grep_fills_patterns_from_hint_text() -> None:
    args = normalize_grep_search_args(
        {"include": "*.sql"},
        hint_text="Add order_timeline table and customer service API endpoint",
    )
    assert "patterns" in args or "pattern" in args
    joined = " ".join(args.get("patterns") or [args.get("pattern", "")]).casefold()
    assert "create table" in joined
    assert "order_timeline" in joined or "timeline" in joined


def test_normalize_grep_fills_exception_handler_patterns_from_hint() -> None:
    args = normalize_grep_search_args(
        {},
        hint_text="把统一数据库异常日志接口接到现有的与数据库有关的接口上",
    )
    assert "patterns" in args or "pattern" in args
    joined = " ".join(args.get("patterns") or [args.get("pattern", "")])
    assert "exception_handler" in joined or "handle_" in joined
    assert args.get("include") == "main.py"
    assert args.get("path") == "main.py"


def test_normalize_grep_leaves_empty_without_hint_or_subtask() -> None:
    args = normalize_grep_search_args({"include": "*.py"})
    assert "pattern" not in args
    assert "patterns" not in args


def test_normalize_shell_replaces_missing_workspace(tmp_path: Path) -> None:
    args = normalize_shell_exec_args(
        {"command": "pytest test_api.py -q", "working_dir": "/workspace"},
        project_root=tmp_path,
    )
    assert args["working_dir"] == str(tmp_path.resolve())


def test_normalize_shell_defaults_cwd(tmp_path: Path) -> None:
    args = normalize_shell_exec_args(
        {"command": "pytest -q"},
        project_root=tmp_path,
    )
    assert args["working_dir"] == str(tmp_path.resolve())


def test_unwrap_raw_salvages_truncated_json_fields() -> None:
    args = unwrap_raw_tool_arguments({"_raw": '{"include": "*.'})
    assert args.get("include") == "*."


def test_normalize_grep_from_raw_partial_json_fills_pattern_from_hint() -> None:
    args = normalize_grep_search_args(
        {"_raw": '{"include": "*.'},
        hint_text="优化 grpc_server @mention 检测并更新 noise_policy.yaml bot_nicknames",
    )
    assert "pattern" in args or "patterns" in args
    joined = " ".join(args.get("patterns") or [args.get("pattern", "")]).casefold()
    assert "mention" in joined or "bot" in joined or "noise" in joined
