from __future__ import annotations

from pathlib import Path
from src.tools.arg_normalize import normalize_grep_search_args


class MockSubtask:
    def __init__(self) -> None:
        self.description = "Fix registration transaction commit/rollback in main.py"
        self.acceptance_criteria = "sp_register_passenger has try/except"
        self.context_files = ["main.py"]


def test_normalize_grep_fills_pattern_from_subtask() -> None:
    subtask = MockSubtask()
    args = normalize_grep_search_args({"path": "main.py"}, subtask=subtask)
    assert "pattern" in args
    assert "register" in args["pattern"]
    assert args["path"] == "main.py"


def test_normalize_shell_replaces_missing_workspace(tmp_path: Path) -> None:
    from src.tools.arg_normalize import normalize_shell_exec_args

    args = normalize_shell_exec_args(
        {"command": "pytest test_api.py -q", "working_dir": "/workspace"},
        project_root=tmp_path,
    )
    assert args["working_dir"] == str(tmp_path.resolve())


def test_normalize_shell_defaults_cwd(tmp_path: Path) -> None:
    from src.tools.arg_normalize import normalize_shell_exec_args

    args = normalize_shell_exec_args(
        {"command": "pytest -q"},
        project_root=tmp_path,
    )
    assert args["working_dir"] == str(tmp_path.resolve())
