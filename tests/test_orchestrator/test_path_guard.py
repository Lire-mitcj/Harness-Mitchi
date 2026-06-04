from __future__ import annotations

from pathlib import Path

from src.orchestrator.path_guard import is_path_allowed, normalize_rel_path


def test_path_whitelist_allows_listed_file(tmp_path: Path) -> None:
    root = tmp_path
    (root / "api.py").write_text("x")
    assert is_path_allowed(root, "api.py", ["api.py"])
    assert not is_path_allowed(root, "other.py", ["api.py"])


def test_empty_whitelist_allows_any(tmp_path: Path) -> None:
    root = tmp_path
    assert is_path_allowed(root, "anything.py", [])


def test_whitelist_accepts_dot_slash_path(tmp_path: Path) -> None:
    root = tmp_path
    sql = root / "db" / "init" / "01_schema.sql"
    sql.parent.mkdir(parents=True)
    sql.write_text("CREATE TABLE x;")
    assert is_path_allowed(root, "./db/init/01_schema.sql", ["db/init/01_schema.sql"])


def test_should_apply_context_whitelist_only_mutations() -> None:
    from src.orchestrator.path_guard import should_apply_context_whitelist

    assert should_apply_context_whitelist("edit_file")
    assert should_apply_context_whitelist("write_file")
    assert not should_apply_context_whitelist("read_file")
    assert not should_apply_context_whitelist("grep_search")


def test_normalize_rel_path(tmp_path: Path) -> None:
    f = tmp_path / "src" / "main.py"
    f.parent.mkdir()
    f.write_text("ok")
    rel = normalize_rel_path(tmp_path, str(f))
    assert rel == "src/main.py"


def test_format_whitelist_denial_outside_project(tmp_path: Path) -> None:
    from src.orchestrator.path_guard import format_whitelist_denial

    msg = format_whitelist_denial(
        "write_file",
        ["/tmp/note.txt"],
        ["app.py", "main.py"],
        project_root=tmp_path,
    )
    assert "outside the project" in msg
    assert "/tmp" in msg
    assert "grep_search" in msg
