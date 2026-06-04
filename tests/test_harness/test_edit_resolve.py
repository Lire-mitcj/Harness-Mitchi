from __future__ import annotations

from pathlib import Path

from src.harness.edit.resolve import resolve_edit_targets


def test_resolve_edit_targets_from_symbol_hit(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text(
        "def make_boarding_pass_pdf(order):\n    sql = 'SELECT 1'\n    return sql\n",
        encoding="utf-8",
    )
    summary = (
        "Findings\n\n"
        "| File | Line | Symbol |\n"
        "| app.py | 1 | make_boarding_pass_pdf |\n"
    )
    targets = resolve_edit_targets(
        project_root=tmp_path,
        prior_summaries={"st-1": summary},
        whitelist_files=["app.py"],
    )
    assert len(targets) == 1
    assert targets[0].symbol == "make_boarding_pass_pdf"
    assert targets[0].path == "app.py"
    assert "make_boarding_pass_pdf" in targets[0].original_source
    assert targets[0].anchor_hash


def test_resolve_edit_targets_from_line_ref(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    summary = "Target at app.py:2-3 for edit."
    targets = resolve_edit_targets(
        project_root=tmp_path,
        prior_summaries={"st-1": summary},
        whitelist_files=["app.py"],
    )
    assert targets
    assert targets[0].path == "app.py"
    assert targets[0].start_line <= 2
