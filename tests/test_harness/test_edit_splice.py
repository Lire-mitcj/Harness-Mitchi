from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.edit.apply import execute_replace_symbol
from src.harness.edit.extract import anchor_hash, slice_file_lines
from src.harness.edit.resolve import refresh_target_span
from src.harness.edit.errors import AnchorError
from src.harness.edit.splice import apply_splice, verify_anchor
from src.harness.edit.target import EditTarget
from src.indexer.parser import CodeParser


def _make_target(content: str, *, rel: str = "app.py", symbol: str = "foo") -> EditTarget:
    parser = CodeParser()
    path = Path(rel)
    path.write_text(content, encoding="utf-8")
    parsed = parser.parse_file(path)
    sym = parsed.functions[0]
    original = slice_file_lines(content, sym.start_line, sym.end_line)
    return EditTarget(
        path=rel,
        symbol=symbol,
        kind="function",
        start_line=sym.start_line,
        end_line=sym.end_line,
        original_source=original,
        anchor_hash=anchor_hash(original),
    )


def test_apply_splice_replaces_span(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    content = "before\ndef foo():\n    return 1\n\nafter\n"
    target = _make_target(content, symbol="foo")
    result = apply_splice(
        content,
        start_line=target.start_line,
        end_line=target.end_line,
        new_body="def foo():\n    return 2\n",
        expected_hash=target.anchor_hash,
    )
    assert result.success
    assert "return 2" in (result.new_content or "")
    assert "before" in (result.new_content or "")


def test_apply_splice_rejects_bad_anchor(tmp_path: Path) -> None:
    content = "def foo():\n    return 1\n"
    target = _make_target(content)
    with pytest.raises(AnchorError):
        verify_anchor(
            "def foo():\n    return 9\n",
            start_line=target.start_line,
            end_line=target.end_line,
            expected_hash=target.anchor_hash,
        )


def test_refresh_target_span_after_line_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    content = "def foo():\n    return 1\n"
    target = _make_target(content, symbol="foo")
    drifted = "# inserted\n" + content
    (tmp_path / "app.py").write_text(drifted, encoding="utf-8")

    with pytest.raises(AnchorError):
        verify_anchor(
            drifted,
            start_line=target.start_line,
            end_line=target.end_line,
            expected_hash=target.anchor_hash,
        )

    refreshed = refresh_target_span(tmp_path, target)
    assert refreshed is not None
    assert refreshed.start_line == 2
    assert refreshed.anchor_hash == target.anchor_hash


@pytest.mark.asyncio
async def test_replace_symbol_retries_after_anchor_drift(tmp_path: Path) -> None:
    content = "def make_boarding_pass_pdf(order):\n    return order\n"
    app = tmp_path / "app.py"
    app.write_text(content, encoding="utf-8")
    parser = CodeParser()
    sym = parser.parse_file(app).functions[0]
    original = slice_file_lines(content, sym.start_line, sym.end_line)
    target = EditTarget(
        path="app.py",
        symbol="make_boarding_pass_pdf",
        kind="function",
        start_line=sym.start_line,
        end_line=sym.end_line,
        original_source=original,
        anchor_hash=anchor_hash(original),
    )

    app.write_text("# header\n" + content, encoding="utf-8")

    result = execute_replace_symbol(
        project_root=tmp_path,
        targets=[target],
        path="app.py",
        symbol="make_boarding_pass_pdf",
        new_body="def make_boarding_pass_pdf(order):\n    return 'view'\n",
        max_attempts=2,
    )
    assert result.success
    assert "view" in app.read_text(encoding="utf-8")
    assert "re-resolved" in (result.output or "")
