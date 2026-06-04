from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.file.edit import EditFileTool


@pytest.mark.asyncio
async def test_edit_file_rejects_identical_old_and_new(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("def make_boarding_pass_pdf(order):\n    pass\n", encoding="utf-8")
    tool = EditFileTool()

    result = await tool.execute(
        path=str(app),
        old_string="def make_boarding_pass_pdf(order):",
        new_string="def make_boarding_pass_pdf(order):",
    )

    assert not result.success
    assert result.error is not None
    assert "identical" in result.error.lower()
    assert app.read_text(encoding="utf-8") == "def make_boarding_pass_pdf(order):\n    pass\n"


@pytest.mark.asyncio
async def test_edit_file_applies_unique_replacement(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x = 1\n", encoding="utf-8")
    tool = EditFileTool()

    result = await tool.execute(
        path=str(app),
        old_string="x = 1",
        new_string="x = 2",
    )

    assert result.success
    assert app.read_text(encoding="utf-8") == "x = 2\n"


@pytest.mark.asyncio
async def test_edit_file_not_found_hints_on_short_old_string(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text(
        "def make_boarding_pass_pdf(order):\n    sql = 'SELECT 1'\n    return sql\n",
        encoding="utf-8",
    )
    tool = EditFileTool()

    result = await tool.execute(
        path=str(app),
        old_string="make_boarding_pass_pdf",
        new_string="make_boarding_pass_pdf",
    )

    assert not result.success
    assert result.error is not None
    assert "identical" in result.error.lower() or "Near line" in result.error
