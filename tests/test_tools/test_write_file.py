from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.file.write import WriteFileTool


@pytest.mark.asyncio
async def test_write_file_rejects_suspicious_partial_overwrite(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    original = "".join(f"line_{i} = {i}\n" for i in range(20))
    app.write_text(original, encoding="utf-8")
    tool = WriteFileTool()

    result = await tool.execute(path=str(app), content="line_10 = 999\n")

    assert not result.success
    assert result.error is not None
    assert "suspicious write_file overwrite" in result.error
    assert app.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_write_file_allows_new_file(tmp_path: Path) -> None:
    app = tmp_path / "new_app.py"
    tool = WriteFileTool()

    result = await tool.execute(path=str(app), content="print('hello')\n")

    assert result.success
    assert app.read_text(encoding="utf-8") == "print('hello')\n"


@pytest.mark.asyncio
async def test_write_file_allows_complete_overwrite(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("".join(f"line_{i} = {i}\n" for i in range(20)), encoding="utf-8")
    replacement = "".join(f"line_{i} = {i + 1}\n" for i in range(18))
    tool = WriteFileTool()

    result = await tool.execute(path=str(app), content=replacement)

    assert result.success
    assert app.read_text(encoding="utf-8") == replacement
