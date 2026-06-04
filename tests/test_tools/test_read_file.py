from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.file.read import ReadFileTool


@pytest.fixture
def read_tool() -> ReadFileTool:
    return ReadFileTool()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.py"
    p.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
    return p


@pytest.mark.asyncio
async def test_read_full_file(read_tool: ReadFileTool, sample_file: Path) -> None:
    result = await read_tool.execute(path=str(sample_file))
    assert result.success
    assert "line 1" in result.output
    assert "line 5" in result.output
    assert result.metadata is not None
    assert result.metadata["total_lines"] == 5


@pytest.mark.asyncio
async def test_read_line_range(read_tool: ReadFileTool, sample_file: Path) -> None:
    result = await read_tool.execute(path=str(sample_file), start_line=2, end_line=4)
    assert result.success
    assert "line 2" in result.output
    assert "line 4" in result.output
    assert "line 1" not in result.output
    assert "line 5" not in result.output


@pytest.mark.asyncio
async def test_read_nonexistent_file(read_tool: ReadFileTool, tmp_path: Path) -> None:
    result = await read_tool.execute(path=str(tmp_path / "nope.txt"))
    assert not result.success
    assert result.error is not None
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_read_directory_lists_entries(read_tool: ReadFileTool, tmp_path: Path) -> None:
    (tmp_path / "child.txt").write_text("x")
    result = await read_tool.execute(path=str(tmp_path))
    assert result.success
    assert "Directory listing" in result.output
    assert "child.txt" in result.output
    assert result.metadata is not None
    assert result.metadata.get("is_directory") is True


@pytest.mark.asyncio
async def test_read_empty_file(read_tool: ReadFileTool, tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    result = await read_tool.execute(path=str(empty))
    assert result.success
    assert result.metadata is not None
    assert result.metadata["total_lines"] == 0


@pytest.mark.asyncio
async def test_line_numbers_in_output(read_tool: ReadFileTool, sample_file: Path) -> None:
    result = await read_tool.execute(path=str(sample_file))
    assert result.success
    lines = result.output.strip().splitlines()
    assert lines[0].strip().startswith("1|")
    assert lines[-1].strip().startswith("5|")
