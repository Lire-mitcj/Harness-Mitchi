from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.grep_discovery import grep_scope_for_task
from src.tools.assembled.grep_search import GrepSearchTool


@pytest.fixture
def go_project(tmp_path: Path) -> Path:
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.22\n", encoding="utf-8")
    handler = tmp_path / "handler.go"
    handler.write_text(
        "package demo\n\n"
        "func GetOrder(w http.ResponseWriter, r *http.Request) {\n"
        "    w.WriteHeader(200)\n"
        "}\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.asyncio
async def test_grep_symbol_mode_finds_go_func(go_project: Path) -> None:
    tool = GrepSearchTool(project_root=go_project)
    result = await tool.execute(
        pattern="GetOrder",
        path=str(go_project),
        include="*.go",
        mode="symbol",
    )
    assert result.success
    assert "GetOrder" in result.output
    assert "handler.go" in result.output


@pytest.mark.asyncio
async def test_grep_default_identifier_finds_go_definition(go_project: Path) -> None:
    import json

    tool = GrepSearchTool(project_root=go_project)
    result = await tool.execute(
        pattern="GetOrder",
        path=str(go_project),
        include="*.go",
        mode="default",
    )
    assert result.success
    payload = json.loads(result.output)
    assert any(m.get("symbol") == "GetOrder" for m in payload.get("matches", []))


def test_grep_scope_for_go_project(go_project: Path) -> None:
    include, path = grep_scope_for_task("add HTTP handler for orders", project_root=go_project)
    assert include == "*.go"
    assert path == "."
