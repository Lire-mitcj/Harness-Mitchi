from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.assembled.view_symbol_code import (
    ViewSymbolCodeTool,
    _find_symbol_span_by_regex,
)


GO_SERVER = """package ws

type Server struct {
    addr string
}

func NewServer(addr string) *Server {
    return &Server{addr: addr}
}
"""


def test_go_symbol_span_prefers_newserver_over_type_server() -> None:
    span = _find_symbol_span_by_regex(GO_SERVER, "NewServer", file_path="internal/ws/server.go")
    assert span is not None
    start, _end = span
    line = GO_SERVER.splitlines()[start - 1]
    assert "func NewServer" in line
    assert "type Server" not in line


@pytest.mark.asyncio
async def test_view_symbol_code_resolves_go_constructor(tmp_path: Path) -> None:
    target = tmp_path / "internal" / "ws"
    target.mkdir(parents=True)
    (target / "server.go").write_text(GO_SERVER, encoding="utf-8")

    tool = ViewSymbolCodeTool(project_root=tmp_path, settings=object())
    result = await tool.execute(
        target_file="internal/ws/server.go",
        symbol="NewServer",
    )

    assert result.success is True
    assert "func NewServer" in result.metadata["verbatim_code"]
    assert result.metadata["span"][0] == 7
