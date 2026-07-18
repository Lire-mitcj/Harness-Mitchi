from __future__ import annotations

import shutil

import pytest

from src.agent.validator import CursorValidator, _go_test_scope, _gofmt_syntax_issues

gofmt_available = shutil.which("gofmt") is not None


@pytest.mark.skipif(not gofmt_available, reason="gofmt not installed")
def test_gofmt_syntax_issues_detects_bad_go() -> None:
    issues = _gofmt_syntax_issues("package main\n\nfunc main() {\n")
    assert issues
    assert issues[0].startswith("go_syntax_error:")


@pytest.mark.skipif(not gofmt_available, reason="gofmt not installed")
def test_gofmt_syntax_issues_passes_valid_go() -> None:
    issues = _gofmt_syntax_issues("package main\n\nfunc main() {}\n")
    assert issues == []


def test_go_test_scope_points_at_package_dir() -> None:
    assert _go_test_scope("internal/ws/server.go") == "./internal/ws"
    assert _go_test_scope("main.go") == "./..."


@pytest.mark.skipif(not gofmt_available, reason="gofmt not installed")
@pytest.mark.asyncio
async def test_validate_ast_flags_go_syntax(tmp_path) -> None:
    validator = CursorValidator(tmp_path)
    result = validator.validate_ast(
        "broken.go",
        "",
        "package main\n\nfunc main() {\n",
    )
    assert not result["pass"]
    assert any(str(item).startswith("go_syntax_error") for item in result["issues"])
