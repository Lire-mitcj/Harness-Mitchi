from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest

from src.agent.validator import CursorValidator, _proto_syntax_issues

buf_available = shutil.which("buf") is not None
protoc_available = shutil.which("protoc") is not None


def test_proto_syntax_issues_detects_invalid_proto_with_mock() -> None:
    with patch("shutil.which", return_value="/usr/bin/buf"):
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stdout="", stderr="syntax error: missing ;")
            issues = _proto_syntax_issues(
                "service Demo {",
                target_file="api/demo.proto",
                project_root=MagicMock(),
            )
    assert issues
    assert issues[0].startswith("proto_syntax_error:")


def test_proto_syntax_issues_passes_valid_proto_with_mock() -> None:
    with patch("shutil.which", return_value="/usr/bin/buf"):
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            issues = _proto_syntax_issues(
                'syntax = "proto3";\n\nservice Demo {}\n',
                target_file="api/demo.proto",
                project_root=MagicMock(),
            )
    assert issues == []


@pytest.mark.skipif(not (buf_available or protoc_available), reason="buf/protoc not installed")
def test_proto_syntax_issues_integration_valid_proto(tmp_path) -> None:
    issues = _proto_syntax_issues(
        'syntax = "proto3";\n\npackage demo.v1;\n\nservice Demo {}\n',
        target_file="api/demo.proto",
        project_root=tmp_path,
    )
    assert issues == []


@pytest.mark.skipif(not (buf_available or protoc_available), reason="buf/protoc not installed")
def test_validate_ast_flags_proto_syntax(tmp_path) -> None:
    validator = CursorValidator(tmp_path)
    result = validator.validate_ast(
        "api/demo.proto",
        "",
        "service Broken {\n",
    )
    assert not result["pass"]
    assert any(str(item).startswith("proto_syntax_error") for item in result["issues"])
