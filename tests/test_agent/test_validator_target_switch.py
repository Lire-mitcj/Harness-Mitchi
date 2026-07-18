from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.validator import CursorValidator
from src.indexer.project_stack import (
    detect_project_stack,
    maven_module_for_target,
    validator_command_for_target,
)


def test_validator_command_for_go_file(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)

    assert validator_command_for_target(
        stack=stack,
        target_file="internal/ws/server.go",
        project_root=tmp_path,
    ) == ("go", "test", "./...")


def test_validator_command_for_python_file_in_mixed_monorepo(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    assert stack.primary == "mixed"

    assert validator_command_for_target(
        stack=stack,
        target_file="agentmesh_orchestrator/list.py",
        project_root=tmp_path,
    ) == ("pytest",)


def test_validator_command_skips_proto_and_sql(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)

    assert validator_command_for_target(
        stack=stack,
        target_file="api/agentmesh.proto",
        project_root=tmp_path,
    ) == ()
    assert validator_command_for_target(
        stack=stack,
        target_file="db/schema.sql",
        project_root=tmp_path,
    ) == ()


def test_maven_module_for_target_picks_longest_match() -> None:
    modules = ("service-a", "service-a/web")
    assert maven_module_for_target("service-a/web/src/main/java/App.java", modules) == "service-a/web"
    assert maven_module_for_target("service-a/core/src/main/java/App.java", modules) == "service-a"


@pytest.mark.asyncio
async def test_mixed_monorepo_uses_pytest_for_python_edit(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    validator = CursorValidator(
        tmp_path,
        command=stack.validator_command,
        stack=stack,
        per_file_commands=True,
    )

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    mock_proc.returncode = 0
    mock_proc.kill = AsyncMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
        result = await validator.validate_execution("pkg/orchestrator/list.py")

    assert result["pass"] is True
    spawn.assert_called_once()
    assert spawn.call_args.args[:2] == ("pytest",)


@pytest.mark.asyncio
async def test_mixed_monorepo_uses_go_test_for_go_edit(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    validator = CursorValidator(
        tmp_path,
        command=stack.validator_command,
        stack=stack,
        per_file_commands=True,
    )

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    mock_proc.returncode = 0
    mock_proc.kill = AsyncMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
        result = await validator.validate_execution("internal/ws/server.go")

    assert result["pass"] is True
    spawn.assert_called_once()
    assert spawn.call_args.args[:3] == ("go", "test", "./internal/ws")


@pytest.mark.asyncio
async def test_proto_edit_skips_execution_validator(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    validator = CursorValidator(
        tmp_path,
        command=stack.validator_command,
        stack=stack,
        per_file_commands=True,
    )

    with patch("asyncio.create_subprocess_exec") as spawn:
        result = await validator.validate_execution("api/agentmesh.proto")

    assert result["pass"] is True
    assert result["status"] == "SKIPPED"
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_per_file_switch_disabled_keeps_stack_command(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    validator = CursorValidator(
        tmp_path,
        command=stack.validator_command,
        stack=stack,
        per_file_commands=False,
    )

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    mock_proc.returncode = 0
    mock_proc.kill = AsyncMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
        result = await validator.validate_execution("pkg/orchestrator/list.py")

    assert result["pass"] is True
    spawn.assert_called_once()
    assert spawn.call_args.args[:2] == ("go", "test")


@pytest.mark.asyncio
async def test_maven_validator_scopes_to_module(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text(
        "<project><modules><module>service-a</module><module>service-b</module></modules></project>",
        encoding="utf-8",
    )
    stack = detect_project_stack(tmp_path)
    validator = CursorValidator(
        tmp_path,
        command=stack.validator_command,
        stack=stack,
        per_file_commands=True,
    )

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    mock_proc.returncode = 0
    mock_proc.kill = AsyncMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
        result = await validator.validate_execution(
            "service-b/src/main/java/com/example/App.java"
        )

    assert result["pass"] is True
    assert spawn.call_args.args == ("mvn", "-q", "-pl", "service-b", "test")
