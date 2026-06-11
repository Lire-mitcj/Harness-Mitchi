from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.harness.scorer.test_runner import TestRunner, FRAMEWORK_COMMANDS
from src.skills.base import SkillContext, SkillResult


class VerifySkill:
    name = "verify"

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.test_runner = TestRunner(self.project_root)

    async def run(self, context: SkillContext, **kwargs: object) -> SkillResult:
        changed_files = list(kwargs.get("changed_files", ()) or ())

        # Enforce SQL references check first
        from src.skills.validator import validate_sql_references
        db_errors = validate_sql_references(self.project_root, changed_files)
        if db_errors:
            return SkillResult(
                success=False,
                summary="Verification failed: " + "; ".join(db_errors),
                validation_result="failed",
                missing_info=tuple(db_errors),
            )

        framework = self.test_runner._detect_test_framework()
        if framework is None:
            return SkillResult(
                success=False,
                summary="Verification failed: No test framework detected in the project.",
                validation_result="failed",
            )

        related = self.test_runner._find_related_tests(changed_files, framework)

        cmd = list(FRAMEWORK_COMMANDS[framework])
        if framework == "pytest" and cmd[:3] == ["python", "-m", "pytest"]:
            cmd[0] = sys.executable
        if related:
            cmd.extend(related)
            test_scope_desc = f"{len(related)} related test file(s)"
        else:
            test_scope_desc = "all tests (fallback)"

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                env=_subprocess_env(self.project_root, framework),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SkillResult(
                success=False,
                summary=f"Verification failed: Tests timed out after 120s ({framework})",
                validation_result="failed",
            )
        except FileNotFoundError:
            return SkillResult(
                success=False,
                summary=f"Verification failed: {framework} command not found on PATH.",
                validation_result="failed",
            )

        output = (completed.stdout or "") + (completed.stderr or "")
        passed = completed.returncode == 0

        summary = f"{framework} ({test_scope_desc}): {'PASSED' if passed else 'FAILED'}"
        if not passed:
            # Pytest exit code 5 (no tests collected) is also treated as failed verification
            err_details = f"{summary}\n\nOutput:\n{output[-2000:]}"
            return SkillResult(
                success=False,
                summary=err_details,
                validation_result="failed",
                metadata={"test_output": output[-3000:]},
            )

        return SkillResult(
            success=True,
            summary=summary,
            validation_result="passed",
            metadata={"test_output": output[-3000:]},
        )


def _subprocess_env(project_root: Path, framework: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTEST_CURRENT_TEST", None)
    if framework == "pytest" and not _has_pytest_config(project_root):
        env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    return env


def _has_pytest_config(project_root: Path) -> bool:
    for name in ("pytest.ini", "pyproject.toml", "setup.cfg"):
        path = project_root / name
        if not path.is_file():
            continue
        if name == "pyproject.toml":
            content = path.read_text(errors="replace")
            if "[tool.pytest" not in content and "pytest" not in content.lower():
                continue
        return True
    return False
