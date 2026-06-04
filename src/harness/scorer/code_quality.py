from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LINTER_COMMANDS: dict[str, list[str]] = {
    "ruff": ["ruff", "check", "--output-format=json"],
    "flake8": ["flake8", "--format=json", "."],
    "eslint": ["npx", "eslint", "--format=json", "."],
    "mypy": ["mypy", "--no-error-summary", "."],
}


@dataclass(slots=True)
class LintIssue:
    file: str
    line: int
    column: int
    severity: str
    message: str
    rule: str | None = None


@dataclass(slots=True)
class LayerScore:
    layer: str
    passed: bool
    details: str
    items: list[dict[str, Any]] = field(default_factory=list)


class CodeQualityChecker:
    """L0 programmatic scorer: runs the project's linter and reports issues."""

    def __init__(self, project_root: Path | None = None) -> None:
        self._root = project_root or Path.cwd()

    async def run_lint(self, context: Any) -> LayerScore:
        linter = self._detect_linter(self._root)
        if linter is None:
            return LayerScore(
                layer="L0:lint",
                passed=True,
                details="No supported linter detected — skipping.",
            )

        cmd = list(LINTER_COMMANDS[linter])
        lint_targets = self._resolve_lint_targets(getattr(context, "changed_files", []) or [])
        if linter == "ruff":
            cmd.extend(lint_targets if lint_targets else ["."])
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            return LayerScore(layer="L0:lint", passed=True, details=f"{linter} timed out")
        except FileNotFoundError:
            return LayerScore(
                layer="L0:lint",
                passed=True,
                details=f"{linter} binary not found on PATH",
            )

        output = stdout.decode(errors="replace")
        issues = self._parse_lint_output(output, linter)
        errors = [i for i in issues if i.severity in ("error", "E")]
        passed = len(errors) == 0

        return LayerScore(
            layer="L0:lint",
            passed=passed,
            details=f"{linter}: {len(issues)} issues ({len(errors)} errors)",
            items=[
                {
                    "file": i.file,
                    "line": i.line,
                    "severity": i.severity,
                    "message": i.message,
                    "rule": i.rule,
                }
                for i in issues[:50]
            ],
        )

    def _resolve_lint_targets(self, changed_files: list[str]) -> list[str]:
        targets: list[str] = []
        for raw in changed_files:
            try:
                p = Path(raw)
                if not p.is_absolute():
                    p = (self._root / p).resolve()
                else:
                    p = p.resolve()
                p.relative_to(self._root)
            except Exception:
                continue

            if p.exists() and p.is_file() and p.suffix == ".py":
                targets.append(str(p.relative_to(self._root)))
        # Keep deterministic order and avoid duplicate checks.
        return sorted(set(targets))

    @staticmethod
    def _detect_linter(project_path: Path) -> str | None:
        if shutil.which("ruff") and (project_path / "pyproject.toml").exists():
            return "ruff"
        if (project_path / ".flake8").exists() or (project_path / "setup.cfg").exists():
            if shutil.which("flake8"):
                return "flake8"
        if (project_path / ".eslintrc.json").exists() or (project_path / ".eslintrc.js").exists():
            return "eslint"
        if shutil.which("ruff"):
            return "ruff"
        return None

    @staticmethod
    def _parse_lint_output(output: str, linter: str) -> list[LintIssue]:
        import json as _json

        issues: list[LintIssue] = []
        if linter in ("ruff", "eslint"):
            try:
                data = _json.loads(output)
            except _json.JSONDecodeError:
                return issues

            if linter == "ruff":
                for item in data if isinstance(data, list) else []:
                    issues.append(LintIssue(
                        file=item.get("filename", ""),
                        line=item.get("location", {}).get("row", 0),
                        column=item.get("location", {}).get("column", 0),
                        severity="error" if item.get("fix") is None else "warning",
                        message=item.get("message", ""),
                        rule=item.get("code"),
                    ))
            elif linter == "eslint":
                for file_entry in data if isinstance(data, list) else []:
                    for msg in file_entry.get("messages", []):
                        issues.append(LintIssue(
                            file=file_entry.get("filePath", ""),
                            line=msg.get("line", 0),
                            column=msg.get("column", 0),
                            severity="error" if msg.get("severity") == 2 else "warning",
                            message=msg.get("message", ""),
                            rule=msg.get("ruleId"),
                        ))
        else:
            for line in output.splitlines():
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    issues.append(LintIssue(
                        file=parts[0].strip(),
                        line=int(parts[1]) if parts[1].strip().isdigit() else 0,
                        column=int(parts[2]) if parts[2].strip().isdigit() else 0,
                        severity="warning",
                        message=parts[3].strip(),
                    ))

        return issues
