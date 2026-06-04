from __future__ import annotations

import py_compile
from pathlib import Path

from src.skills.base import SkillContext, SkillResult


class ValidatorSkill:
    name = "validator"

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    async def run(self, context: SkillContext, **kwargs: object) -> SkillResult:
        changed_files = tuple(str(path) for path in kwargs.get("changed_files", ()) or ())
        if not changed_files:
            return SkillResult(
                success=True,
                summary="No changed files to validate.",
                validation_result="skipped",
            )

        errors: list[str] = []
        checked: list[str] = []
        for rel in changed_files:
            path = _resolve_under_root(self.project_root, rel)
            if path is None:
                errors.append(f"{rel}: outside project root")
                continue
            if not path.is_file():
                errors.append(f"{rel}: file not found")
                continue
            checked.append(rel)
            if path.suffix == ".py":
                try:
                    py_compile.compile(str(path), doraise=True)
                except py_compile.PyCompileError as exc:
                    errors.append(f"{rel}: py_compile failed: {exc.msg}")

        if errors:
            return SkillResult(
                success=False,
                summary="Validation failed: " + "; ".join(errors),
                validation_result="failed",
                missing_info=tuple(errors),
            )
        return SkillResult(
            success=True,
            summary=f"Validated {len(checked)} changed file(s).",
            validation_result="passed",
        )


def _resolve_under_root(project_root: Path, rel: str) -> Path | None:
    try:
        path = (project_root / rel.replace("\\", "/").lstrip("./")).resolve()
        path.relative_to(project_root)
        return path
    except (OSError, ValueError):
        return None
