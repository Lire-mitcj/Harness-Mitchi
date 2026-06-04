from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Rubric:
    id: str
    category: str
    severity: str  # "blocker" | "warning" | "hint"
    question: str
    pass_criteria: str
    fail_criteria: str
    evidence_required: bool = True


_DEFAULT_RUBRICS: list[Rubric] = [
    Rubric(
        id="TC-01",
        category="task_completion",
        severity="blocker",
        question="Does the diff address the user's request?",
        pass_criteria="The changes directly fulfil what the user asked for.",
        fail_criteria="The diff is unrelated or only partially addresses the request.",
    ),
    Rubric(
        id="TC-02",
        category="correctness",
        severity="blocker",
        question="Is the code syntactically and logically correct?",
        pass_criteria="No syntax errors; control flow is sound.",
        fail_criteria="Contains obvious bugs, typos, or broken logic.",
    ),
    Rubric(
        id="TC-03",
        category="safety",
        severity="blocker",
        question="Are there any dangerous side-effects?",
        pass_criteria="No destructive operations, credential leaks, or data loss risks.",
        fail_criteria="Contains rm -rf, exposed secrets, or unbounded resource consumption.",
    ),
    Rubric(
        id="TC-04",
        category="style",
        severity="warning",
        question="Does the code follow project conventions?",
        pass_criteria="Consistent naming, formatting, and file organization.",
        fail_criteria="Introduces inconsistent style that deviates from the project.",
    ),
]


class RubricLoader:
    """Loads scoring rubrics from defaults, YAML files, and rules.md."""

    def load(
        self,
        default_path: Path | None = None,
        project_path: Path | None = None,
    ) -> list[Rubric]:
        rubrics = list(_DEFAULT_RUBRICS)

        if default_path and default_path.exists():
            rubrics.extend(self._load_yaml(default_path))

        if project_path:
            rules_md = project_path / "rules.md"
            if rules_md.exists():
                rubrics.extend(self._parse_rules_md(rules_md))

            for yaml_file in project_path.glob("rubrics*.yaml"):
                rubrics.extend(self._load_yaml(yaml_file))
            for yml_file in project_path.glob("rubrics*.yml"):
                rubrics.extend(self._load_yaml(yml_file))

        return rubrics

    def _load_yaml(self, path: Path) -> list[Rubric]:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            log.warning("PyYAML not installed — skipping %s", path)
            return []

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to parse %s: %s", path, exc)
            return []

        if not isinstance(data, list):
            data = data.get("rubrics", []) if isinstance(data, dict) else []

        results: list[Rubric] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                results.append(Rubric(
                    id=item["id"],
                    category=item.get("category", "custom"),
                    severity=item.get("severity", "warning"),
                    question=item["question"],
                    pass_criteria=item.get("pass_criteria", ""),
                    fail_criteria=item.get("fail_criteria", ""),
                    evidence_required=item.get("evidence_required", True),
                ))
            except KeyError as exc:
                log.warning("Skipping rubric entry missing key %s in %s", exc, path)
        return results

    def _parse_rules_md(self, path: Path) -> list[Rubric]:
        """Convert ``rules.md`` headings into rubric entries prefixed with ``PRJ-``."""
        text = path.read_text(encoding="utf-8")
        rubrics: list[Rubric] = []
        counter = 1

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            heading = stripped.lstrip("#").strip()
            if not heading:
                continue

            rubrics.append(Rubric(
                id=f"PRJ-{counter:02d}",
                category="project_rule",
                severity="warning",
                question=f"Does the change comply with project rule: {heading}?",
                pass_criteria=f"The diff respects the project rule: {heading}.",
                fail_criteria=f"The diff violates the project rule: {heading}.",
                evidence_required=False,
            ))
            counter += 1

        return rubrics
