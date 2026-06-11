from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from src.harness.scorer.code_quality import CodeQualityChecker
from src.harness.scorer.code_quality import LayerScore
from src.harness.scorer.feedback import FeedbackFormatter
from src.harness.scorer.rubric_loader import RubricLoader
from src.harness.scorer.task_completion import LLMJudge
from src.harness.scorer.test_runner import TestRunner

log = logging.getLogger(__name__)


@dataclass
class ScoringContext:
    user_message: str = ""
    acceptance_criteria: str = ""
    diff: str = ""
    language: str = ""
    has_related_tests: bool = False
    project_rubrics_path: Path | None = None
    recent_tool_calls: list[Any] = field(default_factory=list)
    rules_md_summary: str = ""
    changed_files: list[str] = field(default_factory=list)
    project_root: Path | None = None


@dataclass
class ScoreResult:
    passed: bool
    scores: dict[str, Any] = field(default_factory=dict)
    feedback: str | None = None
    needs_retry: bool = False
    warnings: list[str] = field(default_factory=list)
    retry_reasons: list[str] = field(default_factory=list)


class ScoringEngine:
    """Three-layer scoring pipeline: L0 → L1 → L2.

    L0 — Programmatic checks (lint, tests).  Fast, deterministic.
    L1 — LLM rubric judge.  Evaluates task completion and correctness.
    L2 — Quality hints.  Non-blocking warnings aggregated from L0/L1.

    Short-circuit: if L0 fails on a blocker, L1 and L2 are skipped.
    """

    def __init__(
        self,
        llm_client: Callable[..., Awaitable[str]] | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._lint = CodeQualityChecker(project_root)
        self._tests = TestRunner(project_root)
        self._judge = LLMJudge(llm_client)
        self._rubric_loader = RubricLoader()
        self._formatter = FeedbackFormatter()

    async def evaluate(self, context: ScoringContext, *, skip_l1: bool = False) -> ScoreResult:
        scores: dict[str, Any] = {}
        warnings: list[str] = []

        # --- L0: programmatic ---
        lint_score = await self._lint.run_lint(context)
        scores["lint"] = _layer_dict(lint_score)

        test_score = await self._tests.run(context)
        scores["tests"] = _layer_dict(test_score)

        l0_passed = lint_score.passed and test_score.passed
        if not l0_passed:
            feedback = self._build_l0_feedback(lint_score, test_score)
            return ScoreResult(
                passed=False,
                scores=scores,
                feedback=feedback,
                needs_retry=True,
                warnings=warnings,
                retry_reasons=["L0"],
            )

        if skip_l1:
            return ScoreResult(
                passed=True,
                scores=scores,
                feedback=None,
                needs_retry=False,
                warnings=warnings,
            )

        # --- L1: LLM rubric judge ---
        rubrics = self._rubric_loader.load(
            project_path=context.project_rubrics_path,
        )
        verdict = await self._judge.judge(context, rubrics)
        scores["rubric"] = {
            "verdict": verdict.verdict,
            "blockers": verdict.blockers,
            "warnings": verdict.warnings,
            "results": [
                {"id": r.id, "passed": r.passed, "reason": r.reason}
                for r in verdict.results
            ],
        }

        if verdict.blockers:
            feedback = self._formatter.format_rubric_feedback(verdict)
            return ScoreResult(
                passed=False,
                scores=scores,
                feedback=feedback,
                needs_retry=True,
                warnings=verdict.warnings,
                retry_reasons=["L1"],
            )

        # --- L2: quality hints (non-blocking) ---
        warnings.extend(verdict.warnings)
        hint_feedback = self._formatter.format_warnings(warnings) if warnings else None

        return ScoreResult(
            passed=True,
            scores=scores,
            feedback=hint_feedback,
            needs_retry=False,
            warnings=warnings,
        )

    def _build_l0_feedback(
        self,
        lint_score: LayerScore,
        test_score: LayerScore,
    ) -> str:
        parts: list[str] = []
        if not lint_score.passed:
            parts.append(f"Lint failed: {lint_score.details}")
            for item in lint_score.items[:5]:
                parts.append(
                    f"  {item.get('file', '?')}:{item.get('line', '?')} "
                    f"{item.get('message', '')}"
                )
        if not test_score.passed:
            parts.append(f"Tests failed: {test_score.details}")
        return "\n".join(parts)


def _layer_dict(score: LayerScore) -> dict[str, Any]:
    return {
        "layer": score.layer,
        "passed": score.passed,
        "details": score.details,
        "items": score.items,
    }
