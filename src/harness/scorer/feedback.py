from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.harness.scorer.task_completion import RubricVerdict


class FeedbackFormatter:
    """Formats scoring results into concise, actionable feedback strings
    suitable for injecting back into the agent's context."""

    @staticmethod
    def format_rubric_feedback(verdict: RubricVerdict) -> str:
        if verdict.passed:
            return "All rubric checks passed."

        lines: list[str] = []

        for result in verdict.results:
            if result.passed:
                continue
            severity_tag = "BLOCKER" if result.severity == "blocker" else "WARNING"
            lines.append(f"- [{result.id}] [{severity_tag}] {result.reason}")
            if result.evidence:
                evidence_str = "; ".join(result.evidence[:3])
                lines.append(f"  evidence: {evidence_str}")

        if verdict.blockers:
            lines.insert(0, f"## {len(verdict.blockers)} blocker(s) — must fix before continuing\n")

        return "\n".join(lines) if lines else "No actionable feedback."

    @staticmethod
    def format_warnings(warnings: list[str]) -> str:
        if not warnings:
            return ""
        header = "## Quality warnings\n"
        body = "\n".join(f"- {w}" for w in warnings)
        return header + body

    @staticmethod
    def format_score_summary(
        lint_passed: bool,
        test_passed: bool,
        rubric_passed: bool,
    ) -> str:
        parts: list[str] = []
        parts.append(f"Lint: {'PASS' if lint_passed else 'FAIL'}")
        parts.append(f"Tests: {'PASS' if test_passed else 'FAIL'}")
        parts.append(f"Rubric: {'PASS' if rubric_passed else 'FAIL'}")
        overall = lint_passed and test_passed and rubric_passed
        return f"[{'PASS' if overall else 'FAIL'}] {' | '.join(parts)}"
