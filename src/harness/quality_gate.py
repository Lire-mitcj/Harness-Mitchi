from __future__ import annotations

import asyncio
import difflib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.harness.scorer.engine import ScoringContext

if TYPE_CHECKING:
    from src.harness.engine import HarnessEngine

log = logging.getLogger(__name__)


async def build_diff(harness: HarnessEngine, changed_files: list[str]) -> str:
    rel_files: list[str] = []
    abs_files: list[Path] = []
    for f in changed_files:
        try:
            p = Path(f)
            if not p.is_absolute():
                p = (harness.project_root / p).resolve()
            else:
                p = p.resolve()
            rel = p.relative_to(harness.project_root)
            rel_files.append(str(rel))
            abs_files.append(p)
        except Exception:
            continue
    if not rel_files:
        return ""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "--",
            *sorted(set(rel_files)),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(harness.project_root),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        git_diff = stdout.decode(errors="replace")
        if git_diff.strip():
            return git_diff
        return _fallback_diff_from_files(harness, abs_files)
    except Exception:
        return _fallback_diff_from_files(harness, abs_files)


def _fallback_diff_from_files(harness: HarnessEngine, files: list[Path]) -> str:
    chunks: list[str] = []
    for p in files:
        try:
            rel = str(p.relative_to(harness.project_root))
        except Exception:
            continue
        if not p.exists():
            chunks.append(f"--- a/{rel}\n+++ /dev/null\n@@\n- <deleted file>\n")
            continue
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = content.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff([], lines, fromfile="/dev/null", tofile=f"b/{rel}", n=3)
        )
        if diff_lines:
            chunks.append("".join(diff_lines))
    return "\n".join(chunks)


async def evaluate_quality_gate(
    harness: HarnessEngine,
    *,
    user_msg: str,
    changed_files: list[str],
    skip_l1: bool = False,
    acceptance_criteria: str = "",
) -> dict[str, Any] | None:
    """Run L0/L1/L2 scoring and return a normalized gate payload."""
    if not changed_files:
        return None
    try:
        diff_text = await build_diff(harness, changed_files)
        context = ScoringContext(
            user_message=user_msg,
            acceptance_criteria=acceptance_criteria,
            diff=diff_text,
            changed_files=changed_files,
            recent_tool_calls=changed_files,
            project_root=harness.project_root,
        )
        score_result = await harness.scorer.evaluate(context, skip_l1=skip_l1)
        lint = score_result.scores.get("lint", {})
        tests = score_result.scores.get("tests", {})
        rubric = score_result.scores.get("rubric", {})
        l0_passed = bool(lint.get("passed", True)) and bool(tests.get("passed", True))
        l1_passed = skip_l1 or (
            (not isinstance(rubric, dict)) or rubric.get("verdict", "pass") == "pass"
        )
        l2_warning_only = (
            bool(score_result.warnings) and score_result.passed and not score_result.needs_retry
        )
        auto_rewrite = (not l0_passed) or (not l1_passed) or bool(score_result.needs_retry)
        blockers = rubric.get("blockers", []) if isinstance(rubric, dict) else []
        checks: list[dict[str, Any]] = []
        for name, data in score_result.scores.items():
            message = data.get("details", "") if isinstance(data, dict) else ""
            passed = bool(data.get("passed", False)) if isinstance(data, dict) else False
            if name == "rubric":
                passed = l1_passed
                if blockers:
                    message = blockers[0]
                elif isinstance(rubric, dict):
                    message = f"verdict={rubric.get('verdict', 'unknown')}"
            checks.append({"name": name, "passed": passed, "message": message})

        result: dict[str, Any] = {
            "passed": score_result.passed,
            "needs_retry": score_result.needs_retry,
            "feedback": score_result.feedback,
            "warnings": score_result.warnings,
            "gate": "PASS" if not auto_rewrite else "FAIL",
            "l0_passed": l0_passed,
            "l1_passed": l1_passed,
            "l2_warning_only": l2_warning_only,
            "auto_rewrite": auto_rewrite,
            "checks": checks,
            "blockers": blockers,
            "blocker_count": len(blockers),
            "l1_skipped": skip_l1,
        }
        return result
    except Exception as exc:
        log.debug("Quality gate unavailable: %s", exc)
        return None


def scorer_signature(score_data: dict[str, Any]) -> str:
    feedback = (score_data.get("feedback") or "").strip()
    checks = score_data.get("checks") or []
    check_bits = "|".join(
        f"{c.get('name')}:{c.get('passed')}:{c.get('message', '')}"
        for c in checks
        if isinstance(c, dict)
    )
    return (
        f"gate={score_data.get('gate')}|"
        f"blockers={score_data.get('blocker_count')}|"
        f"retry={score_data.get('needs_retry')}|"
        f"{feedback}|{check_bits}"
    )
