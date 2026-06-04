from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.harness.scorer.rubric_loader import Rubric

log = logging.getLogger(__name__)

_JUDGE_PROMPT_PATH = (
    Path(__file__).resolve().parent / "rubrics" / "judge_prompt.md"
)

_JUDGE_FALLBACK = (
    "You are a code-review judge. Output JSON with verdict, results, blockers, warnings."
)


def load_judge_system_prompt() -> str:
    if _JUDGE_PROMPT_PATH.is_file():
        return _JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    log.warning("judge_prompt.md missing, using embedded fallback")
    return _JUDGE_FALLBACK

@dataclass(slots=True)
class RubricResult:
    id: str
    passed: bool
    severity: str
    reason: str
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RubricVerdict:
    verdict: str  # "pass" | "fail" | "partial"
    results: list[RubricResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


class LLMJudge:
    """L1 scorer: uses an LLM to evaluate the agent's work against rubrics."""

    def __init__(
        self,
        llm_client: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self._llm_client = llm_client

    async def judge(
        self,
        context: Any,
        rubrics: list[Rubric],
    ) -> RubricVerdict:
        if self._llm_client is None:
            return RubricVerdict(verdict="pass")

        prompt = self._build_judge_prompt(
            user_message=getattr(context, "user_message", ""),
            acceptance_criteria=getattr(context, "acceptance_criteria", ""),
            diff=getattr(context, "diff", ""),
            tool_log=getattr(context, "recent_tool_calls", []),
            project_rules=getattr(context, "rules_md_summary", ""),
            rubrics=rubrics,
        )

        raw = await self._llm_client(messages=prompt)
        return self._parse_response(raw, rubrics)

    def _build_judge_prompt(
        self,
        user_message: str,
        acceptance_criteria: str,
        diff: str,
        tool_log: list[Any],
        project_rules: str,
        rubrics: list[Rubric],
    ) -> list[dict[str, Any]]:
        rubric_text = "\n".join(
            f"- [{r.id}] ({r.severity}) {r.question}\n"
            f"  pass: {r.pass_criteria}\n"
            f"  fail: {r.fail_criteria}\n"
            f"  evidence_required: {r.evidence_required}"
            for r in rubrics
        )

        tool_summary = "\n".join(
            f"- {t}" if isinstance(t, str) else f"- {json.dumps(t)}"
            for t in (tool_log or [])[:20]
        )

        user_parts = [
            f"## User request\n{user_message}\n",
        ]
        if acceptance_criteria.strip():
            user_parts.append(f"## Acceptance criteria\n{acceptance_criteria.strip()}\n")
        user_parts.extend([
            f"## Diff\n```\n{diff[:8000]}\n```\n",
            f"## Tool log (recent)\n{tool_summary}\n",
            f"## Project rules\n{project_rules}\n",
            f"## Rubrics\n{rubric_text}",
        ])

        return [
            {"role": "system", "content": load_judge_system_prompt()},
            {"role": "user", "content": "".join(user_parts)},
        ]

    def _parse_response(
        self,
        raw: str,
        rubrics: list[Rubric],
    ) -> RubricVerdict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            log.warning("LLM judge returned unparseable response, treating as fail")
            return RubricVerdict(
                verdict="fail",
                blockers=["Judge response was not valid JSON"],
            )

        results: list[RubricResult] = []
        blockers: list[str] = []
        warnings: list[str] = []

        rubric_map = {r.id: r for r in rubrics}

        for item in data.get("results", []):
            rid = item.get("id", "unknown")
            passed = item.get("passed", False)
            evidence = item.get("evidence", [])
            rubric = rubric_map.get(rid)

            if rubric and rubric.evidence_required and not evidence:
                passed = False
                item["reason"] = item.get("reason", "") + " (missing evidence → Fail)"

            result = RubricResult(
                id=rid,
                passed=passed,
                severity=item.get("severity", "warning"),
                reason=item.get("reason", ""),
                evidence=evidence,
            )
            results.append(result)

            if not passed:
                msg = f"[{rid}] {result.reason}"
                if result.severity == "blocker":
                    blockers.append(msg)
                else:
                    warnings.append(msg)

        verdict = data.get("verdict", "fail")
        if blockers and verdict == "pass":
            verdict = "fail"

        return RubricVerdict(
            verdict=verdict,
            results=results,
            blockers=blockers,
            warnings=warnings,
        )
