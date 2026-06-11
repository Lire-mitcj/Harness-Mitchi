from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.harness.quality_gate import evaluate_quality_gate
from src.harness.scorer.code_quality import LayerScore
from src.harness.scorer.engine import ScoreResult, ScoringContext, ScoringEngine


class _FakeScorer:
    def __init__(self, result: ScoreResult) -> None:
        self.result = result

    async def evaluate(self, context: ScoringContext, *, skip_l1: bool = False) -> ScoreResult:
        return self.result


class _FakeHarness:
    def __init__(self, project_root: Path, result: ScoreResult) -> None:
        self.project_root = project_root
        self.scorer = _FakeScorer(result)


def _passing_l0_scores() -> dict[str, Any]:
    return {
        "lint": {"passed": True, "details": "ok"},
        "tests": {"passed": True, "details": "ok"},
    }


@pytest.mark.asyncio
async def test_quality_gate_does_not_rewrite_for_l2_warnings(tmp_path: Path) -> None:
    changed = tmp_path / "app.py"
    changed.write_text("x = 1\n", encoding="utf-8")
    result = ScoreResult(
        passed=True,
        scores={
            **_passing_l0_scores(),
            "rubric": {
                "verdict": "fail",
                "blockers": [],
                "warnings": ["[QL-01] readability"],
            },
        },
        feedback="## Quality warnings\n- [QL-01] readability",
        needs_retry=False,
        warnings=["[QL-01] readability"],
    )

    gate = await evaluate_quality_gate(
        _FakeHarness(tmp_path, result),  # type: ignore[arg-type]
        user_msg="make the change",
        changed_files=["app.py"],
    )

    assert gate is not None
    assert gate["gate"] == "PASS"
    assert gate["auto_rewrite"] is False
    assert gate["l2_warning_only"] is True
    assert gate["l1_passed"] is True


@pytest.mark.asyncio
async def test_quality_gate_rewrites_for_l1_blockers(tmp_path: Path) -> None:
    changed = tmp_path / "app.py"
    changed.write_text("x = 1\n", encoding="utf-8")
    result = ScoreResult(
        passed=False,
        scores={
            **_passing_l0_scores(),
            "rubric": {
                "verdict": "fail",
                "blockers": ["[TC-01] missing requested behavior"],
                "warnings": [],
            },
        },
        feedback="missing requested behavior",
        needs_retry=True,
        retry_reasons=["L1"],
    )

    gate = await evaluate_quality_gate(
        _FakeHarness(tmp_path, result),  # type: ignore[arg-type]
        user_msg="make the change",
        changed_files=["app.py"],
    )

    assert gate is not None
    assert gate["gate"] == "FAIL"
    assert gate["auto_rewrite"] is True
    assert gate["rewrite_reasons"] == ["L1"]


@pytest.mark.asyncio
async def test_scoring_engine_treats_warning_only_verdict_as_l2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def judge_response(*args: Any, **kwargs: Any) -> str:
        return json.dumps({
            "verdict": "fail",
            "results": [
                {
                    "id": "QL-01",
                    "passed": False,
                    "severity": "warning",
                    "reason": "readability issue",
                    "evidence": ["diff: app.py hunk"],
                }
            ],
            "blockers": [],
            "warnings": ["QL-01"],
        })

    engine = ScoringEngine(llm_client=judge_response, project_root=tmp_path)
    monkeypatch.setattr(
        engine._lint,  # noqa: SLF001 - targeted unit fixture
        "run_lint",
        lambda context: _async_layer("L0:lint", True, "ok"),
    )
    monkeypatch.setattr(
        engine._tests,  # noqa: SLF001 - targeted unit fixture
        "run",
        lambda context: _async_layer("L0:test", True, "ok"),
    )

    result = await engine.evaluate(
        ScoringContext(user_message="change app", diff="diff --git a/app.py b/app.py")
    )

    assert result.passed is True
    assert result.needs_retry is False
    assert result.retry_reasons == []
    assert result.warnings == ["[QL-01] readability issue"]


async def _async_layer(layer: str, passed: bool, details: str) -> LayerScore:
    return LayerScore(layer=layer, passed=passed, details=details)
