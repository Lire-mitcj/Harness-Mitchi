from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Layer1Metrics:
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hits: tuple[str, ...] = ()
    misses: tuple[str, ...] = ()
    retrieved: tuple[str, ...] = ()
    expected: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Layer2Metrics:
    patch_correctness: float
    execution_success: float
    code_diff_correctness: float
    task_passed: bool
    observation: str = ""
    committed: bool = False
    validation: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FullPipelineMetrics:
    step: int
    target_file: str
    layer1: Layer1Metrics = field(default_factory=Layer1Metrics)
    layer2: Layer2Metrics = field(
        default_factory=lambda: Layer2Metrics(
            patch_correctness=0.0,
            execution_success=0.0,
            code_diff_correctness=0.0,
            task_passed=False,
        )
    )

    @property
    def task_passed(self) -> bool:
        return self.layer2.task_passed


class CursorEvaluator:
    def __init__(
        self,
        output_dir: Path,
        *,
        filename: str = "harness_evaluation_metrics.jsonl",
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.path = self.output_dir / filename

    def record(self, metrics: FullPipelineMetrics) -> None:
        self.emit_console(metrics)
        self.append_jsonl(metrics)

    def append_jsonl(self, metrics: FullPipelineMetrics) -> None:
        payload = asdict(metrics)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def emit_console(self, metrics: FullPipelineMetrics) -> None:
        log.warning("\n%s", format_bi_report(metrics))


def compute_layer1_metrics(
    retrieved: tuple[str, ...],
    expected: tuple[str, ...] = (),
) -> Layer1Metrics:
    retrieved_set = set(retrieved)
    expected_set = set(expected)
    hits = tuple(item for item in retrieved if item in expected_set)
    misses = tuple(item for item in expected if item not in retrieved_set)
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(expected) if expected else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return Layer1Metrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1_score, 4),
        hits=hits,
        misses=misses,
        retrieved=retrieved,
        expected=expected,
    )


def format_bi_report(metrics: FullPipelineMetrics) -> str:
    status = (
        "TRANSACTION COMMIT SUCCESS"
        if metrics.layer2.task_passed
        else "TRANSACTION ROLLBACK"
    )
    icon = "🟩" if metrics.layer2.task_passed else "🟥"
    layer1 = metrics.layer1
    layer2 = metrics.layer2
    return "\n".join((
        f"{icon} [{status}] Universal Self-Healing Step Evaluation (Step {metrics.step})",
        "====================================================================",
        '🟢 LAYER 1: RETRIEVAL METRICS (RAG Analysis - "有没有找到")',
        f"  ├── Precision : {layer1.precision:.4f} ({len(layer1.hits)}/{len(layer1.retrieved)})",
        f"  ├── Recall    : {layer1.recall:.4f} ({len(layer1.hits)}/{len(layer1.expected)})",
        f"  ├── F1-Score  : {layer1.f1_score:.4f}",
        f"  ├── Hits [✅] : {list(layer1.hits)}",
        f"  └── Misses [❌]: {list(layer1.misses)}",
        "",
        '🔴 LAYER 2: TASK SUCCESS METRICS (Execution Sandbox - "有没有解决")',
        f"  ├── Patch Correctness     : {layer2.patch_correctness:.1f} "
        "(SEARCH/REPLACE Structural Match)",
        f"  ├── Execution Success     : {layer2.execution_success:.1f} "
        "(Subprocess Verification & Pytest)",
        f"  ├── Code Diff Correctness : {layer2.code_diff_correctness:.1f} "
        "(Immutable Sandbox Protection)",
        f"  └── Validator Decision    : {layer2.validation.get('decision', 'unknown')}",
        "====================================================================",
    ))
