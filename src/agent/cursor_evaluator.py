from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_WINDOWS_DRIVE_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\\\/](?P<rest>.*)$")
_COLLAPSED_WINDOWS_EVAL_PATH_RE = re.compile(
    r"^(?P<drive>[A-Za-z]):(?P<rest>eval_json.*)$", re.IGNORECASE
)
_CASE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def resolve_cursor_eval_path(path: Path | str) -> Path:
    """Resolve Windows drive paths to their WSL mount when running on POSIX."""
    raw = str(path).strip()
    match = _WINDOWS_DRIVE_PATH_RE.match(raw)
    if match is None:
        # Some dotenv parsers consume backslashes in unquoted Windows paths.
        # Recover only the known evaluation root instead of treating it as cwd-relative.
        collapsed = _COLLAPSED_WINDOWS_EVAL_PATH_RE.match(raw)
        if collapsed is not None:
            rest = collapsed.group("rest")
            if rest.startswith("eval_jsonretrieval_test_cases.jsonl"):
                rest = "eval_json/cases/retrieval_test_cases.jsonl"
            elif rest.startswith("eval_jsoncases"):
                rest = f"eval_json/cases{rest.removeprefix('eval_jsoncases')}"
            elif rest.startswith("eval_jsonmetrics"):
                rest = f"eval_json/metrics{rest.removeprefix('eval_jsonmetrics')}"
            return (Path("/mnt") / collapsed.group("drive").casefold() / rest).resolve()
    if match is not None and os.name != "nt":
        rest = match.group("rest").replace("\\", "/")
        return (Path("/mnt") / match.group("drive").casefold() / rest).resolve()
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class Layer1Metrics:
    available: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hits: tuple[str, ...] = ()
    misses: tuple[str, ...] = ()
    retrieved: tuple[str, ...] = ()
    expected: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalTestCase:
    """Explicit, non-empty ground truth for one retrieval evaluation run."""

    name: str
    ground_truth: tuple[str, ...]
    description: str = ""

    def validate(self) -> None:
        if not self.ground_truth:
            raise ValueError("GT cannot be empty")

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> RetrievalTestCase:
        raw_gt = data.get("ground_truth", ())
        if not isinstance(raw_gt, (list, tuple)):
            raise ValueError("ground_truth must be a list or tuple")
        case = cls(
            name=str(data.get("name") or "unnamed"),
            ground_truth=tuple(str(item) for item in raw_gt),
            description=str(data.get("description") or ""),
        )
        case.validate()
        return case


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    step: int
    retrieved: tuple[str, ...]
    fused_files: tuple[str, ...]


class RetrievalTestCaseLoader:
    """Load one explicit GT case from JSON, JSON array, or JSONL fixtures."""

    @staticmethod
    def load_json(path: Path) -> RetrievalTestCase:
        cases = RetrievalTestCaseLoader.load_cases(path)
        if len(cases) != 1:
            raise ValueError("evaluation fixture contains multiple cases; select one by name")
        return next(iter(cases.values()))

    @staticmethod
    def load_case(path: Path, name: str | None = None) -> RetrievalTestCase:
        cases = RetrievalTestCaseLoader.load_cases(path)
        if name:
            try:
                return cases[name]
            except KeyError as exc:
                available = ", ".join(sorted(cases))
                raise ValueError(
                    f"evaluation case {name!r} was not found; available: {available}"
                ) from exc
        if len(cases) != 1:
            raise ValueError("evaluation fixture contains multiple cases; set case_name")
        return next(iter(cases.values()))

    @staticmethod
    def load_cases(path: Path) -> dict[str, RetrievalTestCase]:
        raw = resolve_cursor_eval_path(path).read_text(encoding="utf-8")
        entries = _parse_case_entries(raw)
        cases: dict[str, RetrievalTestCase] = {}
        for entry in entries:
            case = RetrievalTestCase.from_mapping(entry)
            if case.name in cases:
                raise ValueError(f"duplicate evaluation case name: {case.name}")
            cases[case.name] = case
        if not cases:
            raise ValueError("evaluation fixture contains no cases")
        return cases

    @staticmethod
    def select_case(
        path: Path,
        query_terms: tuple[str, ...],
    ) -> RetrievalTestCase:
        terms = _case_tokens(" ".join(query_terms))
        if not terms:
            raise ValueError("cannot auto-select evaluation case without searchable terms")
        scored: list[tuple[int, str, RetrievalTestCase]] = []
        for name, case in RetrievalTestCaseLoader.load_cases(path).items():
            overlap = len(_case_tokens(name) & terms)
            if overlap:
                scored.append((overlap, name, case))
        if not scored:
            raise ValueError("no evaluation case matches the query terms")
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, _name, best_case = scored[0]
        if len(scored) > 1 and scored[1][0] == best_score:
            tied = ", ".join(name for score, name, _case in scored if score == best_score)
            raise ValueError(f"ambiguous evaluation case auto-selection: {tied}")
        return best_case


def _parse_case_entries(raw: str) -> tuple[dict[str, object], ...]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        entries: list[dict[str, object]] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL case at line {line_no}: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"JSONL case at line {line_no} must be an object")
            entries.append(entry)
        return tuple(entries)
    if isinstance(data, dict):
        return (data,)
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return tuple(data)
    raise ValueError("evaluation fixture must be a JSON object, array, or JSONL objects")


def _case_tokens(value: str) -> set[str]:
    return {token.casefold() for token in _CASE_TOKEN_RE.findall(value)}


class CursorEvalHarnessV2:
    """Accumulates real runtime traces and scores them against bound GT only."""

    def __init__(self, test_case: RetrievalTestCase) -> None:
        test_case.validate()
        self.test_case = test_case
        self.traces: list[RetrievalTrace] = []

    def add_trace(self, trace: RetrievalTrace) -> None:
        self.traces.append(trace)

    def evaluate(self) -> Layer1Metrics:
        expected = tuple(dict.fromkeys(self.test_case.ground_truth))
        retrieved = tuple(dict.fromkeys(
            file
            for trace in self.traces
            for file in trace.fused_files
        ))
        return compute_layer1_metrics(retrieved, expected)


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
        self.output_dir = resolve_cursor_eval_path(output_dir)
        self.path = self.output_dir / filename

    def record(self, metrics: FullPipelineMetrics) -> None:
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
        available=bool(expected),
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
    if layer1.available:
        precision_line = (
            f"  ├── Precision : {layer1.precision:.4f} "
            f"({len(layer1.hits)}/{len(layer1.retrieved)})"
        )
        recall_line = (
            f"  ├── Recall    : {layer1.recall:.4f} "
            f"({len(layer1.hits)}/{len(layer1.expected)})"
        )
    else:
        precision_line = "  ├── Precision : N/A (ground truth unavailable)"
        recall_line = "  ├── Recall    : N/A (ground truth unavailable)"
    return "\n".join((
        f"{icon} [{status}] Universal Self-Healing Step Evaluation (Step {metrics.step})",
        "====================================================================",
        '🟢 LAYER 1: RETRIEVAL METRICS (RAG Analysis - "有没有找到")',
        precision_line,
        recall_line,
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
