from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    step: int
    target_file: str
    line_number: int | None = None
    offset: int | None = None
    exception_type: str = ""
    error_msg: str = ""


@dataclass(frozen=True, slots=True)
class PatchMemory:
    step: int
    target_file: str
    patch_content: str
    rollback_reason: str
    diff_status: str
    base_snapshot: str = ""
    attempted_snapshot: str = ""


@dataclass(frozen=True, slots=True)
class CursorState:
    task: str
    current_file: str
    last_patch: str
    last_observation: str
    status: Literal["running", "success", "failed"]
    current_step: int = 1
    max_steps: int = 10
    stage_completion: float = 0.0
    execution_traces: tuple[ExecutionTrace, ...] = field(default_factory=tuple)
    patch_memory: tuple[PatchMemory, ...] = field(default_factory=tuple)
    decision_signatures: tuple[str, ...] = field(default_factory=tuple)
    retry_bias: int = 0
    decision_cost_total: float = 0.0
    entropy_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

