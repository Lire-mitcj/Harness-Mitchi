from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from src.planner.kinds import SubTaskKind, default_needs_l1
from src.planner.tool_policy import default_allowed_tools, normalize_allowed_tools


class SubTaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass
class SubTaskNode:
    """Atomic unit of work dispatched to the Executor."""

    id: str
    description: str
    status: SubTaskStatus = SubTaskStatus.PENDING
    kind: SubTaskKind = SubTaskKind.EDIT
    acceptance_criteria: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    needs_l1: bool | None = None
    context_files: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    requires_artifacts: list[str] = field(default_factory=list)
    produces_artifacts: list[str] = field(default_factory=list)
    write_scope: list[str] = field(default_factory=list)
    checkpoint_id: str | None = None
    error_trace: list[str] = field(default_factory=list)

    def effective_needs_l1(self) -> bool:
        if self.needs_l1 is not None:
            return self.needs_l1
        return default_needs_l1(self.kind)

    def effective_allowed_tools(self) -> frozenset[str]:
        return frozenset(normalize_allowed_tools(self.kind, self.allowed_tools or None))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "kind": self.kind.value,
            "acceptance_criteria": self.acceptance_criteria,
            "allowed_tools": list(self.allowed_tools),
            "needs_l1": self.needs_l1,
            "context_files": list(self.context_files),
            "depends_on": list(self.depends_on),
            "requires_artifacts": list(self.requires_artifacts),
            "produces_artifacts": list(self.produces_artifacts),
            "write_scope": list(self.write_scope),
            "checkpoint_id": self.checkpoint_id,
            "error_trace": list(self.error_trace),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubTaskNode:
        raw_kind = str(data.get("kind", SubTaskKind.EDIT))
        try:
            kind = SubTaskKind(raw_kind)
        except ValueError:
            kind = SubTaskKind.EDIT
        needs_l1 = data.get("needs_l1")
        raw_tools = data.get("allowed_tools")
        allowed_tools = (
            [str(t) for t in raw_tools if isinstance(t, str)]
            if isinstance(raw_tools, list)
            else []
        )
        if not allowed_tools:
            allowed_tools = default_allowed_tools(kind)
        return cls(
            id=str(data["id"]),
            description=str(data["description"]),
            status=SubTaskStatus(str(data.get("status", SubTaskStatus.PENDING))),
            kind=kind,
            acceptance_criteria=str(data.get("acceptance_criteria") or ""),
            allowed_tools=normalize_allowed_tools(kind, allowed_tools),
            needs_l1=needs_l1 if isinstance(needs_l1, bool) else None,
            context_files=[str(p) for p in data.get("context_files") or []],
            depends_on=[str(d) for d in data.get("depends_on") or []],
            requires_artifacts=[str(a) for a in data.get("requires_artifacts") or []],
            produces_artifacts=[str(a) for a in data.get("produces_artifacts") or []],
            write_scope=[str(p) for p in data.get("write_scope") or []],
            checkpoint_id=data.get("checkpoint_id"),
            error_trace=[str(e) for e in data.get("error_trace") or []],
        )


@dataclass
class TaskTree:
    """Ordered plan produced by the Planner and mutated by the Orchestrator."""

    root_task: str
    nodes: list[SubTaskNode] = field(default_factory=list)
    version: int = 1

    def first_pending(self) -> SubTaskNode | None:
        from src.planner.strategy import next_ready_node

        return next_ready_node(self)

    def get(self, node_id: str) -> SubTaskNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def has_pending(self) -> bool:
        return any(node.status == SubTaskStatus.PENDING for node in self.nodes)

    def mark_running(self, node_id: str) -> None:
        node = self._require(node_id)
        node.status = SubTaskStatus.RUNNING

    def mark_success(self, node_id: str, *, checkpoint_id: str | None = None) -> None:
        node = self._require(node_id)
        node.status = SubTaskStatus.SUCCESS
        node.checkpoint_id = checkpoint_id
        node.error_trace.clear()

    def mark_failed(self, node_id: str, *, errors: list[str] | None = None) -> None:
        node = self._require(node_id)
        node.status = SubTaskStatus.FAILED
        if errors:
            node.error_trace.extend(errors)

    def completed_nodes(self) -> list[SubTaskNode]:
        return [n for n in self.nodes if n.status == SubTaskStatus.SUCCESS]

    def remaining_nodes(self) -> list[SubTaskNode]:
        return [
            n
            for n in self.nodes
            if n.status in {SubTaskStatus.PENDING, SubTaskStatus.FAILED, SubTaskStatus.RUNNING}
        ]

    def to_outline(self) -> str:
        icons = {
            SubTaskStatus.PENDING: "○",
            SubTaskStatus.RUNNING: "●",
            SubTaskStatus.SUCCESS: "✓",
            SubTaskStatus.FAILED: "✗",
        }
        from src.cli.report_format import _KIND_ZH

        lines = [f"任务: {self.root_task}", f"计划版本: {self.version}", ""]
        for node in self.nodes:
            icon = icons.get(node.status, "?")
            kind_label = _KIND_ZH.get(node.kind.value, node.kind.value)
            files = ", ".join(node.context_files) if node.context_files else "(无)"
            l1 = "是" if node.effective_needs_l1() else "否"
            tools = ", ".join(sorted(node.effective_allowed_tools()))
            lines.append(f"  {icon} [{node.id}] ({kind_label}) {node.description}")
            if node.acceptance_criteria:
                lines.append(f"      完成条件: {node.acceptance_criteria}")
            lines.append(f"      文件: {files} | 工具: {tools} | L1: {l1}")
            if node.checkpoint_id:
                lines.append(f"      检查点: {node.checkpoint_id}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task": self.root_task,
            "version": self.version,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            root_task=str(data.get("root_task", "")),
            version=int(data.get("version", 1)),
            nodes=[SubTaskNode.from_dict(n) for n in data.get("nodes") or []],
        )

    @classmethod
    def from_json(cls, raw: str) -> Self:
        return cls.from_dict(json.loads(raw))

    def replace_remaining(self, new_nodes: list[SubTaskNode]) -> None:
        """Keep completed nodes; swap out pending/failed/running tail after re-plan."""
        completed = self.completed_nodes()
        self.nodes = completed + new_nodes
        self.version += 1

    def _require(self, node_id: str) -> SubTaskNode:
        node = self.get(node_id)
        if node is None:
            raise KeyError(f"SubTaskNode '{node_id}' not found")
        return node
