from __future__ import annotations

from enum import Enum

from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree


class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MIXED = "mixed"


def select_strategy(tree: TaskTree) -> ExecutionStrategy:
    """Choose execution strategy based on subtask dependencies."""
    pending = [n for n in tree.nodes if n.status == SubTaskStatus.PENDING]
    if len(pending) <= 1:
        return ExecutionStrategy.SEQUENTIAL

    has_deps = any(n.depends_on for n in pending)
    if not has_deps:
        return ExecutionStrategy.PARALLEL

    independent_count = sum(1 for n in pending if not n.depends_on)
    if independent_count > 1:
        return ExecutionStrategy.MIXED

    return ExecutionStrategy.SEQUENTIAL


def ready_pending_nodes(tree: TaskTree) -> list[SubTaskNode]:
    """Return pending nodes whose dependencies are all SUCCESS."""
    completed = {n.id for n in tree.nodes if n.status == SubTaskStatus.SUCCESS}
    ready: list[SubTaskNode] = []
    for node in tree.nodes:
        if node.status != SubTaskStatus.PENDING:
            continue
        if all(dep in completed for dep in node.depends_on):
            ready.append(node)
    return ready
