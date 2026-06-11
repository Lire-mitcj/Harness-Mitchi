from __future__ import annotations

from enum import Enum

from src.planner.kinds import SubTaskKind
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


def ready_execution_batches(tree: TaskTree) -> list[list[SubTaskNode]]:
    """Group ready pending nodes into deterministic batches with no write conflicts.

    The Orchestrator can execute a batch sequentially today or parallelize it later.
    Edit nodes that share a write scope are split into later batches unless the
    plan already serialized them through depends_on.
    """
    batches: list[list[SubTaskNode]] = []
    for node in ready_pending_nodes(tree):
        for batch in batches:
            if not any(_write_conflict(node, existing) for existing in batch):
                batch.append(node)
                break
        else:
            batches.append([node])
    return batches


def batch_parallelizable(batch: list[SubTaskNode]) -> bool:
    """Return True for batches that can run concurrently without write semantics."""
    if len(batch) <= 1:
        return False
    return all(node.kind in {SubTaskKind.DIAGNOSE, SubTaskKind.VERIFY} for node in batch)


def next_ready_node(tree: TaskTree) -> SubTaskNode | None:
    """Return the first node from the current ready execution frontier."""
    batches = ready_execution_batches(tree)
    return batches[0][0] if batches and batches[0] else None


def _write_conflict(left: SubTaskNode, right: SubTaskNode) -> bool:
    if left.kind != SubTaskKind.EDIT or right.kind != SubTaskKind.EDIT:
        return False
    left_scope = _write_scope(left)
    right_scope = _write_scope(right)
    return bool(left_scope and right_scope and left_scope & right_scope)


def _write_scope(node: SubTaskNode) -> frozenset[str]:
    paths: set[str] = set()
    for raw in [*node.context_files, *node.write_scope]:
        rel = raw.replace("\\", "/").strip().lstrip("./")
        if rel:
            paths.add(rel)
    return frozenset(paths)
