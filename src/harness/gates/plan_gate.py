from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.harness.gates.types import GateResult
from src.planner.context_policy import enrich_task_tree_context_files
from src.planner.dependency_graph import DependencyGraph
from src.planner.task_tree import SubTaskKind, SubTaskNode, TaskTree
from src.planner.tool_policy import validate_node_tools

_MUTATION_HINTS = re.compile(r"\b(INSERT|DELETE|UPDATE|TRUNCATE|DROP)\b", re.I)
_DIAGNOSE_HINTS = re.compile(r"\b(DESCRIBE|SHOW CREATE|schema|读表|表结构|诊断)\b", re.I)
_LOW_VALUE_DIAGNOSE_ACTIONS = re.compile(
    r"\b(read|inspect|analy[sz]e|search|grep|map-search|look\s+at|查看|阅读|读取|分析|搜索|查找)\b",
    re.I,
)
_HANDOFF_FILE_LINE = re.compile(
    r"(file\s*:?\s*line|path\s*:?\s*line|file:line|line\s+range|行号|行范围|路径.*行|文件.*行)",
    re.I,
)
_HANDOFF_SYMBOL = re.compile(
    r"\b(symbol|function|class|method|view|query|sql)\b|符号|函数|方法|类|视图|查询",
    re.I,
)
_HANDOFF_SNIPPET = re.compile(
    r"\b(snippet|excerpt|decision|code|sql)\b|片段|代码|决策|结论",
    re.I,
)


@dataclass(frozen=True)
class ReplanGateContext:
    """Failed subtask snapshot used to reject identical re-plans."""

    failed_subtask_id: str
    failed_kind: SubTaskKind
    failed_description: str
    failed_context_files: tuple[str, ...] = ()
    failed_acceptance_criteria: str = ""

    @classmethod
    def from_node(cls, node: SubTaskNode) -> ReplanGateContext:
        return cls(
            failed_subtask_id=node.id,
            failed_kind=node.kind,
            failed_description=node.description,
            failed_context_files=tuple(node.context_files),
            failed_acceptance_criteria=node.acceptance_criteria or "",
        )


def _normalize_description(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _normalize_path_set(paths: tuple[str, ...] | list[str]) -> frozenset[str]:
    out: set[str] = set()
    for raw in paths:
        rel = raw.replace("\\", "/").strip().lstrip("./")
        if rel:
            out.add(rel)
    return frozenset(out)


def find_replan_duplicates(tree: TaskTree, ctx: ReplanGateContext) -> list[str]:
    """Return PlanGate blocks when re-plan repeats the failed subtask unchanged."""
    blocks: list[str] = []
    failed_desc = _normalize_description(ctx.failed_description)
    failed_ctx = _normalize_path_set(ctx.failed_context_files)

    for node in tree.nodes:
        if _normalize_description(node.description) != failed_desc:
            continue
        if node.kind != ctx.failed_kind:
            continue

        if ctx.failed_kind == SubTaskKind.EDIT:
            node_ctx = _normalize_path_set(node.context_files)
            if node_ctx != failed_ctx:
                continue

        blocks.append(
            f"Re-plan subtask [{node.id}] duplicates failed [{ctx.failed_subtask_id}]: "
            f"same kind={node.kind.value}, description"
            + (
                f", and context_files ({', '.join(sorted(failed_ctx)) or 'none'})"
                if ctx.failed_kind == SubTaskKind.EDIT
                else ""
            )
            + ". Split into diagnose + a different edit step, or widen context_files."
        )
    return blocks


def find_replan_structure_repeats(tree: TaskTree, ctx: ReplanGateContext) -> list[str]:
    """Block replans that only rename the failed step while keeping the same shape."""
    blocks: list[str] = []
    failed_ctx = _normalize_path_set(ctx.failed_context_files)
    for node in tree.nodes:
        if node.kind != ctx.failed_kind:
            continue
        if ctx.failed_kind == SubTaskKind.EDIT:
            node_ctx = _normalize_path_set(node.context_files)
            if node_ctx != failed_ctx:
                continue
            if _normalize_description(node.acceptance_criteria) != _normalize_description(
                ctx.failed_acceptance_criteria
            ):
                continue
        else:
            if _normalize_description(node.acceptance_criteria) != _normalize_description(
                ctx.failed_acceptance_criteria
            ):
                continue
        blocks.append(
            f"Re-plan subtask [{node.id}] repeats failed [{ctx.failed_subtask_id}] structure: "
            "same kind, context_files, and acceptance output. Change the milestone: add a "
            "diagnose handoff, widen/narrow context_files, or change the required output."
        )
    return blocks


def validate_plan(
    tree: TaskTree,
    project_root: Path,
    *,
    max_nodes: int = 4,
    replan_context: ReplanGateContext | None = None,
) -> GateResult:
    """Rule-based PlanGate on a parsed TaskTree — no LLM calls."""
    enrich_task_tree_context_files(tree)

    blocks: list[str] = []
    warns: list[str] = []
    actions: list[str] = ["re_plan"]

    if replan_context is not None:
        blocks.extend(find_replan_duplicates(tree, replan_context))
        blocks.extend(find_replan_structure_repeats(tree, replan_context))

    if not tree.nodes:
        blocks.append("TaskTree has no subtasks.")
    elif len(tree.nodes) > max_nodes:
        blocks.append(f"TaskTree has {len(tree.nodes)} subtasks (max {max_nodes}).")

    ids = [n.id for n in tree.nodes]
    if len(ids) != len(set(ids)):
        blocks.append("Duplicate subtask id in TaskTree.")

    id_set = set(ids)
    graph = DependencyGraph()
    for node in tree.nodes:
        graph.add_node(node.id)
        if not node.description.strip():
            blocks.append(f"Subtask [{node.id}] has empty description.")
        elif len(node.description) > 500:
            blocks.append(f"Subtask [{node.id}] description exceeds 500 characters.")
        if not node.acceptance_criteria.strip():
            blocks.append(
                f"Subtask [{node.id}] missing acceptance_criteria output/deliverable."
            )
        for dep in node.depends_on:
            if dep not in id_set:
                blocks.append(f"Subtask [{node.id}] depends on unknown id '{dep}'.")
            else:
                graph.add_edge(dep, node.id)
        if node.kind == SubTaskKind.EDIT and not node.context_files:
            if re.search(r"\.\w+", node.description):
                warns.append(
                    f"Subtask [{node.id}] kind=edit but context_files is empty — "
                    "executor may lack file whitelist."
                )
        if node.kind == SubTaskKind.EDIT:
            desc = node.description.lower()
            if any(k in desc for k in ("schema", "procedure", "sp_", "sql", "存储过程", "事务")):
                if not any(f.endswith(".sql") for f in node.context_files):
                    warns.append(
                        f"Subtask [{node.id}] edit involves DB/schema but context_files "
                        "has no .sql — executor cannot read schema files."
                    )

        tool_blocks, tool_warns = validate_node_tools(node)
        blocks.extend(tool_blocks)
        warns.extend(tool_warns)

    if graph.has_cycle():
        blocks.append("TaskTree dependency cycle detected.")

    blocks.extend(_check_milestone_structure(tree))

    root = project_root.resolve()
    for node in tree.nodes:
        for rel in node.context_files:
            path = _resolve_under_root(root, rel)
            if path is None:
                blocks.append(f"Subtask [{node.id}] context file outside project: {rel}")
            elif not path.is_file():
                warns.append(f"Subtask [{node.id}] context file missing: {rel}")

    _check_db_step_order(tree, warns)

    if blocks:
        return GateResult.block("plan_gate", blocks, actions=["re_plan"])

    if warns:
        return GateResult.warn("plan_gate", warns, actions=actions, node_count=len(tree.nodes))

    return GateResult.pass_("plan_gate", node_count=len(tree.nodes))


def _resolve_under_root(project_root: Path, rel: str) -> Path | None:
    try:
        path = (project_root / rel.replace("\\", "/").lstrip("./")).resolve()
        path.relative_to(project_root)
        return path
    except (ValueError, OSError):
        return None


def _check_db_step_order(tree: TaskTree, warns: list[str]) -> None:
    seen_diagnose = False
    for node in tree.nodes:
        desc = node.description
        if _DIAGNOSE_HINTS.search(desc):
            seen_diagnose = True
        if _MUTATION_HINTS.search(desc) and not seen_diagnose:
            warns.append(
                f"Subtask [{node.id}] may mutate data before a schema/diagnose step — "
                "consider reordering."
            )


def _check_milestone_structure(tree: TaskTree) -> list[str]:
    blocks: list[str] = []
    by_id = {node.id: node for node in tree.nodes}
    edit_deps: dict[str, list[SubTaskNode]] = {}
    for node in tree.nodes:
        if node.kind != SubTaskKind.EDIT:
            continue
        for dep_id in node.depends_on:
            dep = by_id.get(dep_id)
            if dep and dep.kind == SubTaskKind.DIAGNOSE:
                edit_deps.setdefault(dep.id, []).append(node)

    for diag_id, edits in edit_deps.items():
        diag = by_id[diag_id]
        if _is_low_value_diagnose(diag, edits):
            blocks.append(
                f"Subtask [{diag.id}] is a low-value read/search/analyze step before edit. "
                "Merge scoped read/grep into the edit subtask, or make diagnose produce a "
                "concrete milestone output."
            )
        if not _diagnose_handoff_complete(diag.acceptance_criteria):
            blocks.append(
                f"Subtask [{diag.id}] feeds edit but acceptance_criteria lacks required "
                "handoff output: file:line, symbol, and snippet/decision."
            )
    return blocks


def _is_low_value_diagnose(diag: SubTaskNode, edits: list[SubTaskNode]) -> bool:
    text = f"{diag.description} {diag.acceptance_criteria}"
    if not _LOW_VALUE_DIAGNOSE_ACTIONS.search(text):
        return False
    if _diagnose_handoff_complete(diag.acceptance_criteria):
        return False
    diag_ctx = _normalize_path_set(diag.context_files)
    if not diag_ctx:
        return True
    return any(diag_ctx <= _normalize_path_set(edit.context_files) for edit in edits)


def _diagnose_handoff_complete(text: str) -> bool:
    return bool(
        _HANDOFF_FILE_LINE.search(text)
        and _HANDOFF_SYMBOL.search(text)
        and _HANDOFF_SNIPPET.search(text)
    )
