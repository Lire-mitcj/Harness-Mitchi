from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from src.harness.gates.types import GateResult
from src.harness.task_analysis import HarnessTaskAnalysis, is_edit_ready
from src.planner.context_policy import enrich_task_tree_context_files
from src.planner.dependency_graph import DependencyGraph
from src.planner.task_tree import SubTaskKind, SubTaskNode, SubTaskStatus, TaskTree
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
_FUNCTION_REFACTOR_HINTS = re.compile(
    r"\b(refactor|replace\s+function|replace\s+\w+\s+with\s+\w+|reuse\s+existing\s+function|normalize|masking)\b"
    r"|重构|复用|替换函数|替换方法|统一逻辑|脱敏|提取.*helper|多个接口|多个调用点",
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


def _semantic_key(text: str) -> str:
    text = _normalize_description(text)
    text = re.sub(r"^(定位目标代码|定位|查找|搜索|修改|修复|验证|run|fix|patch)[:：\s]+", "", text)
    text = re.sub(r"[\s,，。.!！?？:：;；、\"'`]+", "", text)
    return text


def _compact_raw(text: str) -> str:
    text = _normalize_description(text)
    return re.sub(r"[\s,，。.!！?？:：;；、\"'`]+", "", text)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


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
        if node.status == SubTaskStatus.SUCCESS:
            continue
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
        if node.status == SubTaskStatus.SUCCESS:
            continue
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
    task_analysis: HarnessTaskAnalysis | None = None,
) -> GateResult:
    """Rule-based PlanGate on a parsed TaskTree — no LLM calls."""
    enrich_task_tree_context_files(tree)

    blocks: list[str] = []
    warns: list[str] = []
    actions: list[str] = ["re_plan"]

    if replan_context is not None:
        blocks.extend(find_replan_duplicates(tree, replan_context))
        blocks.extend(find_replan_structure_repeats(tree, replan_context))
        blocks.extend(_check_replan_preserves_failed_objective(tree, replan_context))

    effective_max_nodes = _effective_max_nodes(max_nodes, task_analysis)

    if not tree.nodes:
        blocks.append("TaskTree has no subtasks.")
    elif len(tree.nodes) > effective_max_nodes:
        blocks.append(f"TaskTree has {len(tree.nodes)} subtasks (max {effective_max_nodes}).")

    ids = [n.id for n in tree.nodes]
    if len(ids) != len(set(ids)):
        blocks.append("Duplicate subtask id in TaskTree.")

    id_set = set(ids)
    graph = DependencyGraph()
    for node in tree.nodes:
        graph.add_node(node.id)
        for dep in node.depends_on:
            if dep not in id_set:
                blocks.append(f"Subtask [{node.id}] depends on unknown id '{dep}'.")
            else:
                graph.add_edge(dep, node.id)

        if node.status == SubTaskStatus.SUCCESS:
            continue

        if not node.description.strip():
            blocks.append(f"Subtask [{node.id}] has empty description.")
        elif len(node.description) > 500:
            blocks.append(f"Subtask [{node.id}] description exceeds 500 characters.")
        if not node.acceptance_criteria.strip():
            blocks.append(
                f"Subtask [{node.id}] missing acceptance_criteria output/deliverable."
            )
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

    if task_analysis is not None:
        blocks.extend(_check_harness_task_policy(tree, task_analysis))
    blocks.extend(_check_required_diagnose_first(tree))
    blocks.extend(_check_distinct_milestones(tree))
    blocks.extend(_check_milestone_structure(tree))
    blocks.extend(_check_handoff_dependency_structure(tree))

    root = project_root.resolve()
    for node in tree.nodes:
        if node.status == SubTaskStatus.SUCCESS:
            continue
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


def _effective_max_nodes(
    configured_max: int,
    task_analysis: HarnessTaskAnalysis | None,
) -> int:
    if task_analysis is None:
        return configured_max
    target_count = len(task_analysis.editable_targets or ())
    if target_count <= 1:
        return configured_max
    # diagnose + design + one edit per target + verify
    split_target_max = target_count + 3
    return max(configured_max, split_target_max)


def _check_harness_task_policy(
    tree: TaskTree,
    analysis: HarnessTaskAnalysis,
) -> list[str]:
    blocks: list[str] = []
    if not tree.nodes:
        return blocks

    kinds = [node.kind for node in tree.nodes]
    first = tree.nodes[0]
    if first.kind == SubTaskKind.EDIT and not is_edit_ready(analysis):
        blocks.append(
            "Harness edit_ready=false; Planner cannot start with edit. "
            "Plan diagnose/design before edit."
        )

    non_completed_kinds = [
        node.kind for node in tree.nodes
        if node.status != SubTaskStatus.SUCCESS
    ]

    if analysis.complexity == "high":
        required = [SubTaskKind.DIAGNOSE, SubTaskKind.DESIGN, SubTaskKind.EDIT, SubTaskKind.VERIFY]
        for kind in required:
            if kind not in kinds:
                blocks.append(
                    f"Harness policy for high-complexity tasks requires {kind.value} step."
                )
        ordered = [kind for kind in non_completed_kinds if kind in required]
        if not _stage_order_valid(ordered, required):
            blocks.append(
                "Harness policy requires high-complexity task order: diagnose -> design -> edit -> verify."
            )
    elif analysis.complexity == "medium":
        required = [SubTaskKind.DIAGNOSE, SubTaskKind.EDIT, SubTaskKind.VERIFY]
        for kind in required:
            if kind not in kinds:
                blocks.append(
                    f"Harness policy for medium-complexity tasks requires {kind.value} step."
                )
        ordered = [kind for kind in non_completed_kinds if kind in required]
        if not _stage_order_valid(ordered, required):
            blocks.append(
                "Harness policy requires medium-complexity task order: diagnose -> edit -> verify."
            )

    if analysis.intent == "sql_view_rewrite":
        has_view_dep = any(
            dep.get("kind") == "database_view"
            for dep in analysis.resolved_dependencies
            if isinstance(dep, dict)
        )
        if not has_view_dep:
            has_resolving_step = any(
                node.kind in (SubTaskKind.DIAGNOSE, SubTaskKind.DESIGN)
                for node in tree.nodes
            )
            if not has_resolving_step:
                blocks.append(
                    "Harness policy for sql_view_rewrite requires a resolved database_view dependency."
                )
    return blocks


def _stage_order_valid(
    observed: list[SubTaskKind],
    required_order: list[SubTaskKind],
) -> bool:
    rank = {kind: idx for idx, kind in enumerate(required_order)}
    last = -1
    for kind in observed:
        current = rank[kind]
        if current < last:
            return False
        last = current
    return True


def _check_required_diagnose_first(tree: TaskTree) -> list[str]:
    if not tree.nodes or not _FUNCTION_REFACTOR_HINTS.search(tree.root_task):
        return []
    first = tree.nodes[0]
    if first.status == SubTaskStatus.SUCCESS:
        return []
    if first.kind == SubTaskKind.DIAGNOSE:
        return []
    return [
        "Function refactor/reuse/normalize/masking tasks must start with a diagnose "
        "handoff before edit. Plan st-1 diagnose, then st-2 edit."
    ]


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
        if node.status == SubTaskStatus.SUCCESS:
            continue
        if node.kind != SubTaskKind.EDIT:
            continue
        for dep_id in node.depends_on:
            dep = by_id.get(dep_id)
            if dep and dep.kind == SubTaskKind.DIAGNOSE:
                if dep.status == SubTaskStatus.SUCCESS:
                    continue
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


def _check_distinct_milestones(tree: TaskTree) -> list[str]:
    blocks: list[str] = []
    nodes = [node for node in tree.nodes if node.status != SubTaskStatus.SUCCESS]
    if len(nodes) <= 1:
        return blocks

    root_key = _compact_raw(tree.root_task)
    seen: dict[str, SubTaskNode] = {}
    for node in nodes:
        desc_key = _semantic_key(node.description)
        if not desc_key:
            continue

        raw_desc_key = _compact_raw(node.description)
        if root_key and node.kind != SubTaskKind.VERIFY:
            generic_prefixed_copy = bool(
                re.match(
                    r"^\s*(定位目标代码|定位目标|修改目标|执行任务|完成任务)[:：]",
                    node.description,
                    re.I,
                )
            ) and root_key in raw_desc_key
            root_copied = raw_desc_key == root_key or generic_prefixed_copy
            if root_copied:
                blocks.append(
                    f"Subtask [{node.id}] description copies root_task instead of a "
                    "stage-specific milestone. Rewrite it as a concrete diagnose/edit "
                    "output, not the whole user request."
                )

        prior = seen.get(desc_key)
        if prior is not None and node.kind != SubTaskKind.VERIFY:
            blocks.append(
                f"Subtask [{node.id}] duplicates [{prior.id}] description. Each plan "
                "step must have a distinct milestone output."
            )
        else:
            seen[desc_key] = node

    non_verify = [node for node in nodes if node.kind != SubTaskKind.VERIFY]
    for idx, left in enumerate(non_verify):
        left_key = _semantic_key(left.description)
        for right in non_verify[idx + 1 :]:
            right_key = _semantic_key(right.description)
            if _similarity(left_key, right_key) >= 0.9:
                blocks.append(
                    f"Subtasks [{left.id}] and [{right.id}] have near-duplicate "
                    "descriptions. Split diagnose handoff, edit target, and verify "
                    "milestones explicitly."
                )
    return blocks


def _check_replan_preserves_failed_objective(
    tree: TaskTree,
    ctx: ReplanGateContext,
) -> list[str]:
    if ctx.failed_kind != SubTaskKind.EDIT:
        return []
    has_pending_edit = any(
        node.kind == SubTaskKind.EDIT
        and node.status in {SubTaskStatus.PENDING, SubTaskStatus.RUNNING}
        for node in tree.nodes
    )
    if has_pending_edit:
        return []
    return [
        f"Re-plan for failed edit [{ctx.failed_subtask_id}] removed the edit objective. "
        "Replacement nodes may add diagnose first, but must include a follow-up edit "
        "subtask that completes the original user change."
    ]


def _check_handoff_dependency_structure(tree: TaskTree) -> list[str]:
    blocks: list[str] = []
    by_id = {node.id: node for node in tree.nodes}

    def ancestors(node: SubTaskNode) -> list[SubTaskNode]:
        ordered: list[SubTaskNode] = []
        seen: set[str] = set()

        def walk(dep_id: str) -> None:
            if dep_id in seen:
                return
            seen.add(dep_id)
            dep = by_id.get(dep_id)
            if dep is None:
                return
            for parent_id in dep.depends_on:
                walk(parent_id)
            ordered.append(dep)

        for dep_id in node.depends_on:
            walk(dep_id)
        return ordered

    for node in tree.nodes:
        if node.status == SubTaskStatus.SUCCESS:
            continue
        required = {item.strip() for item in node.requires_handoff if item.strip()}
        if "PATCH_INTENT_JSON" not in required:
            continue
        design_ancestors = [
            dep for dep in ancestors(node)
            if dep.kind == SubTaskKind.DESIGN
            and (
                "PATCH_INTENT_JSON" in dep.handoff_outputs
                or "PATCH_INTENT_JSON" in dep.acceptance_criteria
            )
        ]
        if not design_ancestors:
            blocks.append(
                f"Subtask [{node.id}] requires PATCH_INTENT_JSON but does not depend "
                "on a prior design step that outputs PATCH_INTENT_JSON."
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
