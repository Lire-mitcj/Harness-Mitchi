from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


class TaskMode(StrEnum):
    ANSWER = "answer"
    INVESTIGATE = "investigate"


@dataclass(frozen=True)
class TaskStepTemplate:
    id: str
    kind: str
    goal: str
    outputs: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskTemplate:
    mode: TaskMode
    steps: tuple[TaskStepTemplate, ...]
    validation: tuple[str, ...] = ()


ANSWER_TEMPLATE = TaskTemplate(
    mode=TaskMode.ANSWER,
    steps=(
        TaskStepTemplate(
            id="context",
            kind="context",
            goal="Find the minimal relevant code/files for the user request",
            outputs=("relevant_files", "key_symbols", "constraints"),
        ),
    ),
)

INVESTIGATE_TEMPLATE = TaskTemplate(
    mode=TaskMode.INVESTIGATE,
    steps=(
        TaskStepTemplate(
            id="context",
            kind="context",
            goal="Find the minimal relevant code/files for the user request",
            outputs=("relevant_files", "key_symbols", "constraints"),
        ),
        TaskStepTemplate(
            id="work",
            kind="implement",
            goal="Make the smallest safe change or answer based on discovered context",
            depends_on=("context",),
            outputs=("patch_or_answer",),
        ),
        TaskStepTemplate(
            id="verify",
            kind="verify",
            goal="Run targeted validation or explain why validation is unavailable",
            depends_on=("work",),
            outputs=("validation_result", "risks"),
        ),
    ),
    validation=("targeted_tests_or_static_check",),
)

_CHANGE_INTENT = (
    "fix",
    "implement",
    "refactor",
    "add",
    "delete",
    "patch",
    "migrate",
    "debug",
    "update",
    "修复",
    "实现",
    "重构",
    "添加",
    "删除",
    "优化",
    "修改",
    "改成",
    "改为",
    "写入",
    "创建",
)
_CHANGE_OBJECTS = (
    "接口",
    "查询",
    "视图",
    "sql",
    "handler",
    "endpoint",
    "route",
    "query",
    "view",
    "schema",
)
_ANSWER_INTENT = (
    "哪些",
    "什么",
    "有没有",
    "列出",
    "介绍",
    "说明",
    "查看",
    "统计",
    "架构",
    "结构",
    "what",
    "which",
    "list",
    "show",
    "describe",
    "overview",
)


def select_task_template(user_request: str) -> TaskTemplate:
    """Choose a fallback template from scored intent facets, not a single keyword."""
    text = user_request.strip().lower()
    if not text:
        return ANSWER_TEMPLATE

    change_score = _score_terms(text, _CHANGE_INTENT) * 2 + _score_terms(text, _CHANGE_OBJECTS)
    answer_score = _score_terms(text, _ANSWER_INTENT)
    if text.endswith("?") or text.endswith("？"):
        answer_score += 1

    if change_score > answer_score:
        return INVESTIGATE_TEMPLATE
    return ANSWER_TEMPLATE


def task_tree_from_template(
    user_request: str,
    template: TaskTemplate,
    *,
    task_analysis: Any | None = None,
) -> TaskTree:
    complexity = None
    if task_analysis is not None:
        if hasattr(task_analysis, "complexity"):
            complexity = str(getattr(task_analysis, "complexity") or "")
        elif isinstance(task_analysis, dict):
            complexity = str(task_analysis.get("complexity") or "")

    strategy = _analysis_strategy(task_analysis)

    is_multi_step = (
        complexity in ("high", "medium")
        or strategy in ("function_refactor", "sql_view_rewrite")
        or template.mode == TaskMode.INVESTIGATE
    )

    if is_multi_step:
        target = _infer_target_label(user_request)
        desired = _desired_change_for_strategy(strategy)

        if complexity == "high" or strategy == "function_refactor" or strategy == "sql_view_rewrite":
            design_description = f"设计{target}的补丁意图并处理依赖"[:120]
            if strategy == "function_refactor":
                diag_desc = f"定位{target}调用点、旧逻辑和可复用依赖"[:120]
                diag_crit = "输出 HANDOFF_CONTRACT_JSON：file:line、symbol、调用点、片段/决策"
                design_description = f"设计{target}的函数重构补丁意图"[:120]
                edit_crit = "仅修改设计指定目标，复用目标函数并移除旧重复逻辑"
            elif strategy == "sql_view_rewrite":
                diag_desc = f"定位{target}、当前 SQL 和 Harness 解析的视图依赖"[:120]
                diag_crit = "输出 HANDOFF_CONTRACT_JSON：file:line、symbol、SQL、resolved dependency、片段/决策"
                design_description = f"设计{target}的视图替换补丁意图"[:120]
                edit_crit = "仅修改设计指定目标，使用 Harness 批准的视图依赖且 old/new 不同"
            else:
                diag_desc = f"定位{target}的目标代码和当前实现"[:120]
                diag_crit = "输出 HANDOFF_CONTRACT_JSON：file:line、symbol、当前片段、片段/决策"
                edit_crit = "仅修改设计指定目标，按 Harness/Design 批准的策略完成修改"

            editable_targets = []
            if task_analysis is not None:
                if hasattr(task_analysis, "editable_targets"):
                    editable_targets = list(getattr(task_analysis, "editable_targets") or [])
                elif isinstance(task_analysis, dict):
                    editable_targets = list(task_analysis.get("editable_targets") or [])

            score = calculate_edit_complexity_score(task_analysis)

            nodes = [
                SubTaskNode(
                    id="st-1",
                    description=diag_desc,
                    kind=SubTaskKind.DIAGNOSE,
                    acceptance_criteria=diag_crit,
                    allowed_tools=["context_search"],
                    context_files=[],
                    needs_l1=False,
                    handoff_outputs=[],
                ),
                SubTaskNode(
                    id="st-2",
                    description=design_description,
                    kind=SubTaskKind.DESIGN,
                    acceptance_criteria=(
                        "输出完整 PATCH_INTENT_JSON：edit_ready=true、edit_strategy、"
                        "edit_targets、dependencies、acceptance_criteria、target_view"
                    ),
                    allowed_tools=["context_search"],
                    context_files=[],
                    depends_on=["st-1"],
                    needs_l1=False,
                    handoff_outputs=["PATCH_INTENT_JSON"],
                ),
            ]

            edit_nodes = []
            if len(editable_targets) > 1 and score >= 2.0:
                for idx, t_item in enumerate(editable_targets):
                    symbol_name = t_item.get("symbol") or f"target_{idx+1}"
                    file_name = t_item.get("file") or ""
                    sub_id = f"st-3_{idx+1}"
                    dep = [edit_nodes[-1].id] if edit_nodes else ["st-2"]
                    edit_nodes.append(
                        SubTaskNode(
                            id=sub_id,
                            description=f"分块修改第{idx+1}个目标 {symbol_name} ({file_name})"[:120],
                            kind=SubTaskKind.EDIT,
                            acceptance_criteria=edit_crit,
                            allowed_tools=["context_search", "edit_file"],
                            context_files=[file_name] if file_name else [],
                            depends_on=dep,
                            needs_l1=True,
                            requires_handoff=["PATCH_INTENT_JSON"],
                        )
                    )
            else:
                edit_nodes.append(
                    SubTaskNode(
                        id="st-3",
                        description=f"按 PATCH_INTENT_JSON 分块修改单个 edit_target：{target}改为{desired}"[:120],
                        kind=SubTaskKind.EDIT,
                        acceptance_criteria=edit_crit,
                        allowed_tools=["context_search", "edit_file"],
                        context_files=[],
                        depends_on=["st-2"],
                        needs_l1=True,
                        requires_handoff=["PATCH_INTENT_JSON"],
                    )
                )

            nodes.extend(edit_nodes)

            nodes.append(
                SubTaskNode(
                    id="st-4",
                    description=f"验证{target}重构行为"[:120],
                    kind=SubTaskKind.VERIFY,
                    acceptance_criteria="Relevant verification command exits 0 or blocker is reported",
                    allowed_tools=["shell_exec"],
                    context_files=[],
                    depends_on=[edit_nodes[-1].id],
                    needs_l1=False,
                )
            )

            return TaskTree(
                root_task=user_request,
                nodes=nodes,
            )
        else:
            diag_desc = f"定位{target}的目标代码和当前实现"[:120]
            diag_crit = "输出 HANDOFF_CONTRACT_JSON：file:line、symbol、当前片段、片段/决策"
            edit_crit = "仅修改 handoff 指定代码，按 Harness 批准的策略完成修改"
            return TaskTree(
                root_task=user_request,
                nodes=[
                    SubTaskNode(
                        id="st-1",
                        description=diag_desc,
                        kind=SubTaskKind.DIAGNOSE,
                        acceptance_criteria=diag_crit,
                        allowed_tools=["context_search"],
                        context_files=[],
                        needs_l1=False,
                        handoff_outputs=[],
                    ),
                    SubTaskNode(
                        id="st-2",
                        description=f"将{target}改为{desired}"[:120],
                        kind=SubTaskKind.EDIT,
                        acceptance_criteria=edit_crit,
                        allowed_tools=["context_search", "edit_file"],
                        context_files=[],
                        depends_on=["st-1"],
                        needs_l1=True,
                    ),
                    SubTaskNode(
                        id="st-3",
                        description=f"验证{target}查询行为"[:120],
                        kind=SubTaskKind.VERIFY,
                        acceptance_criteria="Relevant verification command exits 0 or blocker is reported",
                        allowed_tools=["shell_exec"],
                        context_files=[],
                        depends_on=["st-2"],
                        needs_l1=False,
                    ),
                ],
            )

    return TaskTree(
        root_task=user_request,
        nodes=[
            SubTaskNode(
                id="st-1",
                description=f"查找{_infer_target_label(user_request)}并总结证据"[:120],
                kind=SubTaskKind.DIAGNOSE,
                acceptance_criteria="输出用户可读结论，并列出 file:line、symbol、片段/决策证据",
                allowed_tools=["context_search"],
                context_files=[],
                needs_l1=False,
            )
        ],
    )


def _analysis_strategy(task_analysis: Any | None) -> str:
    if hasattr(task_analysis, "edit_strategy"):
        return str(getattr(task_analysis, "edit_strategy") or "")
    if isinstance(task_analysis, dict):
        return str(task_analysis.get("edit_strategy") or task_analysis.get("intent") or "")
    return ""


def _desired_change_for_strategy(strategy: str) -> str:
    if strategy == "function_refactor":
        return "复用 Harness 解析的目标函数/ helper"
    if strategy == "sql_view_rewrite":
        return "使用 Harness 解析的目标视图查询"
    return "符合请求的实现"


def _score_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if _contains_term(text, term))


def _contains_term(text: str, term: str) -> bool:
    if re.search(r"[a-z]", term):
        return bool(re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))
    return term in text


def _infer_target_label(user_request: str) -> str:
    text = user_request.strip()
    lower = text.lower()
    if "登机牌" in text:
        if "接口" in text or "查询" in text:
            return "登机牌查询接口"
        return "登机牌相关代码"
    if "视图" in text or "view" in lower:
        if any(word in text for word in ("哪些", "查找", "搜索", "列出")):
            return "项目视图使用情况"
        return "视图相关查询"
    if "接口" in text:
        return _short_phrase_before_change(text, "接口")
    if "sql" in lower:
        return "目标 SQL"
    if "查询" in text:
        return "目标查询"
    return "目标代码"


def _short_phrase_before_change(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return "目标代码"
    start = max(0, idx - 10)
    phrase = text[start : idx + len(marker)]
    phrase = re.sub(r"^[，。！？、\s]*(把|将|当前|现在)?", "", phrase)
    return phrase or "目标代码"


def calculate_edit_complexity_score(task_analysis: Any | None) -> float:
    if task_analysis is None:
        return 0.0
    editable_targets = []
    if hasattr(task_analysis, "editable_targets"):
        editable_targets = getattr(task_analysis, "editable_targets") or []
    elif isinstance(task_analysis, dict):
        editable_targets = task_analysis.get("editable_targets") or []
    
    score = 0.0
    # Each target increases complexity
    score += len(editable_targets) * 1.0
    
    for target in editable_targets:
        if isinstance(target, dict):
            code = str(target.get("current_code") or "")
            code_upper = code.upper()
            if "SELECT" in code_upper and "FROM" in code_upper:
                # SQL complexity
                score += code_upper.count("JOIN") * 0.5
                select_count = code_upper.count("SELECT")
                if select_count > 1:
                    score += (select_count - 1) * 0.5
                if "WITH" in code_upper:
                    score += 0.5
                if "UNION" in code_upper:
                    score += 0.5
    return score

