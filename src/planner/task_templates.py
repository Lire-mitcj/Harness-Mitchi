from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

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


def task_tree_from_template(user_request: str, template: TaskTemplate) -> TaskTree:
    if template.mode == TaskMode.INVESTIGATE:
        target = _infer_target_label(user_request)
        desired = _infer_desired_change(user_request)
        return TaskTree(
            root_task=user_request,
            nodes=[
                SubTaskNode(
                    id="st-1",
                    description=f"定位{target}、当前查询和可用视图"[:120],
                    kind=SubTaskKind.DIAGNOSE,
                    acceptance_criteria=(
                        "输出 HANDOFF_CONTRACT_JSON：file:line、symbol、SQL、视图、片段/决策"
                    ),
                    allowed_tools=["context_search"],
                    context_files=[],
                    needs_l1=False,
                ),
                SubTaskNode(
                    id="st-2",
                    description=f"将{target}改为{desired}"[:120],
                    kind=SubTaskKind.EDIT,
                    acceptance_criteria="仅修改 handoff 指定代码，查询使用目标视图且 old/new 不同",
                    allowed_tools=["context_search", "edit_file"],
                    context_files=[],
                    depends_on=["st-1"],
                    needs_l1=True,
                ),
                SubTaskNode(
                    id="st-3",
                    description=f"验证{target}查询行为"[:120],
                    kind=SubTaskKind.VERIFY,
                    acceptance_criteria=(
                        "Relevant verification command exits 0 or blocker is reported"
                    ),
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


def _infer_desired_change(user_request: str) -> str:
    text = user_request.strip()
    lower = text.lower()
    if "视图" in text or "view" in lower:
        return "使用目标视图查询"
    if "接口" in text:
        return "符合请求的接口行为"
    return "符合请求的实现"


def _short_phrase_before_change(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return "目标代码"
    start = max(0, idx - 10)
    phrase = text[start : idx + len(marker)]
    phrase = re.sub(r"^[，。！？、\s]*(把|将|当前|现在)?", "", phrase)
    return phrase or "目标代码"
