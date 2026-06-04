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
        return TaskTree(
            root_task=user_request,
            nodes=[
                SubTaskNode(
                    id="st-1",
                    description=f"定位目标代码：{user_request}"[:120],
                    kind=SubTaskKind.DIAGNOSE,
                    acceptance_criteria=(
                        "Output file:line, symbol, and snippet/decision for the edit target"
                    ),
                    allowed_tools=["context_search"],
                    context_files=[],
                    needs_l1=False,
                ),
                SubTaskNode(
                    id="st-2",
                    description=user_request[:120],
                    kind=SubTaskKind.EDIT,
                    acceptance_criteria="Target behavior is changed as requested",
                    allowed_tools=["context_search", "edit_file"],
                    context_files=[],
                    depends_on=["st-1"],
                    needs_l1=True,
                ),
                SubTaskNode(
                    id="st-3",
                    description="验证相关行为",
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
                description=user_request[:120],
                kind=SubTaskKind.DIAGNOSE,
                acceptance_criteria="Search project and list relevant file paths with evidence",
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
