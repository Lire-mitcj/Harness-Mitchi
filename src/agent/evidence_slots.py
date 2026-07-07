"""Single source of truth for task-text classification.

Historically the same "task text -> keyword hit -> category" decision was
duplicated across three modules that drifted apart:

* ``run_state.requirements_for_task`` (which evidence slots a task activates),
* ``manifest._SLOT_TEMPLATES`` (how a slot is described / satisfied),
* ``grep_discovery._SLOT_GREP`` (how a slot is located via grep).

This module unifies all of that. Each evidence slot is declared once as an
``EvidenceSlotDef`` carrying every facet: activation triggers, human-readable
need, item type, satisfaction keywords, and grep discovery hints. Coarse
discovery *themes* (used by grep pattern selection, independent of evidence
slots) also live here so every classification keyword table is co-located.

Consumers read from this registry instead of maintaining private dicts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_BATCH_PATTERNS = 8

_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+(\w+)", re.MULTILINE)
_DDL_RE = re.compile(
    r"create\s+(?:or\s+replace\s+)?(?:temp\w*\s+)?"
    r"(?:table|view|trigger|procedure)",
    re.IGNORECASE,
)
_DECORATOR_RE = re.compile(r"^\s*@(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class SatisfyRule:
    """Structural predicate deciding whether a durable anchor satisfies a slot.

    Unlike flat substring keywords, this inspects the *shape* of the loaded code
    (real definitions, decorators, DDL blocks, line count) so that single-line
    assignments (``logger = logging.getLogger(__name__)``) or import lines
    (``from fastapi.responses import JSONResponse``) can no longer falsely
    ground a slot merely because they contain a matching substring.

    Evaluation is: all structural *gates* (``min_lines`` / ``require_def`` /
    ``require_ddl``) must hold, and if any positive pattern group is declared at
    least one pattern across all groups must match (logical OR).
    """

    # Minimum number of non-empty code lines (filters 1-line assignments).
    min_lines: int = 1
    # Require a real definition (def / async def / class) in the anchor.
    require_def: bool = False
    # Require a CREATE TABLE/VIEW/TRIGGER/PROCEDURE DDL block.
    require_ddl: bool = False
    # Any-of regex matched against the full anchor code (case-insensitive).
    code_patterns: tuple[str, ...] = ()
    # Any-of regex matched against decorator lines (``@...``).
    decorator_patterns: tuple[str, ...] = ()
    # Any-of regex matched against def/class names (and the anchor symbol).
    def_name_patterns: tuple[str, ...] = ()


def rule_matches(rule: SatisfyRule, code: str, symbol: str = "") -> bool:
    """Return True when ``code`` structurally satisfies ``rule``."""
    non_empty = [line for line in code.splitlines() if line.strip()]
    if len(non_empty) < rule.min_lines:
        return False

    def_names = _DEF_RE.findall(code)
    has_def = bool(def_names)
    if rule.require_def and not has_def:
        return False
    if rule.require_ddl and not _DDL_RE.search(code):
        return False

    has_positive_group = bool(
        rule.code_patterns or rule.decorator_patterns or rule.def_name_patterns
    )
    if not has_positive_group:
        return True

    for pattern in rule.code_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return True
    if rule.decorator_patterns:
        decorator_lines = _DECORATOR_RE.findall(code)
        for pattern in rule.decorator_patterns:
            if any(re.search(pattern, line, re.IGNORECASE) for line in decorator_lines):
                return True
    if rule.def_name_patterns:
        names = [*def_names]
        if symbol:
            names.append(symbol.split(".")[-1])
        for pattern in rule.def_name_patterns:
            if any(re.search(pattern, name, re.IGNORECASE) for name in names):
                return True
    return False


@dataclass(frozen=True)
class EvidenceSlotDef:
    """One evidence requirement, with all facets its consumers need."""

    id: str
    need: str
    item_type: str = "symbol"
    # Task-text substrings that activate this slot (case-folded match).
    triggers: tuple[str, ...] = ()
    # Slot is required regardless of task text (e.g. target_implementation).
    always: bool = False
    # Additional slots implied when this one activates.
    also_requires: tuple[str, ...] = ()
    # Structural predicate a durable anchor must satisfy to ground the slot.
    satisfy: SatisfyRule = field(default_factory=SatisfyRule)
    # grep discovery hints; empty include means "no dedicated grep template".
    grep_patterns: tuple[str, ...] = ()
    grep_include: str = ""
    grep_path: str = "."


EVIDENCE_SLOTS: tuple[EvidenceSlotDef, ...] = (
    EvidenceSlotDef(
        id="target_implementation",
        need="目标实现代码已加载",
        item_type="symbol",
        always=True,
        satisfy=SatisfyRule(require_def=True),
        grep_patterns=("build_router", "include_router", "APIRouter", "create_app"),
        grep_include="*.py",
        grep_path=".",
    ),
    EvidenceSlotDef(
        id="endpoint_implementation",
        need="接口/路由实现已加载",
        item_type="symbol",
        triggers=("endpoint", "route", "接口", "路由"),
        satisfy=SatisfyRule(
            require_def=True,
            decorator_patterns=(
                r"(?:app|router)\.(?:get|post|put|delete|patch|api_route)",
            ),
            code_patterns=(r"apirouter", r"build_router", r"add_api_route", r"\brouter\b"),
        ),
        grep_patterns=(
            "@router\\.",
            "@app\\.(get|post|put|delete)",
            "APIRouter",
            "build_router",
        ),
        grep_include="*.py",
        grep_path=".",
    ),
    EvidenceSlotDef(
        id="integration_or_mount_point",
        need="接入/挂载点已加载",
        item_type="mount",
        triggers=(
            "integrate",
            "integration",
            "mount",
            "caller",
            "接入",
            "挂载",
            "调用",
            "decorator",
            "装饰器",
            "error handler",
            "sqlalchemy",
            "数据库",
            "db error",
            "db异常",
            "db异常处理",
        ),
        satisfy=SatisfyRule(
            code_patterns=(
                r"\.include_router\(",
                r"add_api_route",
                r"\.mount\(",
                r"\.register",
                r"app\.include",
            ),
        ),
        grep_patterns=("include_router", "add_api_route", "mount", "build_router"),
        grep_include="*.py",
        grep_path=".",
    ),
    EvidenceSlotDef(
        id="authentication_context",
        need="认证上下文已加载",
        item_type="symbol",
        triggers=("auth", "login", "token", "current_user", "登录态", "认证", "鉴权"),
        satisfy=SatisfyRule(
            code_patterns=(
                r"current_user",
                r"get_current_user",
                r"\btoken\b",
                r"oauth",
                r"authenticate",
            ),
        ),
        grep_patterns=("get_current_user", "Bearer", "oauth", "authenticate"),
        grep_include="*.py",
        grep_path=".",
    ),
    EvidenceSlotDef(
        id="authorization_policy",
        need="授权/权限策略已加载",
        item_type="symbol",
        triggers=("role", "permission", "authorize", "角色", "权限", "授权"),
        satisfy=SatisfyRule(
            code_patterns=(
                r"\brole\b",
                r"permission",
                r"\brequire",
                r"authorize",
                r"\bscope\b",
                r"depends",
            ),
        ),
        grep_patterns=("role", "permission", "authorize", "Depends"),
        grep_include="*.py",
        grep_path=".",
    ),
    EvidenceSlotDef(
        id="ownership_relation",
        need="归属/数据范围关系已加载",
        item_type="symbol",
        triggers=("owner", "ownership", "tenant", "归属", "数据范围"),
        also_requires=("relevant_schema",),
        satisfy=SatisfyRule(
            code_patterns=(
                r"\bowner",
                r"user_id",
                r"tenant",
                r"created_by",
                r"belongs",
            ),
        ),
    ),
    EvidenceSlotDef(
        id="relevant_schema",
        need="相关表结构/DDL 已加载",
        item_type="schema",
        triggers=(
            "sql",
            "schema",
            "table",
            "database",
            "数据库",
            "表结构",
            "新建表",
            "创建表",
            "建表",
            "数据表",
        ),
        satisfy=SatisfyRule(require_ddl=True),
        grep_patterns=("CREATE TABLE", "CREATE VIEW", "ticket_order", "passenger_info"),
        grep_include="*.sql",
        grep_path="db",
    ),
    EvidenceSlotDef(
        id="exception_handler_context",
        need="统一异常/日志处理接口已加载",
        item_type="symbol",
        triggers=(
            "exception",
            "handler",
            "异常",
            "错误处理",
            "logging",
            "日志",
            "统一",
        ),
        satisfy=SatisfyRule(
            require_def=True,
            decorator_patterns=(
                r"\.exception_handler",
                r"add_exception_handler",
            ),
            def_name_patterns=(r"handle", r"error"),
            code_patterns=(
                r"logger\.exception",
                r"add_exception_handler",
                r"exception_handler",
            ),
        ),
        grep_patterns=(
            "@app\\.exception_handler",
            "add_exception_handler",
            "async def handle_",
            "def handle_",
            "def _handle_",
            "logger\\.exception",
        ),
        grep_include="main.py",
        grep_path="main.py",
    ),
    EvidenceSlotDef(
        id="test_or_validation_path",
        need="测试/验证路径已加载",
        item_type="symbol",
        triggers=("test", "verify", "验证", "测试"),
        satisfy=SatisfyRule(
            def_name_patterns=(r"^test_",),
            code_patterns=(r"\bassert\b", r"pytest", r"validate"),
        ),
    ),
)


_SLOTS_BY_ID: dict[str, EvidenceSlotDef] = {slot.id: slot for slot in EVIDENCE_SLOTS}


# Coarse discovery themes drive grep pattern selection independent of evidence
# slots (e.g. ``db_integration`` has no dedicated slot). Kept here so all task
# classification keyword tables live in one place.
THEME_TRIGGERS: dict[str, tuple[str, ...]] = {
    "exception_handler": (
        "exception",
        "handler",
        "异常",
        "错误处理",
        "logging",
        "日志",
        "log ",
        "统一",
    ),
    "db_integration": ("database", "sqlalchemy", "数据库", "db error", "db异常"),
    "endpoint": ("endpoint", "route", "router", "接口", "路由", "build_router"),
    "schema": (
        "schema",
        "table",
        "database",
        "sql",
        "建表",
        "数据库",
        "表结构",
        "新建表",
        "数据表",
    ),
    "auth": ("auth", "login", "token", "登录态", "认证", "鉴权"),
}


def slot_def(slot_id: str) -> EvidenceSlotDef | None:
    return _SLOTS_BY_ID.get(slot_id)


def slot_satisfy_rule(slot_id: str) -> SatisfyRule | None:
    """Structural predicate for a registry slot, or None for unknown slots."""
    slot = _SLOTS_BY_ID.get(slot_id)
    return slot.satisfy if slot is not None else None


def slots_for_task(task_text: str) -> frozenset[str]:
    """Evidence slots a task activates (registry-driven replacement for the
    former inline keyword ladder in ``requirements_for_task``)."""
    lowered = task_text.casefold()
    activated: set[str] = set()
    for slot in EVIDENCE_SLOTS:
        if slot.always or any(trigger in lowered for trigger in slot.triggers):
            activated.add(slot.id)
            activated.update(slot.also_requires)
    return frozenset(activated)


def themes_for_task(task_text: str) -> frozenset[str]:
    lowered = task_text.casefold()
    return frozenset(
        theme
        for theme, words in THEME_TRIGGERS.items()
        if any(word in lowered for word in words)
    )


def slot_grep_hint(slot_id: str) -> tuple[str, str, tuple[str, ...]] | None:
    """Return (include, path, patterns) for a slot, or None when it has no
    dedicated grep template."""
    slot = _SLOTS_BY_ID.get(slot_id)
    if slot is None or not slot.grep_include:
        return None
    return slot.grep_include, slot.grep_path, slot.grep_patterns
