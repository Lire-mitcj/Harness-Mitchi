from __future__ import annotations

from src.agent.evidence_slots import (
    SatisfyRule,
    rule_matches,
    slot_satisfy_rule,
    slots_for_task,
)
from src.agent.manifest import manifest_from_slots, project_manifest


def _anchor(file: str, code: str, symbol: str = "", span=(1, 10)) -> dict:
    return {"file": file, "span": [span[0], span[1]], "code": code, "symbol": symbol}


# ---------------------------------------------------------------------------
# rule_matches: structural gates
# ---------------------------------------------------------------------------


def test_min_lines_gate_rejects_single_line() -> None:
    rule = SatisfyRule(min_lines=2)
    assert rule_matches(rule, "one line only") is False
    assert rule_matches(rule, "line one\nline two") is True


def test_require_def_rejects_import_and_assignment() -> None:
    rule = SatisfyRule(require_def=True)
    assert rule_matches(rule, "from fastapi.responses import JSONResponse") is False
    assert rule_matches(rule, "logger = logging.getLogger(__name__)") is False
    assert rule_matches(rule, "def handler():\n    return 1") is True
    assert rule_matches(rule, "class Foo:\n    pass") is True


def test_require_ddl_matches_create_table_only() -> None:
    rule = SatisfyRule(require_ddl=True)
    assert rule_matches(rule, "status = 'active'") is False
    assert rule_matches(rule, "CREATE TABLE orders (id INT);") is True
    assert rule_matches(rule, "create or replace view v as select 1") is True


def test_positive_group_requires_at_least_one_match() -> None:
    rule = SatisfyRule(code_patterns=(r"include_router",))
    assert rule_matches(rule, "app.mount('/static', files)") is False
    assert rule_matches(rule, "app.include_router(router)") is True


def test_decorator_pattern_matches_only_decorator_lines() -> None:
    rule = SatisfyRule(
        require_def=True, decorator_patterns=(r"\.exception_handler",)
    )
    # substring present in a comment/body but no decorator line -> no match
    assert (
        rule_matches(rule, "def f():\n    # exception_handler note\n    return 1")
        is False
    )
    handler = "@app.exception_handler(SQLAlchemyError)\ndef on_db_error():\n    ..."
    assert rule_matches(rule, handler) is True


def test_def_name_pattern_uses_definition_names_and_symbol() -> None:
    rule = SatisfyRule(def_name_patterns=(r"^test_",))
    assert rule_matches(rule, "def helper():\n    assert True") is False
    assert rule_matches(rule, "def test_thing():\n    assert True") is True
    # symbol fallback (grep may not include the def line body)
    assert rule_matches(rule, "assert x == 1", symbol="test_case") is True


# ---------------------------------------------------------------------------
# slot registry wiring
# ---------------------------------------------------------------------------


def test_every_activated_slot_exposes_a_rule() -> None:
    task = "统一异常日志 接口 路由 数据库 表结构 认证 权限 归属 测试 挂载"
    for slot_id in slots_for_task(task):
        assert slot_satisfy_rule(slot_id) is not None


def test_unknown_slot_has_no_rule() -> None:
    assert slot_satisfy_rule("does_not_exist") is None


# ---------------------------------------------------------------------------
# manifest projection: structural precision beats substring keywords
# ---------------------------------------------------------------------------


def test_exception_handler_slot_ignores_import_and_assignment_lines() -> None:
    manifest = manifest_from_slots(["exception_handler_context"])

    noise = [
        _anchor("main.py", "from fastapi.responses import JSONResponse"),
        _anchor("main.py", "logger = logging.getLogger(__name__)"),
    ]
    projected = project_manifest(manifest, noise, step=1, task_mode="diagnose")
    assert projected.required_items[0].status == "MISSING"

    handler = [
        _anchor(
            "main.py",
            "@app.exception_handler(SQLAlchemyError)\n"
            "async def handle_db_error(request, exc):\n"
            "    logger.exception('db error')\n"
            "    return JSONResponse(status_code=500, content={})",
            symbol="handle_db_error",
        )
    ]
    projected = project_manifest(manifest, handler, step=2, task_mode="diagnose")
    assert projected.required_items[0].status == "SATISFIED"


def test_endpoint_slot_requires_route_shape_not_any_def() -> None:
    manifest = manifest_from_slots(["endpoint_implementation"])

    plain = [_anchor("util.py", "def helper():\n    return 1", symbol="helper")]
    projected = project_manifest(manifest, plain, step=1, task_mode="edit")
    assert projected.required_items[0].status == "MISSING"

    route = [
        _anchor(
            "api.py",
            "@router.get('/orders')\ndef list_orders():\n    return []",
            symbol="list_orders",
        )
    ]
    projected = project_manifest(manifest, route, step=2, task_mode="edit")
    assert projected.required_items[0].status == "SATISFIED"


def test_schema_slot_still_requires_ddl() -> None:
    manifest = manifest_from_slots(["relevant_schema"])

    grep_only = [{"file": "db/init.sql", "span": [1, 1], "match_line": "status"}]
    projected = project_manifest(manifest, grep_only, step=1, task_mode="diagnose")
    assert projected.required_items[0].status == "MISSING"

    ddl = [_anchor("db/init.sql", "CREATE TABLE order_timeline (id INT);", span=(1, 3))]
    projected = project_manifest(manifest, ddl, step=2, task_mode="diagnose")
    assert projected.required_items[0].status == "SATISFIED"


def test_decorator_db_task_activates_integration_slot() -> None:
    slots = slots_for_task("给 list.py 数据库路由加装饰器，参考 main.py 的异常处理")
    assert "integration_or_mount_point" in slots
    assert "target_implementation" in slots
