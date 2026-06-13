from __future__ import annotations

from src.planner.intent_hints import cap_structure_text, is_exploration_request
from src.planner.planner_node import _extract_json, _parse_task_tree
from src.planner.task_templates import TaskMode, select_task_template
from src.planner.task_tree import SubTaskKind


def test_is_exploration_request_chinese_views() -> None:
    assert is_exploration_request("这个项目里面有哪些视图")
    assert not is_exploration_request("修复注册接口的数据库事务错误")


def test_cap_structure_text() -> None:
    long_text = "x" * 5000
    capped = cap_structure_text(long_text, max_chars=100)
    assert len(capped) <= 100
    assert "truncated" in capped


def test_fallback_exploration_uses_diagnose() -> None:
    tree = _parse_task_tree("", fallback_task="这个项目里面有哪些视图")
    assert len(tree.nodes) == 1
    assert tree.nodes[0].kind == SubTaskKind.DIAGNOSE


def test_fallback_change_request_uses_diagnose_edit_verify() -> None:
    tree = _parse_task_tree("", fallback_task="把当前登机牌查询接口改成用视图查询")
    assert [node.kind for node in tree.nodes] == [
        SubTaskKind.DIAGNOSE,
        SubTaskKind.EDIT,
        SubTaskKind.VERIFY,
    ]
    assert tree.nodes[1].depends_on == ["st-1"]
    assert "file:line" in tree.nodes[0].acceptance_criteria
    assert "登机牌查询接口" in tree.nodes[0].description
    assert "HANDOFF_CONTRACT_JSON" in tree.nodes[0].acceptance_criteria
    assert tree.nodes[1].description != tree.root_task
    assert "符合请求的实现" in tree.nodes[1].description


def test_task_template_selection_scores_change_request() -> None:
    template = select_task_template("把当前登机牌查询接口改成用视图查询")
    assert template.mode == TaskMode.INVESTIGATE
    assert [step.id for step in template.steps] == ["context", "work", "verify"]


def test_repair_unquoted_json_keys() -> None:
    raw = (
        '{root_task:"List views",nodes:[{id:"st-1",kind:"diagnose",'
        'description:"x",acceptance_criteria:"ok",allowed_tools:["list_dir"],'
        "context_files:[],depends_on:[]}]}"
    )
    payload = _extract_json(raw)
    assert payload.get("root_task") == "List views"
    assert len(payload.get("nodes") or []) == 1


def test_repair_trailing_comma() -> None:
    raw = '{"root_task":"x","nodes":[{"id":"st-1","kind":"diagnose",},]}'
    payload = _extract_json(raw)
    assert payload.get("root_task") == "x"


def test_task_template_selection_refinement() -> None:
    # 1. Pure queries should map to ANSWER_TEMPLATE
    t1 = select_task_template("查询机票有哪些视图")
    assert t1.mode == TaskMode.ANSWER

    t2 = select_task_template("show all user tables")
    assert t2.mode == TaskMode.ANSWER

    t3 = select_task_template("有哪些航班查询接口？")
    assert t3.mode == TaskMode.ANSWER

    # 2. Queries with edit/change keywords should map to INVESTIGATE_TEMPLATE
    t4 = select_task_template("修改机票查询视图的逻辑")
    assert t4.mode == TaskMode.INVESTIGATE

    t5 = select_task_template("add a new column to passenger table")
    assert t5.mode == TaskMode.INVESTIGATE

