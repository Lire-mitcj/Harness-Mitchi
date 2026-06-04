from __future__ import annotations

from src.executor.subtask_executor import (
    _diagnose_seed_tool_calls,
    _should_seed_diagnose_search,
)
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


def test_diagnose_seed_tool_calls_batch_patterns_by_module() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="定位目标代码：把当前登机牌查询接口改成用视图查询",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )

    calls = _diagnose_seed_tool_calls(
        root_task="把当前登机牌查询接口改成用视图查询",
        subtask=node,
        available_tools=frozenset({"map_search", "grep_search"}),
    )

    grep_calls = [call for call in calls if call.name == "grep_search"]
    assert [call.arguments["include"] for call in grep_calls] == ["*.py", "*.sql"]
    assert "登机牌|boarding|boarding_pass" in grep_calls[0].arguments["pattern"]
    assert r"@app\.(get|post|put|delete|patch)" in grep_calls[0].arguments["pattern"]
    assert r"CREATE\s+VIEW" in grep_calls[0].arguments["pattern"]


def test_should_seed_diagnose_search_requires_structured_handoff() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="定位目标代码",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )

    assert _should_seed_diagnose_search(
        subtask=node,
        active_runtime_tools=frozenset({"grep_search", "map_search"}),
        prior_summaries=None,
        subtask_attempt=1,
    )


def test_should_not_seed_general_diagnose_without_handoff_contract() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="解释这段日志为什么慢",
        acceptance_criteria="Explain likely root cause",
    )

    assert not _should_seed_diagnose_search(
        subtask=node,
        active_runtime_tools=frozenset({"grep_search", "map_search"}),
        prior_summaries=None,
        subtask_attempt=1,
    )
