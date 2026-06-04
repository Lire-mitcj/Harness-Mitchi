from __future__ import annotations

from src.agent.types import AgentState, ToolCall, assistant_message, tool_message
from src.executor.subtask_executor import (
    _diagnose_handoff_from_seed_if_ready,
    _diagnose_seed_tool_calls,
    _diagnose_summary_from_digest,
    _looks_like_llm_timeout,
    _should_seed_diagnose_search,
)
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


def test_diagnose_seed_tool_calls_use_context_search() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="定位目标代码：把当前登机牌查询接口改成用视图查询",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )

    calls = _diagnose_seed_tool_calls(
        root_task="把当前登机牌查询接口改成用视图查询",
        subtask=node,
        available_tools=frozenset({"context_search"}),
    )

    assert len(calls) == 1
    assert calls[0].name == "context_search"
    assert "登机牌" in calls[0].arguments["query"]
    assert "views" in calls[0].arguments["need"]
    assert "API endpoint" in calls[0].arguments["need"]


def test_should_seed_diagnose_search_requires_structured_handoff() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="定位目标代码",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )

    assert _should_seed_diagnose_search(
        subtask=node,
        active_runtime_tools=frozenset({"context_search"}),
        prior_summaries=None,
        subtask_attempt=1,
    )


def test_should_seed_diagnose_search_with_context_files() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="检查登机牌查询接口的实现",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
        context_files=["main.py"],
    )

    assert _should_seed_diagnose_search(
        subtask=node,
        active_runtime_tools=frozenset({"context_search"}),
        prior_summaries=None,
        subtask_attempt=1,
    )


def test_diagnose_seed_tool_calls_scope_paths() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="检查登机牌查询接口的实现",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )

    calls = _diagnose_seed_tool_calls(
        root_task="把当前登机牌查询接口改成用视图查询",
        subtask=node,
        available_tools=frozenset({"context_search"}),
        search_paths=("main.py",),
    )

    assert calls[0].arguments["paths"] == ["main.py"]


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


def test_llm_timeout_detector_matches_client_timeout_text() -> None:
    assert _looks_like_llm_timeout(
        "LLM request timed out after 180s. Try a smaller context or retry."
    )
    assert not _looks_like_llm_timeout("normal diagnose summary")


def test_diagnose_seed_handoff_skips_llm_when_evidence_is_complete() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="搜索项目中使用的视图",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )
    state = AgentState()
    state.messages = [
        assistant_message(
            "",
            [
                ToolCall(
                    id="tc1",
                    name="context_search",
                    arguments={"query": "project views", "need": "file:line symbol snippet"},
                )
            ],
        ),
        tool_message(
            "tc1",
            '<context_snippets>\n'
            '<snippet path="src/db/views.sql" lines="12-18">\n'
            "12: CREATE VIEW active_orders AS\n"
            "13: SELECT * FROM orders\n"
            "</snippet>\n"
            "</context_snippets>",
        ),
    ]
    memory = ExploreSessionMemory.create()

    handoff = _diagnose_handoff_from_seed_if_ready(
        subtask=node,
        session_memory=memory,
        state=state,
        error_trace=[],
    )

    assert handoff is not None
    final_message, exit_result = handoff
    assert exit_result.passed
    assert "src/db/views.sql:12" in final_message
    assert "CREATE VIEW" in final_message
    assert "Session exploration summary" not in final_message


def test_diagnose_seed_handoff_requires_gate_passing_evidence() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="搜索项目中使用的视图",
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )
    state = AgentState()
    state.messages = [tool_message("tc1", "Found likely view usage but no location.")]

    handoff = _diagnose_handoff_from_seed_if_ready(
        subtask=node,
        session_memory=ExploreSessionMemory.create(),
        state=state,
        error_trace=[],
    )

    assert handoff is None


def test_diagnose_seed_handoff_requires_actionable_findings_even_without_strict_criteria() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="搜索项目中使用的视图",
    )
    state = AgentState()
    state.messages = [
        tool_message(
            "tc1",
            "Session exploration summary (paths, line ranges, map/grep hits, code seen so far):\n"
            "Context/map searches already run:\n"
            "  - 视图 view\n",
        )
    ]

    handoff = _diagnose_handoff_from_seed_if_ready(
        subtask=node,
        session_memory=ExploreSessionMemory.create(),
        state=state,
        error_trace=[],
    )

    assert handoff is None


def test_diagnose_summary_filters_view_definition_noise() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="搜索项目中的视图定义",
        acceptance_criteria="输出文件:行号、符号和代码片段/决策",
    )
    digest = (
        "Session exploration summary (paths, line ranges, map/grep hits, code seen so far):\n"
        "Context/map searches already run:\n"
        "  - 视图 view 查找项目中的视图\n"
        "Grep hits (sample):\n"
        "  - app.py:1429:    key_prefix = f\"seat-{flight_id}\" if interactive else f\"admin-seat-view-{flight_id}\"\n"
        "  - app.py:2057:                st.warning(\"报表视图未返回可识别的满座率字段，已展示明细数据。\")\n"
        "  - db/init/init.sql:352:-- 视图：航班满座率监控。\n"
        "  - db/init/init.sql:353:CREATE VIEW flight_load AS SELECT * FROM flights;\n"
    )

    summary = _diagnose_summary_from_digest(node, digest, [])

    assert "db/init/init.sql:352" in summary
    assert "db/init/init.sql:353" in summary
    assert "app.py:1429" not in summary
    assert "app.py:2057" not in summary
    assert "Session exploration summary" not in summary


def test_diagnose_summary_without_findings_is_user_readable_failure() -> None:
    node = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="搜索项目中使用的视图",
    )
    digest = (
        "Session exploration summary (paths, line ranges, map/grep hits, code seen so far):\n"
        "Context/map searches already run:\n"
        "  - 视图 view\n"
    )

    summary = _diagnose_summary_from_digest(node, digest, [])

    assert "未定位到可交接的具体路径和行号" in summary
    assert "视图 view" in summary
    assert "Session exploration summary" not in summary
