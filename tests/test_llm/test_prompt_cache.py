from __future__ import annotations

from src.llm.prompt_cache import apply_prompt_cache, mark_cache_breakpoint


def test_apply_prompt_cache_strips_markers_when_disabled() -> None:
    messages = [
        mark_cache_breakpoint({"role": "system", "content": "x" * 5000}),
    ]
    out = apply_prompt_cache(messages, model="claude-sonnet-4-20250514", enabled=False)
    assert "_mitkii_cache_breakpoint" not in out[0]
    assert "cache_control" not in out[0]


def test_apply_prompt_cache_skips_small_blocks() -> None:
    messages = [
        mark_cache_breakpoint({"role": "system", "content": "tiny"}),
    ]
    out = apply_prompt_cache(
        messages,
        model="claude-sonnet-4-20250514",
        enabled=True,
        min_tokens=1024,
    )
    assert "cache_control" not in out[0]


def test_planner_messages_have_cached_project_context() -> None:
    from src.planner.planner_node import PlannerNode, _PROJECT_CONTEXT_LABEL

    node = PlannerNode(client=object())  # type: ignore[arg-type]
    msgs = node.plan_messages("fix bug", "<repo_map>sym</repo_map>")
    assert msgs[0]["role"] == "system"
    assert msgs[0].get("_mitkii_cache_breakpoint") is True
    assert msgs[1]["role"] == "system"
    assert _PROJECT_CONTEXT_LABEL in msgs[1]["content"]
    assert msgs[2]["role"] == "user"
    assert _PROJECT_CONTEXT_LABEL not in msgs[2]["content"]


def test_executor_messages_layered_cache_breakpoints() -> None:
    from src.orchestrator.isolation import build_executor_messages
    from src.planner.kinds import SubTaskKind
    from src.planner.task_tree import SubTaskNode, TaskTree

    tree = TaskTree(
        root_task="task",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.EDIT, description="edit"),
        ],
    )
    messages = build_executor_messages(
        root_task="task",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=__import__("pathlib").Path("."),
    )
    assert len(messages) == 3
    assert all(m.role == "system" for m in messages)
    assert all(m.cache_breakpoint for m in messages)
    assert "MitKII Executor" in (messages[0].content or "")
    assert "TaskTree outline" in (messages[1].content or "")
    assert "st-1" in (messages[2].content or "")


def test_llm_client_enables_parallel_tool_calls() -> None:
    from src.llm.client import LLMClient

    client = LLMClient(model="test-model")
    kwargs = client._build_kwargs(  # noqa: SLF001 - intentional unit coverage
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "grep_search"}}],
    )

    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True
