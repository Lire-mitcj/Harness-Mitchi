from __future__ import annotations

import json

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
    assert "EXECUTOR_HANDOFF_JSON" in (messages[1].content or "")
    assert "mitkii.executor_handoff.v1" in (messages[1].content or "")
    assert '"allowed_tools"' in (messages[1].content or "")
    assert "st-1" in (messages[2].content or "")


def test_executor_handoff_json_contains_runtime_tools_only(tmp_path) -> None:
    from src.harness.subtask.prompt_builder import build_executor_messages
    from src.planner.kinds import SubTaskKind
    from src.planner.task_tree import SubTaskNode, TaskTree

    tree = TaskTree(
        root_task="task",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="edit",
                allowed_tools=["context_search", "edit_file", "shell_exec"],
            ),
        ],
    )
    messages = build_executor_messages(
        root_task="task",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        runtime_tools=frozenset({"edit_file"}),
    )
    raw_json = (messages[1].content or "").split("\n", 1)[1]
    handoff = json.loads(raw_json)

    assert handoff["allowed_tools"] == ["edit_file"]
    assert "shell_exec" in handoff["denied_tools"]
    assert "shell_exec" not in json.dumps(handoff["plan_state"], ensure_ascii=False)
    assert handoff["requirements"]["final_output"]["required_keys"] == [
        "status",
        "changed_files",
        "validation",
        "risks",
        "handoff",
    ]


def test_executor_handoff_json_carries_prior_evidence_and_negatives(tmp_path) -> None:
    from src.harness.subtask.prompt_builder import build_executor_messages
    from src.planner.kinds import SubTaskKind
    from src.planner.task_tree import SubTaskNode, TaskTree

    (tmp_path / "app.py").write_text("def query_orders():\n    pass\n", encoding="utf-8")
    tree = TaskTree(
        root_task="task",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.DIAGNOSE, description="find"),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="edit",
                depends_on=["st-1"],
            ),
        ],
    )
    messages = build_executor_messages(
        root_task="task",
        task_tree=tree,
        subtask=tree.nodes[1],
        project_root=tmp_path,
        runtime_tools=frozenset({"edit_file"}),
        prior_summaries={
            "st-1": (
                '{"status":"success","changed_files":[],'
                '"validation":{"ran":[],"result":"skipped","summary":""},'
                '"risks":[],"handoff":{"facts":["located route"],'
                '"evidence":[{"path":"app.py","line":1,'
                '"symbol":"query_orders","snippet":"def query_orders"}],'
                '"known_negatives":[{"query":"tests","reason":"no matches"}],'
                '"next_focus":["app.py"]}}'
            )
        },
    )
    handoff = json.loads((messages[1].content or "").split("\n", 1)[1])

    assert handoff["prior"]["evidence"]
    assert handoff["prior"]["evidence"][0]["path"] == "app.py"
    assert handoff["prior"]["known_negatives"]
    assert handoff["prior"]["facts"][0]["fact"] == "located route"
    assert handoff["prior"]["next_focus"][0]["focus"] == "app.py"


def test_executor_handoff_json_carries_artifact_store(tmp_path) -> None:
    from src.harness.subtask.prompt_builder import build_executor_messages
    from src.planner.kinds import SubTaskKind
    from src.planner.task_tree import SubTaskNode, TaskTree

    tree = TaskTree(
        root_task="task",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.DIAGNOSE, description="find"),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="edit",
                depends_on=["st-1"],
                requires_artifacts=["database_view"],
            ),
        ],
    )
    messages = build_executor_messages(
        root_task="task",
        task_tree=tree,
        subtask=tree.nodes[1],
        project_root=tmp_path,
        runtime_tools=frozenset({"context_search", "edit_file"}),
        prior_summaries={
            "st-1": json.dumps({
                "status": "success",
                "changed_files": [],
                "validation": {"ran": [], "result": "skipped", "summary": ""},
                "risks": [],
                "handoff": {
                    "artifacts": [
                        {
                            "kind": "database_view",
                            "canonical_id": "database_view:v_order_detail",
                            "name": "v_order_detail",
                            "confidence": 0.91,
                        }
                    ]
                },
            })
        },
    )
    handoff = json.loads((messages[1].content or "").split("\n", 1)[1])

    assert handoff["artifact_store"]["policy"]["may_not_block_edit"] is True
    assert handoff["artifact_store"]["artifacts"][0]["name"] == "v_order_detail"


def test_executor_prompt_carries_context_pack_json(tmp_path) -> None:
    from src.context.pack import ContextPack, ContextSnippet
    from src.harness.subtask.prompt_builder import build_executor_messages
    from src.planner.kinds import SubTaskKind
    from src.planner.task_tree import SubTaskNode, TaskTree

    pack = ContextPack(
        user_request="edit",
        candidate_files=({"file": "app.py", "score": 0.95, "reasons": ["symbol_match"]},),
        focused_snippets=(
            ContextSnippet(
                file_path="app.py",
                start_line=10,
                end_line=12,
                text="10: def target():\n11:     return 1",
                source="symbol",
            ),
        ),
        evidence=({"type": "symbol_match", "file": "app.py", "symbol": "target"},),
        tool_policy={"allowed_tools": ["context_search", "edit_file"], "denied_tools": ["delete_file"]},
    )
    tree = TaskTree(
        root_task="task",
        nodes=[SubTaskNode(id="st-1", kind=SubTaskKind.EDIT, description="edit")],
    )

    messages = build_executor_messages(
        root_task="task",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        runtime_tools=frozenset({"context_search", "edit_file"}),
        context_pack=pack,
    )
    handoff = json.loads((messages[1].content or "").split("\n", 1)[1])
    payload = json.loads((messages[2].content or "").split("\n", 1)[1])

    assert handoff["context_pack_summary"]["candidate_files"][0]["file"] == "app.py"
    assert payload["context_pack"]["focused_snippets"][0]["file"] == "app.py"
    assert payload["context_pack"]["tool_policy"]["denied_tools"] == ["delete_file"]


def test_llm_client_enables_parallel_tool_calls() -> None:
    from src.llm.client import LLMClient

    client = LLMClient(model="test-model")
    kwargs = client._build_kwargs(  # noqa: SLF001 - intentional unit coverage
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "grep_search"}}],
    )

    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True


def test_llm_client_allows_per_request_budget_override() -> None:
    from src.llm.client import LLMClient

    client = LLMClient(model="test-model", request_timeout=180)
    kwargs = client._build_kwargs(  # noqa: SLF001 - intentional unit coverage
        [{"role": "user", "content": "summarize"}],
        [],
        max_tokens=512,
        timeout=30,
        response_format={"type": "json_object"},
    )

    assert kwargs["max_tokens"] == 512
    assert kwargs["timeout"] == 30
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "tools" not in kwargs
