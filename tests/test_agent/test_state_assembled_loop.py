from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.context_assembly import ContextAssembly, build_turn_context_block
from src.agent.events import EventType
from src.agent.manifest import EvidenceItem, StepManifest, Sufficiency
from src.agent.state_assembled_loop import (
    ASSEMBLED_TOOL_NAMES,
    AssembledState,
    ContextAnchors,
    ConversationShaper,
    SystemLayerShaper,
    StateAssembledLoop,
    _latest_symbol_slice_projection,
    _build_deduped_loaded_anchors_block,
    _collapse_retrieval_turn,
    _get_git_state,
    _dedupe_code_anchors,
    _anchor_key,
    _anchor_memory_kind,
    _enrich_anchor_contract,
    _fact_lock_replay_result,
    _microcompact_retrieval_payload,
    _retrieval_snapshot_from_output,
    _search_cache_for_context,
    _search_cache_view,
    _tool_history_receipt,
    _run_state_projection,
    _missing_evidence_actions,
    _prune_contained_anchors,
    _response_evidence_summary,
)
from src.agent.run_state import (
    ArtifactRefs,
    Evidence,
    RunEvent,
    RunPhase,
    reduce_run_state,
    start_run,
)
from src.agent.types import LLMResponse, Message, ToolCall, ToolResult, RiskLevel
from src.harness.gates.phase_metrics import PhaseMetrics
from src.hooks.post_tool_context import apply_post_tool_context_hook
from src.tools.assembled.codebase_retrieve import CodebaseRetrieveTool
from src.tools.assembled.decision_edit import DecisionEditTool


def test_trace_manifest_prints_projected_state_and_allowed_tools(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = StepManifest(
        step_id="task.default",
        step_kind="edit",
        updated_at_step=3,
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        required_items=(
            EvidenceItem(
                id="target_implementation",
                need="目标实现代码已加载",
                status="STALE",
                file="src/example.py",
                span=(10, 20),
                symbol="target",
                stale_reason="file modified by decision_edit",
            ),
        ),
    )

    StateAssembledLoop._trace_manifest(
        manifest,
        allowed_tools=frozenset({"decision_edit", "view_symbol_code"}),
    )

    output = capsys.readouterr().out
    assert output.startswith("[debug][manifest][projection-json]\n")
    snapshot = json.loads(output.split("\n", 1)[1])
    assert snapshot["step"] == 3
    assert snapshot["sufficiency"] == "SUFFICIENT_FOR_EDIT"
    assert snapshot["items"][0]["status"] == "STALE"
    assert snapshot["items"][0]["span"] == [10, 20]
    assert snapshot["allowed_tools"] == ["decision_edit", "view_symbol_code"]


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "assembled_system_prompt.md").write_text("Mock Lead Coordinator Prompt.", encoding="utf-8")
    
    mitkii_dir = tmp_path / ".mitkii"
    mitkii_dir.mkdir()
    (mitkii_dir / "rules.md").write_text("PRJ-01: Follow project rules.", encoding="utf-8")
    
    (tmp_path / "target.py").write_text("def run():\n    pass\n", encoding="utf-8")
    return tmp_path


def _prime_acting_run(loop: StateAssembledLoop) -> None:
    run_state = start_run("修改目标函数", edit_mode=True)
    run_state, _ = reduce_run_state(
        run_state,
        RunEvent(
            "evidence_stored",
            evidence=(Evidence(
                slot="target_implementation",
                artifact_id="target.py:1-2",
                file="target.py",
                symbol="run",
                evidence_type="full_symbol",
            ),),
            artifact_refs=ArtifactRefs(code=("target.py:1-2",)),
        ),
    )
    loop.state = replace(loop.state, run_state=run_state)


def test_context_assembly(temp_project: Path) -> None:
    assembly = ContextAssembly(temp_project)
    prompt = assembly.load_system_prompt()
    assert "central commander and orchestrator" in prompt

    messages = assembly.assemble(
        user_query="implement feature X",
        active_files=["target.py"],
        checklist=["[ ] step 1", "[x] step 2"],
        git_diff="dummy diff",
        validation_error="dummy error",
        messages_history=[Message(role="assistant", content="Thinking...")],
        last_tool_result="decision_edit(target_file=\"target.py\") -> failed",
        last_error={"error_type": "SchemaValidationError", "file": "target.py", "line": 42},
    )
    
    system_msg = next(m for m in messages if m["role"] == "system")
    user_msg = next(m for m in messages if m["role"] == "user")
    
    assert "central commander and orchestrator" in system_msg["content"]
    assert "Follow project rules." in user_msg["content"]
    assert "step 1" in user_msg["content"]
    assert "target.py" in user_msg["content"]
    assert "def run():" not in user_msg["content"]
    assert "dummy diff" in user_msg["content"]
    assert "dummy error" in user_msg["content"]
    assert "RUNTIME STATE (this turn)" in user_msg["content"]


@pytest.mark.asyncio
async def test_git_state_contains_status_only_not_patch_source(tmp_path: Path) -> None:
    process = MagicMock(returncode=0)
    process.communicate = AsyncMock(return_value=(b" M app.py\n?? new.py\n", b""))
    with patch("src.agent.state_assembled_loop.asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn:
        state = await _get_git_state(tmp_path)

    command = spawn.await_args.args
    assert command[:4] == ("git", "status", "--short", "--untracked-files=normal")
    assert state == "M app.py\n?? new.py"
    assert "diff --git" not in state


def test_context_assembly_with_search_cache(temp_project: Path) -> None:
    assembly = ContextAssembly(temp_project)
    search_cache = {
        "search_output": "<code_window file=\"target.py\">def hidden_artifact(): pass</code_window>",
        "context_projection": (
            "<projected_retrieval file=\"target.py\">def mock_find(): pass</projected_retrieval>"
        ),
    }
    tool_context = assembly.build_context_block(
        ["target.py"],
        search_cache=search_cache,
        modified_files=[],
    )
    assert "RETRIEVAL CODE ANCHOR (current turn only)" in tool_context
    assert "mock_find" in tool_context
    assert "hidden_artifact" not in tool_context


def test_read_summaries_belong_to_user_context_not_tool_context(temp_project: Path) -> None:
    assembly = ContextAssembly(temp_project)
    cache = {
        "summary_anchors": {
            "target.py:1-2": (
                "[CONTEXT COLLAPSE - target.py:1-2] [已读/DO NOT RETRIEVE]\n"
                "读取目的：确认 run 的行为。"
            )
        }
    }

    user_context = assembly.get_user_context(["target.py"], cache)
    tool_context = assembly.build_context_block(["target.py"], cache, modified_files=[])

    assert "已读代码摘要" in user_context
    assert "确认 run 的行为" in user_context
    assert "CONTEXT COLLAPSE" not in tool_context


def test_summary_body_has_exactly_one_context_injection_source(temp_project: Path) -> None:
    summary = "[CONTEXT COLLAPSE - target.py:1-2] [READ_LOCKED]\nunique body"
    state = AssembledState(
        messages_history=(
            Message(role="system", content="### TURN SUMMARY\n- 已读取：target.py:1-2"),
            Message(role="tool", content="[ANCHOR MOVED TO MEMORY: target.py:1-2]"),
        ),
        context_anchors=ContextAnchors(summaries={"target.py:1-2": summary}),
    )
    assembly = ContextAssembly(temp_project)
    system = SystemLayerShaper().shape(state, assembly.load_system_prompt())
    user_context = assembly.get_user_context(
        ["target.py"], _search_cache_view(state)
    )

    assert (system + user_context).count("unique body") == 1


def test_legacy_history_summary_is_migrated_to_anchor_reference(temp_project: Path) -> None:
    loop = StateAssembledLoop(
        llm=MagicMock(), tools=MagicMock(),
        harness=MagicMock(project_root=temp_project), context=MagicMock(),
        permissions=MagicMock(), settings=MagicMock(max_turns=1),
    )
    body = (
        "[CONTEXT COLLAPSE - target.py:1-2] [已读]\n"
        "函数名称：run\n顶层物理进口：`import os`; `import json`\n"
        "读取目的：legacy full body"
    )
    loop.state = AssembledState(
        messages_history=(Message(role="tool", content=body),),
        context_anchors=ContextAnchors(summaries={"target.py:1-2": body}),
    )

    loop._normalize_anchor_memory()

    assert loop.state.messages_history[0].content == (
        "[ANCHOR MOVED TO MEMORY: target.py:1-2]"
    )
    assert "legacy full body" in loop.state.context_anchors.summaries["target.py:1-2"]
    assert "顶层物理进口" not in loop.state.context_anchors.summaries["target.py:1-2"]
    assert "文件契约引用：`target.py@" in loop.state.context_anchors.summaries["target.py:1-2"]


def test_retrieval_tool_history_stores_receipt_not_verbatim_code() -> None:
    call = ToolCall(id="read", name="codebase_retrieve", arguments={"query": "auth"})
    result = ToolResult(
        success=True,
        output='[{"code":"def auth(): SECRET_SOURCE"}]',
        metadata={
            "raw_evidence_store": [
                {"file": "main.py", "span": [1, 2], "symbol": "auth", "code": "def auth(): SECRET_SOURCE"}
            ]
        },
    )

    receipt = _tool_history_receipt(call, result)

    assert "[RETRIEVAL OK — NEW EVIDENCE STORED]" in receipt


def test_fact_lock_replay_becomes_no_new_evidence() -> None:
    call = ToolCall(
        id="read-again",
        name="view_symbol_code",
        arguments={"target_file": "main.py", "symbol": "app"},
    )
    result = _fact_lock_replay_result(
        call,
        json.dumps({"file": "main.py", "span": [41, 41]}),
        {},
        ("endpoint_implementation",),
    )

    assert result.success is True
    assert result.metadata["is_mock_success"] is True
    assert result.output == json.dumps({"file": "main.py", "span": [41, 41]})


def test_conversation_compaction_is_role_aware_and_excludes_tool_bodies() -> None:
    messages = []
    for index in range(5):
        call = ToolCall(id=str(index), name="codebase_retrieve", arguments={"query": f"q{index}"})
        messages.extend([
            Message(role="assistant", content="检查鉴权并继续计划", tool_calls=[call]),
            Message(role="tool", content=f"[CODE ANCHOR STORED]\n- file: `f{index}.py`\nSECRET{index}"),
        ])
    state = AssembledState(messages_history=tuple(messages))

    compacted = ConversationShaper().shape(state)
    turn_summary = next(msg.content for msg in compacted.messages_history if "### TURN SUMMARY" in msg.content)

    assert "- 决策：" in turn_summary
    assert "codebase_retrieve" in turn_summary
    assert "检查鉴权并继续计划" not in turn_summary
    assert "- 已读取：" in turn_summary
    assert "SECRET" not in turn_summary


def test_conversation_compaction_drops_process_only_intent() -> None:
    messages = []
    for index in range(5):
        call = ToolCall(id=str(index), name="grep_search", arguments={"pattern": "auth"})
        messages.extend(
            [
                Message(
                    role="assistant",
                    content="Let me examine the current state more thoroughly.",
                    tool_calls=[call],
                ),
                Message(role="tool", content="[FILE FACT STORED]"),
            ]
        )
    compacted = ConversationShaper().shape(AssembledState(messages_history=tuple(messages)))
    summary = next(
        msg.content for msg in compacted.messages_history if "### TURN SUMMARY" in msg.content
    )
    assert "Let me examine" not in summary
    assert "继续未完成" not in summary
    assert "STEP EVIDENCE" in summary
    assert "状态快照（折叠时）" in summary


def test_run_phase_derives_actual_llm_tool_set() -> None:
    retrieving = start_run("检查函数", edit_mode=False)
    assert retrieving.allowed_tools == frozenset(
        {"codebase_retrieve", "grep_search", "view_symbol_code"}
    )
    responding, _ = reduce_run_state(
        retrieving,
        RunEvent(
            "evidence_stored",
            evidence=(Evidence(
                "target_implementation", "a1", "main.py", "run", "full_symbol"
            ),),
            artifact_refs=ArtifactRefs(code=("a1",)),
        ),
    )
    assert responding.phase == RunPhase.RESPONDING
    assert responding.allowed_tools == frozenset()


def test_missing_requirement_projection_gives_concrete_tool_action() -> None:
    state = start_run(
        "这个 endpoint 应该如何接入登录态？",
        edit_mode=False,
    )
    projection = _run_state_projection(state)
    assert 'pattern="build_router|include_router"' in projection
    assert 'include="*.py"' in projection
    assert "Locate only the missing" not in projection


def test_every_requirement_has_a_concrete_tool_action() -> None:
    names = [
        "target_implementation",
        "endpoint_implementation",
        "integration_or_mount_point",
        "authentication_context",
        "authorization_policy",
        "ownership_relation",
        "relevant_schema",
        "test_or_validation_path",
    ]
    actions = _missing_evidence_actions(names)
    assert len(actions) == len(names)
    assert all("→" in action for action in actions)
    assert all(
        "grep_search(" in action or "view_symbol_code(" in action
        for action in actions
    )
    assert all("pattern=" in action or "target_file=" in action for action in actions)


def test_duplicate_anchor_replays_existing_facts(temp_project: Path) -> None:
    anchor = {
        "file": "main.py",
        "span": [10, 16],
        "symbol": "auth_me",
        "code": (
            "def auth_me(token):\n"
            "    payload = decode_access_token(token)\n"
            "    user_id = payload['sub']\n"
            "    row = conn.execute('SELECT * FROM user_account')\n"
            "    if not row:\n"
            "        raise ValueError('missing')\n"
            "    return row\n"
        ),
    }
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(project_root=temp_project),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    loop.state = AssembledState(context_anchors=ContextAnchors(code=(anchor,)))
    result = loop._dedupe_result_anchors(
        ToolResult(
            success=True,
            output=json.dumps([anchor]),
            metadata={"raw_evidence_store": [anchor]},
        )
    )
    assert "EXISTING FACTS REPLAYED" in result.output
    assert "decode_access_token" in result.output
    assert "user_account" in result.output
    assert "EXACT SYMBOL SLICE COMPLETE" in result.output
    assert "SOURCE COVERAGE:" in result.output
    assert "FULL SOURCE LOADED" not in result.output
    assert "sub" in result.output
    receipt = _tool_history_receipt(
        ToolCall(id="dup", name="view_symbol_code", arguments={}),
        result,
    )
    assert "[RETRIEVAL DUPLICATE — NO NEW EVIDENCE]" in receipt
    assert "EXISTING FACTS REPLAYED" in receipt


def test_ready_summary_prunes_contained_code_and_dedupes_imports() -> None:
    outer = {
        "file": "list.py",
        "span": [9, 176],
        "symbol": "build_router",
        "code": "def build_router(engine):\n    def archive_passenger(): return True\n    return router",
    }
    inner = {
        "file": "list.py",
        "span": [48, 81],
        "symbol": "archive_passenger",
        "code": "def archive_passenger():\n    return True",
    }
    assert _prune_contained_anchors([inner, outer]) == [outer]
    state = AssembledState(
        context_anchors=ContextAnchors(
            code=(inner, outer),
            file_contracts={
                "./list.py": {"imports": ["from fastapi import APIRouter"]},
                "list.py": {"imports": ["from fastapi import APIRouter", "from sqlalchemy import text"]},
            },
        )
    )
    summary = _response_evidence_summary(state)
    assert '"code":' not in summary
    assert summary.count("from fastapi import APIRouter") == 1
    assert summary.count("FILE IMPORTS (ONCE PER FILE)") == 1


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_codebase_retrieve_tool(temp_project: Path) -> None:
    mock_settings = MagicMock()
    mock_settings.cursor_retrieval_max_files = 5
    mock_settings.cursor_retrieval_max_symbols = 5
    mock_settings.cursor_retrieval_candidate_symbols = 10
    mock_settings.cursor_retrieval_max_queries = 5
    mock_settings.cursor_retrieval_timeout = 5.0
    mock_settings.cursor_reranker_enabled = False
    mock_settings.cursor_max_context_files = 5
    mock_settings.cursor_context_chars_per_file = 1000
    mock_settings.cursor_dependency_affinity_threshold = 0.7
    mock_settings.cursor_query_bridge_timeout = 5.0
    mock_settings.cursor_graph_bridge_depth = 2
    mock_settings.cursor_graph_bridge_max_symbols = 5
    mock_settings.cursor_graph_bridge_max_files = 5
    mock_settings.cursor_graph_bridge_max_seeds = 5
    mock_settings.cursor_inter_enabled = False
    mock_settings.cursor_semantic_tags_enabled = False

    mock_harness = MagicMock()
    mock_harness.project_root = temp_project
    mock_harness.cursor_retrieval_guardrail.validate_bridge_json = MagicMock()
    
    mock_bridge = MagicMock()
    mock_bridge.intent = "explain"
    mock_bridge.domain = "codebase"
    mock_bridge.concepts = []
    mock_harness.cursor_retrieval_guardrail.validate_bridge_json.return_value.bridge = mock_bridge
    
    mock_llm = AsyncMock()
    mock_tools = MagicMock()
    mock_tools.get_schemas.return_value = [{"type": "function", "function": {"name": "decision_edit"}}]

    with patch("src.tools.assembled.codebase_retrieve.CursorRetriever"), \
         patch("src.tools.assembled.codebase_retrieve.CursorQueryBridge") as mock_qb_cls, \
         patch("src.tools.assembled.codebase_retrieve.CursorRepoMapLookup"), \
         patch("src.tools.assembled.codebase_retrieve.CursorGraphQueryBridge") as mock_gb_cls, \
         patch("src.tools.assembled.codebase_retrieve.CursorFusionEngine") as mock_fe_cls, \
         patch("src.tools.assembled.codebase_retrieve.CursorContextPackBuilder") as mock_cpb_cls, \
         patch("src.tools.assembled.codebase_retrieve.CursorAstStructureLayer"), \
         patch("src.tools.assembled.codebase_retrieve.CursorSemanticTagger"):
        
        mock_qb = mock_qb_cls.return_value
        mock_qb.generate_raw = AsyncMock(return_value="{}")
        mock_qb.fallback = MagicMock()
        
        mock_gb = mock_gb_cls.return_value
        mock_graph_res = MagicMock()
        mock_graph_res.expanded_symbols = ("target.py",)
        mock_graph_res.expanded_symbol_ids = ("target.py:run:1",)
        mock_graph_res.expanded_files = ("target.py",)
        mock_graph_res.paths = ()
        mock_graph_res.graph_nodes = ()
        mock_graph_res.graph_edges = ()
        mock_gb.expand_candidates = AsyncMock(return_value=mock_graph_res)
        
        mock_fe = mock_fe_cls.return_value
        mock_fusion_res = MagicMock()
        mock_fusion_res.final_context = ["target.py"]
        
        mock_symbol = MagicMock()
        mock_symbol.name = "run"
        mock_symbol.file = "target.py"
        mock_symbol.start_line = 1
        mock_symbol.end_line = 3
        mock_symbol.score = 0.95
        mock_symbol.kind = "function"
        mock_symbol.calls = ["append", "auth_me"]
        mock_symbol.tables_referenced = ["users"]
        
        mock_fusion_res.retrieval = MagicMock()
        related_symbol = MagicMock()
        related_symbol.name = "auth_me"
        related_symbol.file = "auth.py"
        related_symbol.start_line = 10
        related_symbol.end_line = 20
        related_symbol.score = 0.8
        related_symbol.kind = "function"
        related_symbol.calls = []
        related_symbol.tables_referenced = []
        mock_fusion_res.retrieval.symbols = [mock_symbol, related_symbol]
        mock_fusion_res.retrieval.files = ["target.py"]
        
        mock_fe.decide_async = AsyncMock(return_value=mock_fusion_res)
        
        mock_cpb = mock_cpb_cls.return_value
        mock_context_pack = MagicMock()
        mock_window = MagicMock()
        mock_window.file = "target.py"
        mock_window.start_line = 1
        mock_window.end_line = 3
        mock_window.content = "def run():\n    pass"
        mock_window.semantic_tags = ()
        mock_context_pack.windows = (mock_window,)
        mock_context_pack.candidate_files = ("target.py",)
        mock_cpb.build_context = MagicMock(return_value=mock_context_pack)

        tool = CodebaseRetrieveTool(
            project_root=temp_project,
            settings=mock_settings,
            repo_map=MagicMock(),
            decision_llm=mock_llm,
            inter_llm=mock_llm,
            harness=mock_harness,
            tools=mock_tools,
        )
        
        result = await tool.execute(query="find target")
        assert result.success is True
        assert 'target.py' in result.output
        assert "target.py" in result.metadata["retrieved_files"]
        import json
        payload = json.loads(result.output)
        assert set(payload[0]) == {"file", "span", "symbol"}
        assert payload[0]["file"] == "target.py"
        assert payload[0]["symbol"] == "run"
        assert payload[0]["span"] == [1, 3]
        search_payload = json.loads(result.metadata["search_output"])
        assert search_payload == payload


def test_retrieval_symbol_slice_is_verbatim_and_never_truncated(tmp_path: Path) -> None:
    source = "def long_function():\n" + "\n".join(
        f"    value_{index} = {index}" for index in range(150)
    )
    (tmp_path / "long.py").write_text(source, encoding="utf-8")
    tool = CodebaseRetrieveTool.__new__(CodebaseRetrieveTool)
    tool.project_root = tmp_path
    symbol = MagicMock(file="long.py", start_line=1, end_line=151)

    code, start, end, truncated = tool._symbol_code_slice(symbol, MagicMock(windows=()))

    assert code == source
    assert (start, end, truncated) == (1, 151, False)


def test_anchor_contract_extracts_signature_and_top_level_imports(tmp_path: Path) -> None:
    source = (
        "from typing import Optional\n"
        "from fastapi import APIRouter\n\n"
        "def build_router(engine: Engine) -> APIRouter:\n"
        "    return APIRouter()\n"
    )
    (tmp_path / "list.py").write_text(source, encoding="utf-8")
    anchor = {
        "file": "list.py",
        "span": [4, 5],
        "code": "def build_router(engine: Engine) -> APIRouter:\n    return APIRouter()",
    }

    enriched = _enrich_anchor_contract(tmp_path, anchor)

    assert enriched["signature"] == "def build_router(engine: Engine) -> APIRouter:"
    assert enriched["top_level_imports"] == [
        "from typing import Optional",
        "from fastapi import APIRouter",
    ]


def test_code_anchor_overlap_gate_drops_more_than_ninety_percent_duplicate() -> None:
    shorter = {"file": "main.py", "span": [10, 100], "code": "short"}
    larger = {"file": "main.py", "span": [5, 105], "code": "longer and richer"}
    distinct = {"file": "main.py", "span": [200, 220], "code": "distinct"}

    deduped = _dedupe_code_anchors([shorter, larger, distinct])

    # Anonymous spans have no symbol identity and are therefore preserved unless
    # their exact span/hash identity repeats.
    assert deduped == [shorter, larger, distinct]

    outer = {
        "file": "list.py", "span": [1, 100], "symbol": "build_router",
        "code": "def build_router(): pass",
    }
    inner = {
        "file": "list.py", "span": [5, 95], "symbol": "archive_passenger",
        "code": "def archive_passenger(): pass",
    }
    assert _dedupe_code_anchors([outer, inner]) == [outer, inner]
    assert _anchor_key({**outer, "hash": "abc"}) == ("list.py", "build_router", "abc")

    same_span_other_symbol = {
        **outer,
        "symbol": "login",
        "code": "def login(): pass",
    }
    assert _dedupe_code_anchors([outer, same_span_other_symbol]) == [
        outer,
        same_span_other_symbol,
    ]


def test_grep_symbol_locator_is_not_a_complete_code_anchor() -> None:
    locator = {
        "file": "list.py",
        "symbol": "build_router",
        "span": [16, 16],
        "match_line": "def build_router(engine):",
    }

    assert _anchor_memory_kind(locator) == "fact"


def test_anchor_ingestion_splits_symbols_file_facts_and_schema(temp_project: Path) -> None:
    (temp_project / "main.py").write_text(
        "import os\n\nAPI_URL = 'x'\n\ndef run(value: int) -> str:\n    return str(value)\n",
        encoding="utf-8",
    )
    (temp_project / "schema.sql").write_text(
        "CREATE TABLE users (id INT);\n", encoding="utf-8"
    )
    loop = StateAssembledLoop(
        llm=MagicMock(), tools=MagicMock(),
        harness=MagicMock(project_root=temp_project), context=MagicMock(),
        permissions=MagicMock(), settings=MagicMock(max_turns=2),
    )
    loop.state = AssembledState(run_state=replace(start_run("", edit_mode=False), step=1))
    call = ToolCall(id="r", name="codebase_retrieve", arguments={"query": "run schema"})

    loop._ingest_code_artifacts(call, [
        {"file": "main.py", "span": [3, 3], "code": "API_URL = 'x'", "related_functions": []},
        {"file": "main.py", "span": [5, 6], "code": "def run(value: int) -> str:\n    return str(value)", "related_functions": []},
        {"file": "schema.sql", "span": [1, 1], "code": "CREATE TABLE users (id INT);", "related_functions": []},
    ])

    assert [item["span"] for item in loop.state.context_anchors.code] == [[5, 6]]
    assert "API_URL" in loop.state.context_anchors.file_facts["main.py"][0]
    assert "schema.sql:1-1" in loop.state.context_anchors.schema_contracts
    contract = loop.state.context_anchors.file_contracts["main.py"]
    assert contract["imports"] == ["import os"]

    user_context = loop.context_assembly.get_user_context(
        ["main.py"], _search_cache_view(loop.state)
    )
    assert user_context.count("[FILE CONTRACT - main.py@") == 1


@pytest.mark.asyncio
async def test_decision_edit_tool(temp_project: Path) -> None:
    mock_settings = MagicMock()
    mock_settings.cursor_validator_model = "deepseek"
    mock_settings.cursor_validator_command = ["pytest"]
    mock_settings.cursor_validator_timeout = 10.0
    mock_settings.cursor_observation_max_chars = 1000
    mock_settings.cursor_validator_semantic_timeout = 5.0
    mock_settings.prompt_cache_ttl = "5m"

    mock_harness = MagicMock()
    mock_harness.project_root = temp_project
    mock_harness.before_llm_call = AsyncMock(side_effect=lambda x: x)
    mock_harness.after_llm_call = AsyncMock()

    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = '{"action":"edit","answer":"","clarification":"","target_file":"target.py","patch":"<<<<<<< SEARCH\\ndef run():\\n    pass\\n=======\\ndef run():\\n    return 42\\n>>>>>>> REPLACE","suggested_completion":100}'
    mock_resp.usage = MagicMock()
    mock_llm.chat = AsyncMock(return_value=mock_resp)

    with patch("src.tools.assembled.decision_edit.CursorDecisionLLM") as mock_dec_cls, \
         patch("src.tools.assembled.decision_edit.CursorPatchApplier"), \
         patch("src.tools.assembled.decision_edit.CursorExecutor") as mock_exec_cls, \
         patch("src.tools.assembled.decision_edit.CursorValidator"):

        mock_dec = mock_dec_cls.return_value
        from src.agent.contracts import Decision
        parsed_decision = Decision(
            action="edit",
            target_file="target.py",
            patch="<<<<<<< SEARCH\ndef run():\n    pass\n=======\ndef run():\n    return 42\n>>>>>>> REPLACE",
            suggested_completion=100
        )
        mock_dec.parse = MagicMock(return_value=parsed_decision)
        
        mock_exec = mock_exec_cls.return_value
        from src.agent.contracts import ExecutionResult, ValidationResult
        exec_res = ExecutionResult(success=True, file="target.py")
        val_res = ValidationResult(success=True)
        mock_exec.execute_transaction = AsyncMock(return_value=(exec_res, val_res, MagicMock()))

        tool = DecisionEditTool(
            project_root=temp_project,
            settings=mock_settings,
            decision_llm=mock_llm,
            harness=mock_harness,
        )
        tool.set_active_files(["target.py"])

        result = await tool.execute(target_file="target.py", intent="change run to return 42")
        assert result.success is True
        assert "应用成功" in result.output


@pytest.mark.asyncio
async def test_decision_edit_tool_with_frozen_context_window(temp_project: Path) -> None:
    mock_settings = MagicMock()
    mock_settings.cursor_validator_model = "deepseek"
    mock_settings.cursor_validator_command = ["pytest"]
    mock_settings.cursor_validator_timeout = 10.0
    mock_settings.cursor_observation_max_chars = 1000
    mock_settings.cursor_validator_semantic_timeout = 5.0
    mock_settings.prompt_cache_ttl = "5m"

    mock_harness = MagicMock()
    mock_harness.project_root = temp_project
    mock_harness.before_llm_call = AsyncMock(side_effect=lambda x: x)
    mock_harness.after_llm_call = AsyncMock()

    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = '{"action":"edit","target_file":"target.py","patch":"..."}'
    mock_resp.usage = MagicMock()
    mock_llm.chat = AsyncMock(return_value=mock_resp)

    # Setup dummy files on disk
    target_file = temp_project / "target.py"
    target_file.write_text("def run():\n    pass\n", encoding="utf-8")
    ref_file = temp_project / "ref.py"
    ref_file.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    with patch("src.tools.assembled.decision_edit.CursorDecisionLLM") as mock_dec_cls, \
         patch("src.tools.assembled.decision_edit.CursorPatchApplier"), \
         patch("src.tools.assembled.decision_edit.CursorExecutor") as mock_exec_cls, \
         patch("src.tools.assembled.decision_edit.CursorValidator"):

        mock_dec = mock_dec_cls.return_value
        from src.agent.contracts import Decision
        parsed_decision = Decision(
            action="edit",
            target_file="target.py",
            patch="<<<<<<< SEARCH\ndef run():\n    pass\n=======\ndef run():\n    return 42\n>>>>>>> REPLACE",
            suggested_completion=100
        )
        mock_dec.parse = MagicMock(return_value=parsed_decision)
        mock_dec.build_messages = MagicMock(return_value=[])
        
        mock_exec = mock_exec_cls.return_value
        from src.agent.contracts import ExecutionResult, ValidationResult
        exec_res = ExecutionResult(success=True, file="target.py")
        val_res = ValidationResult(success=True)
        mock_exec.execute_transaction = AsyncMock(return_value=(exec_res, val_res, MagicMock()))

        tool = DecisionEditTool(
            project_root=temp_project,
            settings=mock_settings,
            decision_llm=mock_llm,
            harness=mock_harness,
        )

        # Call with frozen context_window and task_id
        result = await tool.execute(
            target_file="target.py",
            intent="change logic",
            task_id="T-999",
            context_window=[
                {"file": "ref.py", "span": [2, 4], "reason": "test span"}
            ]
        )
        assert result.success is True
        assert "T-999" in result.output

        # Verify that only the specified lines of ref.py were loaded in ContextPack
        call_args = mock_dec.build_messages.call_args[1]
        context_pack = call_args["context_pack"]
        windows = context_pack.windows
        
        ref_windows = [w for w in windows if w.file == "ref.py"]
        assert len(ref_windows) == 1
        assert ref_windows[0].start_line == 2
        assert ref_windows[0].end_line == 4
        assert ref_windows[0].content == "line2\nline3\nline4"



def test_decision_edit_tool_multi_file_context(temp_project: Path) -> None:
    mock_settings = MagicMock()
    mock_llm = AsyncMock()
    mock_harness = MagicMock()
    mock_harness.project_root = temp_project

    tool = DecisionEditTool(
        project_root=temp_project,
        settings=mock_settings,
        decision_llm=mock_llm,
        harness=mock_harness,
    )
    
    # Create another file to test
    (temp_project / "other.py").write_text("def other(): pass\n", encoding="utf-8")
    (temp_project / "target.py").write_text("def target(): pass\n", encoding="utf-8")
    
    tool.set_active_files(["other.py", "target.py"])
    
    # CASE 1: other.py is active but NOT in raw_evidence_store -> it must be pruned
    context_pack = tool._build_context_pack("target.py")
    files_in_pack = [w.file for w in context_pack.windows]
    assert "target.py" in files_in_pack
    assert "other.py" not in files_in_pack
    
    # Verify role/mode attributes for target file
    target_window = [w for w in context_pack.windows if w.file == "target.py"][0]
    assert target_window.role == "target"
    assert target_window.mode == "full"

    # CASE 2: other.py is in raw_evidence_store -> it is included as reference/snippet
    search_cache = {
        "raw_evidence_store": [
            {"file": "other.py", "span": [1, 1], "code": "def other(): pass\n", "hash": "h1"}
        ]
    }
    context_pack_with_ev = tool._build_context_pack("target.py", search_cache)
    files_in_pack_with_ev = [w.file for w in context_pack_with_ev.windows]
    assert "target.py" in files_in_pack_with_ev
    assert "other.py" in files_in_pack_with_ev
    
    other_window = [w for w in context_pack_with_ev.windows if w.file == "other.py"][0]
    assert other_window.role == "reference"
    assert other_window.mode == "snippet"


@pytest.mark.asyncio
async def test_decision_edit_tool_compact_evidence_flag(temp_project: Path) -> None:
    import json
    mock_settings = MagicMock()
    mock_settings.cursor_validator_model = "deepseek"
    mock_settings.cursor_validator_command = ["pytest"]
    mock_settings.cursor_validator_timeout = 10.0
    mock_settings.cursor_observation_max_chars = 1000
    mock_settings.cursor_validator_semantic_timeout = 5.0
    mock_settings.prompt_cache_ttl = "5m"

    mock_harness = MagicMock()
    mock_harness.project_root = temp_project
    mock_harness.before_llm_call = AsyncMock(side_effect=lambda x: x)
    mock_harness.after_llm_call = AsyncMock()

    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = '{"action":"edit","answer":"","clarification":"","target_file":"target.py","patch":"<<<<<<< SEARCH\\ndef target(): pass\\n=======\\ndef target(): return 1\\n>>>>>>> REPLACE","suggested_completion":100}'
    mock_resp.usage = MagicMock()
    mock_llm.chat = AsyncMock(return_value=mock_resp)

    with patch("src.tools.assembled.decision_edit.CursorDecisionLLM") as mock_dec_cls, \
         patch("src.tools.assembled.decision_edit.CursorPatchApplier"), \
         patch("src.tools.assembled.decision_edit.CursorExecutor") as mock_exec_cls, \
         patch("src.tools.assembled.decision_edit.CursorValidator"):

        mock_dec = mock_dec_cls.return_value
        from src.agent.contracts import Decision
        parsed_decision = Decision(
            action="edit",
            target_file="target.py",
            patch="<<<<<<< SEARCH\ndef target(): pass\n=======\ndef target(): return 1\n>>>>>>> REPLACE",
            suggested_completion=100
        )
        mock_dec.parse = MagicMock(return_value=parsed_decision)
        
        mock_exec = mock_exec_cls.return_value
        from src.agent.contracts import ExecutionResult, ValidationResult
        exec_res = ExecutionResult(success=True, file="target.py")
        val_res = ValidationResult(success=True)
        mock_exec.execute_transaction = AsyncMock(return_value=(exec_res, val_res, MagicMock()))

        tool = DecisionEditTool(
            project_root=temp_project,
            settings=mock_settings,
            decision_llm=mock_llm,
            harness=mock_harness,
        )
        
        (temp_project / "target.py").write_text("def target(): pass\n", encoding="utf-8")
        tool.set_active_files(["target.py"])

        search_output = [
            {
                "file": "target.py",
                "span": [1, 1],
                "code": "def target(): pass",
                "related_functions": [
                    {"name": "decode_token", "file": "other.py", "span": [4, 8]}
                ],
            },
            {
                "file": "other.py",
                "span": [4, 8],
                "code": "def decode_token(): pass",
                "related_functions": [],
            },
        ]

        search_cache = {
            "search_output": json.dumps(search_output),
            "raw_evidence_store": []
        }

        await tool.execute(
            target_file="target.py",
            intent="add return 1",
            _search_cache=search_cache
        )

        # Verify that mock_dec.build_messages was called with a compact evidence_flag
        called_args, called_kwargs = mock_dec.build_messages.call_args
        passed_evidence_flag = called_kwargs.get("evidence_flag")
        assert passed_evidence_flag is not None
        assert passed_evidence_flag["target_file"] == "target.py"
        assert passed_evidence_flag["target_code_anchors"] == [
            {"file": "target.py", "span": [1, 1], "status": "loaded"}
        ]
        assert passed_evidence_flag["reference_code_anchors"] == [
            {"file": "other.py", "span": [4, 8], "status": "loaded"}
        ]
        assert passed_evidence_flag["first_hop_functions"] == [
            {"name": "decode_token", "file": "other.py", "span": [4, 8]}
        ]


@pytest.mark.asyncio
async def test_state_assembled_loop_success(temp_project: Path) -> None:
    mock_llm = AsyncMock()
    mock_tools = MagicMock()
    mock_tools.get_schemas.return_value = [
        {"type": "function", "function": {"name": "decision_edit"}}
    ]
    mock_harness = MagicMock()
    mock_harness.project_root = temp_project
    mock_harness.before_llm_call = AsyncMock(side_effect=lambda x: x)
    mock_harness.after_llm_call = AsyncMock()
    mock_harness.save_checkpoint = AsyncMock(return_value="cp1")
    mock_harness.phase_metrics = PhaseMetrics()

    mock_settings = MagicMock()
    mock_settings.max_turns = 3

    responses = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(
                id="inspect-1",
                name="view_symbol_code",
                arguments={
                    "target_file": "target.py",
                    "symbol": "run",
                },
            )],
            usage=None,
            model="coord-model",
        ),
        LLMResponse(
            content="Task completed successfully.",
            tool_calls=None,
            usage=MagicMock(),
            model="coord-model",
        ),
    ]
    response_index = 0
    
    async def mock_stream(*args, **kwargs):
        nonlocal response_index
        response = responses[response_index]
        response_index += 1
        yield ("", response)

    mock_llm.chat_stream = mock_stream
    mock_tools.call = AsyncMock(return_value=ToolResult(
        success=True,
        output="{}",
        metadata={
            "raw_evidence_store": [{
                "file": "target.py",
                "symbol": "run",
                "span": [1, 2],
                "code": "def run():\n    pass",
                "hash": "run123",
            }],
                "run_event": {
                    "kind": "evidence_discovered",
                    "grounded_slots": [
                        "target_implementation",
                        "authentication_context",
                    ],
                    "candidates": [],
                },
        },
    ))

    loop = StateAssembledLoop(
        llm=mock_llm,
        tools=mock_tools,
        harness=mock_harness,
        context=MagicMock(),
        permissions=MagicMock(),
        settings=mock_settings,
    )

    events = []
    async for event in loop.run("do something"):
        events.append(event)
        
    assert any(e.type == EventType.STREAM_START for e in events)
    assert any(e.type == EventType.FINAL_ANSWER and "Task completed successfully." in e.content for e in events)
    assert any(e.type == EventType.STREAM_END for e in events)
    assert mock_tools.get_schemas.call_count == 2
    phase = mock_harness.phase_metrics.records[0]
    assert phase.phase == "assembled_llm"
    assert phase.subtask_id == "1"
    assert phase.verdict == "ok"
    assert len(loop.state.core_context_history) == 2
    recorded = loop.state.core_context_history[0]
    assert recorded["call"] == 1
    assert recorded["step"] == 1
    assert recorded["messages"][0]["role"] == "system"
    assert recorded["messages"][-1]["role"] == "user"
    assert "do something" in recorded["messages"][-1]["content"]
    second_user_context = loop.state.core_context_history[1]["messages"][-1]["content"]
    assert "EXACT COVERAGE LOADED" not in second_user_context
    assert "FULL SOURCE LOADED" not in second_user_context
    assert "Database Schema:" not in second_user_context
    assert response_index == 2


@pytest.mark.asyncio
async def test_malformed_text_tool_call_retries_instead_of_finalizing(
    temp_project: Path,
) -> None:
    malformed = (
        '<tool_calls><invoke name="grep_search">'
        '<parameter name="pattern">auth</parameter>'
    )
    responses = [
        LLMResponse(content=malformed, tool_calls=None, usage=None, model="coord-model"),
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(
                id="inspect-1",
                name="view_symbol_code",
                arguments={
                    "target_file": "target.py",
                    "symbol": "run",
                },
            )],
            usage=None,
            model="coord-model",
        ),
        LLMResponse(content="Recovered final answer.", tool_calls=None, usage=None, model="coord-model"),
    ]
    call_index = 0

    async def chat_stream(*_args, **_kwargs):
        nonlocal call_index
        response = responses[call_index]
        call_index += 1
        yield ("", response)

    llm = MagicMock()
    llm.chat_stream = chat_stream
    tools = MagicMock()
    tools.get_schemas.return_value = [
        {"type": "function", "function": {"name": "grep_search"}}
    ]
    tools.call = AsyncMock(return_value=ToolResult(
        success=True,
        output="{}",
        metadata={
            "raw_evidence_store": [{
                "file": "target.py",
                "symbol": "run",
                "span": [1, 2],
                "code": "def run():\n    pass",
                "hash": "run123",
            }],
            "run_event": {
                "kind": "evidence_discovered",
                "grounded_slots": [
                    "target_implementation",
                    "authentication_context",
                ],
                "candidates": [],
            },
        },
    ))
    harness = MagicMock(project_root=temp_project)
    harness.before_llm_call = AsyncMock(side_effect=lambda value: value)
    harness.after_llm_call = AsyncMock()
    harness.save_checkpoint = AsyncMock(return_value=None)
    harness.phase_metrics = PhaseMetrics()
    loop = StateAssembledLoop(
        llm=llm,
        tools=tools,
        harness=harness,
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=3),
    )

    events = [event async for event in loop.run("diagnose function")]

    finals = [event.content for event in events if event.type == EventType.FINAL_ANSWER]
    assert finals == ["Recovered final answer."]
    assert malformed not in finals
    assert call_index == 3
    assert any(
        "Malformed text tool call detected" in message.content
        for message in loop.state.messages_history
    )


@pytest.mark.asyncio
async def test_responding_phase_never_executes_repeated_retrieval_calls(
    temp_project: Path,
) -> None:
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(
                id="inspect-1",
                name="view_symbol_code",
                arguments={
                    "target_file": "target.py",
                    "symbol": "run",
                },
            )],
            usage=None,
            model="coord-model",
        ),
        LLMResponse(
            content="I will search again.",
            tool_calls=[ToolCall(
                id="grep-1", name="grep_search", arguments={"pattern": "run"}
            )],
            usage=None,
            model="coord-model",
        ),
        LLMResponse(
            content="Still searching.",
            tool_calls=[ToolCall(
                id="grep-2", name="grep_search", arguments={"pattern": "run"}
            )],
            usage=None,
            model="coord-model",
        ),
    ]
    index = 0

    async def chat_stream(*_args, **_kwargs):
        nonlocal index
        response = responses[index]
        index += 1
        yield ("", response)

    llm = MagicMock(chat_stream=chat_stream)
    tools = MagicMock()
    tools.get_schemas.return_value = []
    tools.call = AsyncMock(return_value=ToolResult(
        success=True,
        output="{}",
        metadata={
            "raw_evidence_store": [{
                "file": "target.py",
                "symbol": "run",
                "span": [1, 2],
                "code": "def run():\n    pass",
                "hash": "run123",
            }],
            "run_event": {
                "kind": "evidence_discovered",
                "grounded_slots": ["target_implementation"],
                "candidates": [],
            },
        },
    ))
    harness = MagicMock(project_root=temp_project)
    harness.before_llm_call = AsyncMock(side_effect=lambda value: value)
    harness.after_llm_call = AsyncMock()
    harness.save_checkpoint = AsyncMock(return_value=None)
    harness.phase_metrics = PhaseMetrics()
    loop = StateAssembledLoop(
        llm=llm,
        tools=tools,
        harness=harness,
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=10),
    )

    events = [event async for event in loop.run("diagnose function")]

    assert tools.call.await_count == 1
    tool_results = [event for event in events if event.type == EventType.TOOL_RESULT]
    assert len(tool_results) == 3
    assert tool_results[1].data["success"] is False
    assert "not available in assembled mode" in tool_results[1].content
    assert tool_results[2].data["success"] is False
    assert "not available in assembled mode" in tool_results[2].content
    errors = [event.content for event in events if event.type == EventType.ERROR]
    assert not any("maximum steps" in message.lower() for message in errors)


@pytest.mark.asyncio
async def test_stream_llm_forwards_ready_final_token_cap(temp_project: Path) -> None:
    captured = {}

    async def chat_stream(*_args, **kwargs):
        captured.update(kwargs)
        yield (
            "",
            LLMResponse(content="done", tool_calls=None, usage=None, model="coord-model"),
        )

    llm = MagicMock()
    llm.chat_stream = chat_stream
    loop = StateAssembledLoop(
        llm=llm,
        tools=MagicMock(),
        harness=MagicMock(project_root=temp_project),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    chunks = [
        chunk
        async for chunk in loop._stream_llm(
            [{"role": "user", "content": "summarize"}],
            [],
            max_tokens=768,
        )
    ]
    assert captured["tools"] == []
    assert captured["max_tokens"] == 768
    assert chunks[-1]["response"].content == "done"


@pytest.mark.asyncio
async def test_state_assembled_loop_rejects_tools_outside_allow_list(
    temp_project: Path,
) -> None:
    tools = MagicMock()
    tools.call = AsyncMock()
    harness = MagicMock()
    harness.project_root = temp_project
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=tools,
        harness=harness,
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )

    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="read-1", name="read_file", arguments={"path": "x.py"})],
        usage=None,
        model="coord-model",
    )
    events = [event async for event in loop._process_tool_calls(response)]

    tools.call.assert_not_awaited()
    assert events[-1].type == EventType.TOOL_RESULT
    assert events[-1].data == {"tool": "read_file", "success": False}
    assert "not available in assembled mode" in (events[-1].content or "")


@pytest.mark.asyncio
async def test_state_assembled_loop_tool_execution(temp_project: Path) -> None:
    tools = MagicMock()
    tool_result = ToolResult(success=True, output="Retrieved.", metadata={"retrieved_files": ["target.py"]})
    tools.call = AsyncMock(return_value=tool_result)
    harness = MagicMock()
    harness.project_root = temp_project
    harness.phase_metrics = PhaseMetrics()
    
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=tools,
        harness=harness,
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    
    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(
            id="retrieve-1",
            name="codebase_retrieve",
            arguments={
                "query": "find run",
            },
        )],
        usage=None,
        model="coord-model",
    )
    
    events = [event async for event in loop._process_tool_calls(response)]
    
    tools.call.assert_awaited_once_with(
        "codebase_retrieve",
        {
            "query": "find run",
        },
    )
    
    # Check that a status event was yielded for the tool
    status_events = [e for e in events if e.type == EventType.STATUS]
    assert len(status_events) == 1
    assert "正在检索代码库: find run" in status_events[0].content
    
    # Check that the tool execution is tracked in phase_metrics
    records = harness.phase_metrics.records
    assert len(records) == 1
    assert records[0].phase == "tool_codebase_retrieve"
    assert records[0].verdict == "success"


@pytest.mark.asyncio
async def test_state_assembled_loop_parallel_tool_execution(temp_project: Path) -> None:
    import time
    tools = MagicMock()
    
    async def slow_call(name, args):
        await asyncio.sleep(0.1)
        return ToolResult(success=True, output=f"Retrieved {args.get('query')}.", metadata={"retrieved_files": []})
        
    tools.call = slow_call
    harness = MagicMock()
    harness.project_root = temp_project
    harness.phase_metrics = PhaseMetrics()
    
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=tools,
        harness=harness,
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    
    response = LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id="retrieve-1", name="codebase_retrieve", arguments={"query": "a"}),
            ToolCall(id="retrieve-2", name="codebase_retrieve", arguments={"query": "b"}),
        ],
        usage=None,
        model="coord-model",
    )
    
    t0 = time.monotonic()
    events = [event async for event in loop._process_tool_calls(response)]
    duration = time.monotonic() - t0
    
    assert duration < 0.18
    
    status_events = [e for e in events if e.type == EventType.STATUS]
    assert any("正在检索代码库: a" in str(e.content) for e in status_events)
    assert any("正在检索代码库: b" in str(e.content) for e in status_events)


def test_session_storage_and_recovery(tmp_path: Path) -> None:
    from src.harness.checkpoint.session_storage import SessionStorage
    
    transcripts_dir = tmp_path / "transcripts"
    storage = SessionStorage(transcripts_dir)
    
    session_id = "test_sess_123"
    event_data = {"type": "status", "content": "running test", "timestamp": 12345.0}
    storage.append_event(session_id, event_data)
    
    # Check that transcript is appended and correctly loaded
    loaded = storage.load_session_events(session_id)
    assert len(loaded) == 1
    assert loaded[0]["content"] == "running test"
    
    # Verify sidechain transcripts logging
    storage.append_sidechain_message(session_id, "step_1", "user", "run subtask")
    storage.append_sidechain_message(session_id, "step_1", "assistant", "done")
    
    sidechain_path = storage._get_sidechain_path(session_id, "step_1")
    assert sidechain_path.exists()


def test_context_loader_hierarchical_rules(tmp_path: Path) -> None:
    from src.agent.context_loader import ContextLoader
    
    project_root = tmp_path / "project"
    project_root.mkdir()
    
    mitkii_dir = project_root / ".mitkii"
    mitkii_dir.mkdir()
    (mitkii_dir / "rules.md").write_text("global project rules", encoding="utf-8")
    
    sub_dir = project_root / "src" / "tools"
    sub_dir.mkdir(parents=True)
    sub_mitkii = sub_dir / ".mitkii"
    sub_mitkii.mkdir()
    (sub_mitkii / "rules.md").write_text("sub directory tools rules", encoding="utf-8")
    
    loader = ContextLoader(project_root)
    
    active_files = ["src/tools/git.py"]
    git_file = sub_dir / "git.py"
    git_file.write_text("pass", encoding="utf-8")
    
    rules = loader.get_hierarchical_rules(active_files)
    assert "global project rules" in rules
    assert "sub directory tools rules" in rules


@pytest.mark.asyncio
async def test_state_assembled_loop_edit_failures_update_state(temp_project: Path) -> None:
    tools = MagicMock()
    
    val_err_result = ToolResult(
        success=False,
        output="validation error occurred",
        error="Validation failed",
        metadata={
            "execution": MagicMock(success=True),
            "validation": MagicMock(success=False, error="DEAD_SQL_ALIAS"),
        }
    )
    
    patch_err_result = ToolResult(
        success=False,
        output="patch failure occurred",
        error="Patch match failed",
        metadata=None
    )
    
    tools.call = AsyncMock(return_value=val_err_result)
    
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=tools,
        harness=MagicMock(project_root=temp_project),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    _prime_acting_run(loop)
    
    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="edit-1", name="decision_edit", arguments={"target_file": "target.py", "intent": "modify"})],
        usage=None,
        model="coord-model",
    )
    
    events = [event async for event in loop._process_tool_calls(response)]
    assert loop._validation_error() == "DEAD_SQL_ALIAS"
    assert loop.state.run_state.validation.status == "failed"
    assert loop.state.run_state.phase == RunPhase.ACTING
    
    tools.call = AsyncMock(return_value=patch_err_result)
    events = [event async for event in loop._process_tool_calls(response)]
    assert loop._last_error() is not None
    assert loop.state.run_state.retry_budget.failures >= 2


def test_parse_structured_error() -> None:
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    
    # 1. Test DEAD_SQL_ALIAS
    err_sql = "Schema validation failed: DEAD_SQL_ALIAS on file target.py"
    res = loop._parse_structured_error(err_sql)
    assert res["error_type"] == "SchemaValidationError"
    assert "DEAD_SQL_ALIAS" in res["message"]
    assert res["file"] == "target.py"

    # 2. Test standard traceback
    err_tb = """
Traceback (most recent call last):
  File "src/agent/state_assembled_loop.py", line 32, in queryLoop
    raise NameError("name 'x' is not defined")
NameError: name 'x' is not defined
"""
    res = loop._parse_structured_error(err_tb)
    assert res["error_type"] == "NameError"
    assert res["file"] == "state_assembled_loop.py"
    assert res["line"] == 32
    assert "name 'x' is not defined" in res["message"]

    # 3. Test linter style error
    err_linter = "src/tools/git.py:45:12: SyntaxError: invalid syntax"
    res = loop._parse_structured_error(err_linter)
    assert res["error_type"] == "SyntaxError"
    assert res["file"] == "git.py"
    assert res["line"] == 45
    assert "invalid syntax" in res["message"]


@pytest.mark.asyncio
async def test_last_tool_result_and_last_error(temp_project: Path) -> None:
    tools = MagicMock()
    
    val_err_result = ToolResult(
        success=False,
        output="validation error occurred",
        error="DEAD_SQL_ALIAS in target.py",
        metadata={
            "execution": MagicMock(success=True),
            "validation": MagicMock(success=False, error="DEAD_SQL_ALIAS"),
        }
    )
    
    success_result = ToolResult(
        success=True,
        output="success",
        metadata=None
    )
    
    tools.call = AsyncMock(return_value=val_err_result)
    
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=tools,
        harness=MagicMock(project_root=temp_project),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    _prime_acting_run(loop)
    
    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="edit-1", name="decision_edit", arguments={"target_file": "target.py", "intent": "modify"})],
        usage=None,
        model="coord-model",
    )
    
    events = [event async for event in loop._process_tool_calls(response)]
    
    assert loop._last_tool_result() is not None
    assert "decision_edit(target.py: modify) -> failed" in loop._last_tool_result()
    assert "DEAD_SQL_ALIAS" in loop._last_tool_result()
    assert loop._last_error() is not None
    assert loop._last_error()["error_type"] == "SchemaValidationError"
    assert loop._last_error()["file"] == "target.py"
    
    tools.call = AsyncMock(return_value=success_result)
    events = [event async for event in loop._process_tool_calls(response)]
    
    assert loop._last_tool_result() is not None
    assert "decision_edit(target.py: modify) -> success" in loop._last_tool_result()
    assert loop._last_error() is None


def test_merge_raw_evidence_adjacent_and_overlap(temp_project: Path) -> None:
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(project_root=temp_project),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    
    # Create a mock file on disk
    (temp_project / "list.py").write_text("\n".join(f"line {i}" for i in range(1, 101)), encoding="utf-8")
    
    raw_evidence = [
        {"file": "list.py", "span": [10, 20], "code": "old code 1", "hash": "h1"},
        {"file": "list.py", "span": [25, 30], "code": "old code 2", "hash": "h2"}, # gap is 4 lines (21, 22, 23, 24), should merge
        {"file": "list.py", "span": [40, 50], "code": "old code 3", "hash": "h3"}, # gap is 9 lines, should not merge
    ]
    
    merged = loop._merge_raw_evidence(temp_project, raw_evidence)
    
    # Should result in two merged blocks: [10, 30] and [40, 50]
    assert len(merged) == 2
    assert merged[0]["span"] == [10, 30]
    assert merged[1]["span"] == [40, 50]
    
    # Verify the code is read from disk
    assert "line 10" in merged[0]["code"]
    assert "line 30" in merged[0]["code"]
    assert "line 21" in merged[0]["code"]
    assert "line 40" in merged[1]["code"]
    assert "line 50" in merged[1]["code"]


def test_merge_raw_evidence_fallback_no_disk() -> None:
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(project_root=Path("/nonexistent")),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    
    raw_evidence = [
        {"file": "list.py", "span": [1, 3], "code": "line 1\nline 2\nline 3", "hash": "h1"},
        {"file": "list.py", "span": [2, 4], "code": "line 2\nline 3\nline 4", "hash": "h2"},
    ]
    
    # Should merge to [1, 4] and fallback to code merging
    merged = loop._merge_raw_evidence(Path("/nonexistent"), raw_evidence)
    assert len(merged) == 1
    assert merged[0]["span"] == [1, 4]
    assert merged[0]["code"] == "line 1\nline 2\nline 3\nline 4"


def test_merge_raw_grep_locators_without_code_key(temp_project: Path) -> None:
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(project_root=temp_project),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    (temp_project / "list.py").write_text(
        "def build_router(engine):\n    return engine\n",
        encoding="utf-8",
    )

    merged = loop._merge_raw_evidence(
        temp_project,
        [{
            "file": "list.py",
            "symbol": "build_router",
            "span": [1, 1],
            "match_line": "def build_router(engine):",
        }],
    )

    assert merged[0]["code"] == "def build_router(engine):"
    assert merged[0]["locator_only"] is True


def test_decision_edit_context_pack_with_layer_1(temp_project: Path) -> None:
    mock_settings = MagicMock()
    mock_llm = AsyncMock()
    mock_harness = MagicMock()
    mock_harness.project_root = temp_project

    tool = DecisionEditTool(
        project_root=temp_project,
        settings=mock_settings,
        decision_llm=mock_llm,
        harness=mock_harness,
    )
    
    # Create target.py and other.py
    (temp_project / "target.py").write_text("def target():\n    pass\n", encoding="utf-8")
    (temp_project / "other.py").write_text("def other_original():\n    pass\n", encoding="utf-8")
    
    tool.set_active_files(["target.py", "other.py"])
    
    search_cache = {
        "raw_evidence_store": [
            {"file": "other.py", "span": [1, 2], "code": "def other_cached():\n    pass\n", "hash": "abc"}
        ]
    }
    
    # Test case 1: other.py is read live from disk (overriding search cache)
    context_pack = tool._build_context_pack("target.py", search_cache=search_cache)
    windows = {w.file: w for w in context_pack.windows}
    assert "target.py" in windows
    assert "other.py" in windows
    assert "other_original" in windows["other.py"].content
    assert "other_cached" not in windows["other.py"].content

    # Test case 2: other.py falls back to search cache if file is deleted from disk
    import os
    os.remove(temp_project / "other.py")
    context_pack = tool._build_context_pack("target.py", search_cache=search_cache)
    windows = {w.file: w for w in context_pack.windows}
    assert "target.py" in windows
    assert "other.py" in windows
    assert "other_cached" in windows["other.py"].content


def test_context_assembly_never_injects_active_file_contents(temp_project: Path) -> None:
    assembly = ContextAssembly(temp_project)
    
    # Create files
    (temp_project / "modified.py").write_text("modified code", encoding="utf-8")
    (temp_project / "unmodified.py").write_text("unmodified code", encoding="utf-8")
    
    # Case 1: with empty modified_files filter - build_context_block does not output any active file block
    block1 = assembly.build_context_block(
        active_files=["modified.py", "unmodified.py"],
        search_cache={"context_projection": "cached search projection"},
        modified_files=[]
    )
    assert "cached search projection" in block1
    assert "modified code" not in block1
    assert "unmodified code" not in block1

    # Modified-file bookkeeping is status only; it never activates source injection.
    block2 = assembly.build_context_block(
        active_files=["modified.py", "unmodified.py"],
        search_cache={"context_projection": "cached search projection"},
        modified_files=["modified.py"]
    )
    assert "cached search projection" in block2
    assert "modified code" not in block2
    assert "unmodified code" not in block2


def test_merge_evidence_packs() -> None:
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    
    pack1 = {
        "query": "query 1",
        "intent": {"primary": "explain", "domain": ["auth"]},
        "grounding": {
            "files": [{"path": "a.py", "score": 0.8}],
            "symbols": [{"name": "foo", "file": "a.py", "score": 0.9}]
        },
        "evidence": [
            {
                "symbol": "foo",
                "file": "a.py",
                "what_it_does": "does foo",
                "why_relevant": "relevant foo",
                "risk_flags": ["flag1"]
            }
        ],
        "dependencies": [{"from": "foo", "to": "bar", "type": "call"}],
        "retrieval_graph": {"nodes": ["foo", "bar"], "edges": [["foo", "bar"]]},
        "coverage": {"auth": 0.8}
    }
    
    pack2 = {
        "query": "query 2",
        "intent": {"primary": "modify", "domain": ["sql", "auth"]},
        "grounding": {
            "files": [{"path": "a.py", "score": 0.5}, {"path": "b.py", "score": 0.7}],
            "symbols": [{"name": "foo", "file": "a.py", "score": 0.95}, {"name": "baz", "file": "b.py", "score": 0.6}]
        },
        "evidence": [
            {
                "symbol": "foo",
                "file": "a.py",
                "what_it_does": "does foo longer description",
                "why_relevant": "relevant foo short",
                "risk_flags": ["flag2"]
            },
            {
                "symbol": "baz",
                "file": "b.py",
                "what_it_does": "does baz",
                "why_relevant": "relevant baz",
                "risk_flags": []
            }
        ],
        "dependencies": [{"from": "baz", "to": "qux", "type": "call"}],
        "retrieval_graph": {"nodes": ["baz", "qux"], "edges": [["baz", "qux"]]},
        "coverage": {"auth": 0.6, "data": 0.9}
    }
    
    merged = loop._merge_evidence_packs(pack1, pack2)
    
    assert merged["query"] == "query 1; query 2"
    assert merged["intent"]["primary"] == "modify"
    assert set(merged["intent"]["domain"]) == {"auth", "sql"}
    
    # Grounding files scores maxed
    files = {f["path"]: f["score"] for f in merged["grounding"]["files"]}
    assert files["a.py"] == 0.8
    assert files["b.py"] == 0.7
    
    # Symbols merged, foo score taken from higher (0.95)
    symbols = {s["name"]: s["score"] for s in merged["grounding"]["symbols"]}
    assert symbols["foo"] == 0.95
    assert symbols["baz"] == 0.6
    
    # Evidence merged: what_it_does takes longer description, risk_flags unioned
    evidence = {ev["symbol"]: ev for ev in merged["evidence"]}
    assert evidence["foo"]["what_it_does"] == "does foo longer description"
    assert evidence["foo"]["why_relevant"] == "relevant foo short"
    assert set(evidence["foo"]["risk_flags"]) == {"flag1", "flag2"}
    
    # Coverage maxed
    assert merged["coverage"]["auth"] == 0.8
    assert merged["coverage"]["data"] == 0.9


def test_microcompact_retrieval_payload_clears_graph_only() -> None:
    payload = {
        "query": "auth",
        "evidence": [{"symbol": "auth_me", "what_it_does": "checks token"}],
        "dependencies": [{"from": "list_passengers", "to": "auth_me", "type": "call"}],
        "retrieval_graph": {
            "nodes": ["list_passengers", "auth_me"],
            "edges": [["list_passengers", "auth_me"]],
        },
    }

    compacted = json.loads(_microcompact_retrieval_payload(json.dumps(payload)))

    assert compacted["evidence"] == payload["evidence"]
    assert compacted["dependencies"] == []
    assert compacted["retrieval_graph"] == {"nodes": [], "edges": []}


def test_retrieval_code_anchor_collapses_into_separate_summary_next_turn() -> None:
    anchor = {
        "file": "main.py",
        "span": [1, 3],
        "code": "def auth_me(token):\n    if not token: raise ValueError()\n    return token",
        "related_functions": [{"name": "admin_route", "file": "main.py", "span": [8, 12]}],
    }
    state = AssembledState(
        run_state=replace(start_run("", edit_mode=False), step=5),
        search_cache={"search_output": json.dumps([anchor])},
        context_anchors=ContextAnchors(
            code=(anchor,),
            created_steps={"main.py:1-3": 3},
            last_updated_step=3,
        ),
        messages_history=(Message(role="tool", content=json.dumps([anchor])),),
    )

    collapsed = _collapse_retrieval_turn(state)
    assert collapsed.context_anchors.code == ()
    assert "search_output" not in collapsed.search_cache
    summary = collapsed.context_anchors.summaries["main.py:1-3"]
    assert "[CONTEXT COLLAPSE - main.py:1-3]" in summary
    assert "| `admin_route` | `main.py` | `8-12` | 一级强关联 |" in summary
    assert "return token" not in collapsed.messages_history[0].content

    projected = _search_cache_for_context(
        {"summary_anchors": collapsed.context_anchors.summaries},
        current_step=4,
    )
    assert "context_projection" not in projected


def test_context_anchor_double_buffer_collapses_old_and_keeps_fresh_code() -> None:
    old = {"file": "old.py", "span": [1, 2], "code": "def old(): pass", "related_functions": []}
    fresh = {"file": "fresh.py", "span": [4, 5], "code": "def fresh(): pass", "related_functions": []}
    state = AssembledState(
        run_state=replace(start_run("", edit_mode=False), step=3),
        context_anchors=ContextAnchors(
            code=(old, fresh),
            created_steps={"old.py:1-2": 1, "fresh.py:4-5": 2},
            last_updated_step=2,
        ),
        search_cache={"search_output": json.dumps([old, fresh])},
    )

    collapsed = _collapse_retrieval_turn(
        state,
        {"old.py:1-2": "[CONTEXT COLLAPSE - old.py:1-2] [已读/DO NOT RETRIEVE]"},
    )

    assert [item["file"] for item in collapsed.context_anchors.code] == ["fresh.py"]
    assert set(collapsed.context_anchors.summaries) == {"old.py:1-2"}
    assert collapsed.context_anchors.created_steps == {"fresh.py:4-5": 2}
    projected = _search_cache_for_context(_search_cache_view(collapsed), current_step=3)
    assert "def fresh" in projected["context_projection"]
    assert "def old" not in projected["context_projection"]


def test_collapse_retrieval_turn_collapses_single_dict_output() -> None:
    anchor = {
        "file": "target.py",
        "span": [10, 20],
        "code": "def func(): pass",
        "observation_code": "10| def func(): pass",
        "verbatim_code": "def func(): pass",
        "symbol": "func",
    }
    state = AssembledState(
        run_state=replace(start_run("", edit_mode=False), step=3),
        context_anchors=ContextAnchors(
            code=(anchor,),
            created_steps={"target.py:10-20": 1},
            last_updated_step=1,
        ),
        messages_history=(Message(role="tool", content=json.dumps(anchor)),),
    )

    collapsed = _collapse_retrieval_turn(
        state,
        {"target.py:10-20": "collapsed summary content"}
    )

    assert collapsed.context_anchors.code == ()
    assert "target.py:10-20" in collapsed.context_anchors.summaries
    assert "[ANCHOR MOVED TO MEMORY: target.py:10-20]" in collapsed.messages_history[0].content


def test_successful_edit_post_hook_cleans_stale_target_code(temp_project: Path) -> None:
    target = {"file": "target.py", "span": [1, 2], "code": "old", "related_functions": []}
    reference = {"file": "other.py", "span": [3, 4], "code": "keep", "related_functions": []}
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(project_root=temp_project),
        context=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(max_turns=1),
    )
    loop.state = AssembledState(
        context_anchors=ContextAnchors(
            code=(target, reference),
            summaries={"target.py:1-2": "stale", "other.py:3-4": "keep"},
            purposes={"target.py:1-2": "stale", "other.py:3-4": "keep"},
        ),
        search_cache={
            "search_output": json.dumps([target, reference]),
            "symbol_projections": [{"file": "target.py"}, {"file": "other.py"}],
        },
    )
    result = apply_post_tool_context_hook(
        "decision_edit",
        {"target_file": "target.py", "intent": "change"},
        ToolResult(success=True, output="done", metadata={}),
    )

    completion = loop._apply_context_update(result)

    assert completion == {
        "tool": "decision_edit",
        "target_file": "target.py",
        "status": "completed",
        "validation": "passed",
    }
    assert [item["file"] for item in loop.state.context_anchors.code] == ["other.py"]
    assert set(loop.state.context_anchors.summaries) == {"other.py:3-4"}
    assert [item["file"] for item in loop.state.search_cache["symbol_projections"]] == ["other.py"]
    assert [item["file"] for item in json.loads(loop.state.search_cache["search_output"])] == ["other.py"]


@pytest.mark.asyncio
async def test_deterministic_collapse_retrieval_before_core_llm(temp_project: Path) -> None:
    anchor = {
        "file": "main.py",
        "span": [1, 2],
        "code": "def auth_me():\n    return True",
        "symbol": "auth_me",
        "related_functions": [],
    }
    harness = MagicMock(project_root=temp_project)
    harness.phase_metrics = PhaseMetrics()
    settings = MagicMock(
        max_turns=3,
        max_context_tokens=100,  # small limit to trigger collapse
    )
    loop = StateAssembledLoop(
        llm=MagicMock(),
        tools=MagicMock(),
        harness=harness,
        context=MagicMock(),
        permissions=MagicMock(),
        settings=settings,
    )
    # mock count_messages_tokens to force it above threshold
    loop.llm.count_messages_tokens = MagicMock(return_value=200)

    loop.state = AssembledState(
        run_state=replace(start_run("", edit_mode=False), step=3),
        context_anchors=ContextAnchors(
            code=(anchor,),
            purposes={"main.py:1-2": "确认鉴权"},
            created_steps={"main.py:1-2": 1},
            last_updated_step=1,
        ),
    )

    await loop._collapse_retrieval_before_core_llm()

    # Verify that the anchor was folded/collapsed deterministically
    assert "main.py" in loop.state.context_anchors.summaries
    assert "确认鉴权" in loop.state.context_anchors.summaries["main.py"]
    assert len(loop.state.context_anchors.code) == 0


def test_retrieval_policy_blocks_repeated_signature() -> None:
    search_output = json.dumps(
        {
            "grounding": {
                "files": [{"path": "list.py", "score": 0.9}],
                "symbols": [{"name": "archive_passenger", "file": "list.py"}],
            },
            "evidence": [
                {
                    "symbol": "archive_passenger",
                    "file": "list.py",
                    "how_it_answers_query": "target endpoint",
                }
            ],
        }
    )
    snapshot = _retrieval_snapshot_from_output(
        "list.py archive_passenger authentication",
        search_output,
        step=1,
    )

    assert snapshot is not None
    assert snapshot["files"] == ["list.py"]
    assert snapshot["symbols"] == ["archive_passenger"]



def test_latest_symbol_slice_projection_uses_recent_two() -> None:
    cache = {
        "symbol_projections": [
            {
                "file": "a.py",
                "symbol": "old",
                "span": [1, 1],
                "projection_code": "1| old()",
            },
            {
                "file": "b.py",
                "symbol": "mid",
                "span": [2, 3],
                "projection_code": "2| def mid():\n3|   pass",
            },
            {
                "file": "c.py",
                "symbol": "new",
                "span": [4, 5],
                "projection_code": "4| def new():\n5|   pass",
                "truncated": True,
            },
        ]
    }

    block = _latest_symbol_slice_projection(cache)

    assert "old()" not in block
    assert '<symbol_slice file="b.py" symbol="mid" span="2-3">' in block
    assert '<symbol_slice file="c.py" symbol="new" span="4-5" truncated="true">' in block
    assert "4| def new():" in block


def test_build_deduped_loaded_anchors_block_prefers_symbol_slice_over_raw() -> None:
    cache = {
        "symbol_projections": [
            {
                "file": "svc.py",
                "symbol": "handler",
                "span": [1, 3],
                "projection_code": "1| def handler():\n2|   pass",
            },
        ],
        "raw_evidence_store": [
            {
                "file": "svc.py",
                "symbol": "handler",
                "span": [1, 3],
                "code": "def handler():\n    pass",
            },
        ],
    }

    block = _build_deduped_loaded_anchors_block(cache)

    assert block.count("def handler():") == 1
    assert "<symbol_slice" in block


def test_build_deduped_loaded_anchors_block_dedupes_raw_only() -> None:
    cache = {
        "raw_evidence_store": [
            {
                "file": "a.py",
                "symbol": "foo",
                "span": [1, 2],
                "code": "def foo(): pass",
            },
            {
                "file": "a.py",
                "symbol": "foo",
                "span": [1, 2],
                "code": "def foo(): pass",
            },
        ],
    }

    block = _build_deduped_loaded_anchors_block(cache)

    assert block.count("def foo():") == 1


def test_build_deduped_loaded_anchors_block_edit_ready_keeps_cross_file_projections() -> None:
    from src.agent.manifest import EvidenceItem, StepManifest, Sufficiency

    cache = {
        "symbol_projections": [
            {
                "file": "main.py",
                "symbol": "sqlalchemy_error_handler",
                "span": [49, 55],
                "projection_code": "49| async def sqlalchemy_error_handler(...):",
            },
            {
                "file": "list.py",
                "symbol": "build_router",
                "span": [16, 20],
                "projection_code": "16| def build_router(engine):",
            },
            {
                "file": "list.py",
                "symbol": "passenger_snapshot",
                "span": [80, 90],
                "projection_code": "80| def passenger_snapshot(...):",
            },
        ],
        "raw_evidence_store": [],
    }
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:sqlalchemy_error_handler",
                need="handler",
                type="symbol",
                role="observed",
                file="main.py",
                span=(49, 55),
                symbol="sqlalchemy_error_handler",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="handler",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 350),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )

    bootstrap = _build_deduped_loaded_anchors_block(cache)
    edit_ready_block = _build_deduped_loaded_anchors_block(
        cache,
        edit_ready=True,
        task_text="wire db logging into list.py routes",
        manifest=manifest,
    )

    assert "sqlalchemy_error_handler" not in bootstrap
    assert "sqlalchemy_error_handler" in edit_ready_block
    assert "build_router" in edit_ready_block
    assert "passenger_snapshot" in edit_ready_block


def test_grep_sql_locator_does_not_ground_relevant_schema_slot() -> None:
    from src.agent.state_assembled_loop import _run_evidence_event
    from src.hooks.post_tool_context import apply_post_tool_context_hook

    result = apply_post_tool_context_hook(
        "grep_search",
        {"patterns": ["CREATE TABLE"], "include": "*.sql"},
        ToolResult(
            success=True,
            output="{}",
            metadata={
                "raw_evidence_store": [{
                    "file": "db/init/init.sql",
                    "span": [1, 1],
                    "match_line": "CREATE TABLE ticket_order (",
                }]
            },
        ),
    )
    event = _run_evidence_event(result)
    assert event is not None
    grounded = {item.slot for item in event.evidence}
    assert "relevant_schema" not in grounded


def test_turn_context_block_puts_execution_card_last() -> None:
    block = build_turn_context_block(
        loaded_anchors="def handler(): pass",
        execution_card_text="### STEP EVIDENCE (current step)\ntools_available: grep_search",
    )

    assert block.index("LOADED CODE ANCHORS") < block.index("### STEP EVIDENCE (current step)")
    assert block.rstrip().endswith("tools_available: grep_search")


@pytest.mark.asyncio
async def test_view_symbol_code_tool(temp_project: Path) -> None:
    import json
    from src.tools.assembled.view_symbol_code import ViewSymbolCodeTool
    
    # Create target Python file
    target_code = (
        "def helper_func():\n"
        "    return 42\n"
        "\n"
        "class MyClass:\n"
        "    def run(self):\n"
        "        pass\n"
    )
    (temp_project / "helper.py").write_text(target_code, encoding="utf-8")
    
    tool = ViewSymbolCodeTool(
        project_root=temp_project,
        settings=MagicMock(),
    )
    
    # Test case 1: Locate function via AST/regex
    res = await tool.execute(target_file="helper.py", symbol="helper_func")
    assert res.success is True
    assert "helper_func" in res.output
    assert "1| def helper_func():" in res.output
    assert "2|     return 42" in res.output
    assert res.metadata["span"] == [1, 2]
    assert "return 42" in res.metadata["verbatim_code"]
    assert res.metadata["raw_evidence_store"][0]["code"] == res.metadata["verbatim_code"]
    
    # Test case 2: Locate class via AST/regex
    res2 = await tool.execute(target_file="helper.py", symbol="MyClass")
    assert res2.success is True
    assert "MyClass" in res2.output
    assert "4| class MyClass:" in res2.output
    assert "class MyClass" in res2.metadata["verbatim_code"]
    assert res2.metadata["span"] == [4, 6]
    
    # Test case 3: Locate via cached spans fallback
    mock_cache = {
        "search_output": json.dumps({
            "evidence": [
                {
                    "symbol": "custom_symbol",
                    "file": "helper.py",
                    "span": [4, 5]
                }
            ]
        })
    }
    res3 = await tool.execute(target_file="helper.py", symbol="custom_symbol", _search_cache=mock_cache)
    assert res3.success is True
    assert "4| class MyClass:" in res3.output
    assert "MyClass" in res3.metadata["verbatim_code"]
    assert res3.metadata["span"] == [4, 5]
    
    # Test case 4: Locate nested/dot-separated function (MyClass.run)
    res4 = await tool.execute(target_file="helper.py", symbol="MyClass.run")
    assert res4.success is True
    assert "run" in res4.output

    # Test case 5: Correct a call-site file to the unique definition file.
    (temp_project / "main.py").write_text(
        "app.include_router(build_router(engine))\n", encoding="utf-8"
    )
    (temp_project / "list.py").write_text(
        "def build_router(engine):\n    return engine\n", encoding="utf-8"
    )
    res5 = await tool.execute(target_file="main.py", symbol="build_router")
    assert res5.success is True
    assert res5.metadata["requested_file"] == "main.py"
    assert res5.metadata["resolved_file"] == "list.py"
    assert res5.metadata["file"] == "list.py"
    assert "def build_router" in res5.metadata["verbatim_code"]

    # Test case 6: Preserve route decorators as endpoint evidence.
    (temp_project / "routes.py").write_text(
        '@router.delete("/passengers/{passenger_id}")\n'
        "def archive_passenger(passenger_id: int):\n"
        "    return passenger_id\n",
        encoding="utf-8",
    )
    res6 = await tool.execute(target_file="routes.py", symbol="archive_passenger")
    assert res6.success is True
    assert res6.metadata["span"] == [1, 3]
    assert res6.metadata["verbatim_code"].startswith("@router.delete")
    assert res4.metadata["span"] == [5, 6]
    
    # Test case 5: Locate variable assignment
    assign_code = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
    )
    (temp_project / "app_main.py").write_text(assign_code, encoding="utf-8")
    res5 = await tool.execute(target_file="app_main.py", symbol="app")
    assert res5.success is True
    assert "app = FastAPI()" in res5.metadata["verbatim_code"]
    assert res5.metadata["span"] == [2, 2]

    # Factory pattern: probing generic `app` resolves to create_app body.
    (temp_project / "factory_main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    app.include_router(build_router(engine))\n"
        "    return app\n",
        encoding="utf-8",
    )
    res_factory = await tool.execute(target_file="factory_main.py", symbol="app")
    assert res_factory.success is True
    assert res_factory.metadata["symbol"] == "create_app"
    assert "def create_app" in res_factory.metadata["verbatim_code"]

    # Non-existent symbol
    res6 = await tool.execute(target_file="helper.py", symbol="missing_func")
    assert res6.success is False
    assert "not found" in res6.error

    # Test case 7: Locate SQL view/table via regex (with optional backticks/quotes)
    sql_code = (
        "-- Some comments\n"
        "CREATE VIEW `view_ticket_report_detail` AS\n"
        "SELECT * FROM tickets;\n"
        "CREATE TABLE my_table (id INT);\n"
    )
    (temp_project / "init.sql").write_text(sql_code, encoding="utf-8")
    
    res7 = await tool.execute(target_file="init.sql", symbol="view_ticket_report_detail")
    assert res7.success is True
    assert res7.metadata["span"] == [2, 3]
    assert "CREATE VIEW `view_ticket_report_detail`" in res7.metadata["verbatim_code"]
    assert "SELECT * FROM tickets;" in res7.metadata["verbatim_code"]

    res8 = await tool.execute(target_file="init.sql", symbol="my_table")
    assert res8.success is True
    assert res8.metadata["span"] == [4, 4]
    assert "CREATE TABLE my_table" in res8.metadata["verbatim_code"]

    # Test case 8: Project-wide fallback matching SQL file when called on a different file
    res9 = await tool.execute(target_file="app_main.py", symbol="view_ticket_report_detail")
    assert res9.success is True
    assert res9.metadata["requested_file"] == "app_main.py"
    assert res9.metadata["resolved_file"] == "init.sql"
    assert res9.metadata["file"] == "init.sql"
    assert "CREATE VIEW `view_ticket_report_detail`" in res9.metadata["verbatim_code"]

    # SQL programmable objects respect custom delimiter boundaries.
    programmable_sql = (
        "DELIMITER $$\n"
        "CREATE PROCEDURE `sp_create_ticket_order`(IN p_id INT)\n"
        "BEGIN\n"
        "  INSERT INTO ticket_order (p_id) VALUES (p_id);\n"
        "  SELECT LAST_INSERT_ID();\n"
        "END$$\n"
        "CREATE TRIGGER `tg_release_seat` AFTER UPDATE ON ticket_order\n"
        "FOR EACH ROW\n"
        "BEGIN\n"
        "  UPDATE flight_seat SET is_booked = 0 WHERE seat_id = OLD.seat_id;\n"
        "END$$\n"
        "DELIMITER ;\n"
    )
    (temp_project / "routines.sql").write_text(programmable_sql, encoding="utf-8")
    procedure = await tool.execute(
        target_file="routines.sql", symbol="sp_create_ticket_order"
    )
    trigger = await tool.execute(target_file="routines.sql", symbol="tg_release_seat")
    assert procedure.success is True
    assert "SELECT LAST_INSERT_ID();" in procedure.metadata["verbatim_code"]
    assert procedure.metadata["verbatim_code"].endswith("END$$")
    assert trigger.success is True
    assert "CREATE TRIGGER `tg_release_seat`" in trigger.metadata["verbatim_code"]
    assert trigger.metadata["verbatim_code"].endswith("END$$")

    # Test case 9: Wildcard loads whole file; filename-as-symbol is rejected
    res10 = await tool.execute(target_file="init.sql", symbol="init.sql")
    assert res10.success is False
    assert "filename" in (res10.error or "").lower()

    res11 = await tool.execute(target_file="init.sql", symbol="*")
    assert res11.success is True
    assert res11.metadata["span"] == [1, 4]


@pytest.mark.asyncio
async def test_decision_edit_tool_with_target_file_span(temp_project: Path) -> None:
    mock_settings = MagicMock()
    mock_settings.cursor_validator_model = "none"
    mock_settings.cursor_validator_command = ["pytest"]
    mock_settings.cursor_validator_timeout = 10.0
    mock_settings.cursor_observation_max_chars = 1000
    mock_settings.prompt_cache_ttl = "5m"

    mock_harness = MagicMock()
    mock_harness.project_root = temp_project
    mock_harness.before_llm_call = AsyncMock(side_effect=lambda x: x)
    mock_harness.after_llm_call = AsyncMock()

    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = '{"action":"edit","target_file":"target.py","patch":"..."}'
    mock_resp.usage = MagicMock()
    mock_llm.chat = AsyncMock(return_value=mock_resp)

    # Setup target file with multiple lines
    target_file = temp_project / "target.py"
    target_file.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    with patch("src.tools.assembled.decision_edit.CursorDecisionLLM") as mock_dec_cls, \
         patch("src.tools.assembled.decision_edit.CursorPatchApplier"), \
         patch("src.tools.assembled.decision_edit.CursorExecutor") as mock_exec_cls, \
         patch("src.tools.assembled.decision_edit.CursorValidator"):

        mock_dec = mock_dec_cls.return_value
        from src.agent.contracts import Decision
        parsed_decision = Decision(
            action="edit",
            target_file="target.py",
            patch="<<<<<<< SEARCH\nline3\n=======\nline3_modified\n>>>>>>> REPLACE",
            suggested_completion=100
        )
        mock_dec.parse = MagicMock(return_value=parsed_decision)
        mock_dec.build_messages = MagicMock(return_value=[])
        
        mock_exec = mock_exec_cls.return_value
        from src.agent.contracts import ExecutionResult, ValidationResult
        exec_res = ExecutionResult(success=True, file="target.py")
        val_res = ValidationResult(success=True)
        mock_exec.execute_transaction = AsyncMock(return_value=(exec_res, val_res, MagicMock()))

        tool = DecisionEditTool(
            project_root=temp_project,
            settings=mock_settings,
            decision_llm=mock_llm,
            harness=mock_harness,
        )

        # Call with context_window containing target.py span
        result = await tool.execute(
            target_file="target.py",
            intent="edit line3",
            context_window=[
                {"file": "target.py", "span": [2, 4], "reason": "target span"}
            ]
        )
        assert result.success is True

        # Verify that only lines 2-4 of target.py were loaded as the target ContextWindow
        call_args = mock_dec.build_messages.call_args[1]
        context_pack = call_args["context_pack"]
        windows = context_pack.windows
        
        target_windows = [w for w in windows if w.file == "target.py"]
        assert len(target_windows) == 1
        assert target_windows[0].role == "target"
        assert target_windows[0].mode == "snippet"
        assert target_windows[0].start_line == 2
        assert target_windows[0].end_line == 4
        assert target_windows[0].content == "line2\nline3\nline4"
