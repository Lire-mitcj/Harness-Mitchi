from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.context_assembly import ContextAssembly
from src.agent.events import EventType
from src.agent.state_assembled_loop import ASSEMBLED_TOOL_NAMES, StateAssembledLoop
from src.agent.types import LLMResponse, Message, ToolCall, ToolResult, RiskLevel
from src.harness.gates.phase_metrics import PhaseMetrics
from src.tools.assembled.codebase_retrieve import CodebaseRetrieveTool
from src.tools.assembled.decision_edit import DecisionEditTool


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


def test_context_assembly(temp_project: Path) -> None:
    assembly = ContextAssembly(temp_project)
    prompt = assembly.load_system_prompt()
    assert "Mock Lead Coordinator Prompt." in prompt

    messages = assembly.assemble(
        user_query="implement feature X",
        active_files=["target.py"],
        checklist=["[ ] step 1", "[x] step 2"],
        git_diff="dummy diff",
        validation_error="dummy error",
        messages_history=[Message(role="assistant", content="Thinking...")],
    )
    
    system_msg = next(m for m in messages if m["role"] == "system")
    user_msg = next(m for m in messages if m["role"] == "user")
    
    assert "Mock Lead Coordinator Prompt." in system_msg["content"]
    assert "Follow project rules." in system_msg["content"]
    assert "step 1" in user_msg["content"]
    assert "target.py" in user_msg["content"]
    assert "def run():" in user_msg["content"]
    assert "dummy diff" in user_msg["content"]
    assert "dummy error" in user_msg["content"]


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
    
    mock_llm = AsyncMock()
    mock_tools = MagicMock()

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
        from src.agent.cursor_contracts import Decision
        parsed_decision = Decision(
            action="edit",
            target_file="target.py",
            patch="<<<<<<< SEARCH\ndef run():\n    pass\n=======\ndef run():\n    return 42\n>>>>>>> REPLACE",
            suggested_completion=100
        )
        mock_dec.parse = MagicMock(return_value=parsed_decision)
        
        mock_exec = mock_exec_cls.return_value
        from src.agent.cursor_contracts import ExecutionResult, ValidationResult
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

        result = await tool.execute(target_file="target.py", instruction="change run to return 42")
        assert result.success is True
        assert "应用成功" in result.output


@pytest.mark.asyncio
async def test_state_assembled_loop_success(temp_project: Path) -> None:
    mock_llm = AsyncMock()
    mock_tools = MagicMock()
    mock_harness = MagicMock()
    mock_harness.project_root = temp_project
    mock_harness.before_llm_call = AsyncMock(side_effect=lambda x: x)
    mock_harness.after_llm_call = AsyncMock()
    mock_harness.save_checkpoint = AsyncMock(return_value="cp1")
    mock_harness.phase_metrics = PhaseMetrics()

    mock_settings = MagicMock()
    mock_settings.max_turns = 3

    mock_resp = LLMResponse(
        content="Task completed successfully.",
        tool_calls=None,
        usage=MagicMock(),
        model="coord-model"
    )
    
    async def mock_stream(*args, **kwargs):
        yield ("", mock_resp)

    mock_llm.chat_stream = mock_stream

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
    mock_tools.get_schemas.assert_called_once_with(include=ASSEMBLED_TOOL_NAMES)
    phase = mock_harness.phase_metrics.records[0]
    assert phase.phase == "assembled_llm"
    assert phase.subtask_id == "1"
    assert phase.verdict == "ok"


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
        tool_calls=[ToolCall(id="retrieve-1", name="codebase_retrieve", arguments={"query": "find run"})],
        usage=None,
        model="coord-model",
    )
    
    events = [event async for event in loop._process_tool_calls(response)]
    
    tools.call.assert_awaited_once_with("codebase_retrieve", {"query": "find run"})
    
    # Check that a status event was yielded for the tool
    status_events = [e for e in events if e.type == EventType.STATUS]
    assert len(status_events) == 1
    assert "正在检索代码库: find run" in status_events[0].content
    
    # Check that the tool execution is tracked in phase_metrics
    records = harness.phase_metrics.records
    assert len(records) == 1
    assert records[0].phase == "tool_codebase_retrieve"
    assert records[0].verdict == "success"

