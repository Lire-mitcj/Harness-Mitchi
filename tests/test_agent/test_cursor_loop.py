from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.cursor_ast_structure import AstNode, CursorAstStructureLayer
from src.agent.cursor_context_pack_builder import CursorContextPackBuilder
from src.agent.cursor_contracts import (
    ContextPack,
    ContextWindow,
    Decision,
    ExecutionResult,
    InterHint,
    RetrievalResult,
    RetrievalSymbol,
    ValidationResult,
)
from src.agent.cursor_decision import CursorDecisionLLM, DecisionError
from src.agent.cursor_graph_bridge import GraphBridgeResult, GraphNode
from src.agent.cursor_inter_llm import CursorInterLLM
from src.agent.cursor_loop import CursorLoop
from src.agent.cursor_patch_applier import CursorPatchApplier
from src.agent.cursor_query_bridge import CursorQueryBridge, QueryBridgeResult
from src.agent.cursor_repo_map_lookup import CandidateSymbol, CursorRepoMapLookup
from src.agent.cursor_retriever import CursorRetriever
from src.agent.cursor_semantic_tagger import CursorSemanticTagger
from src.agent.cursor_validator import CursorValidator
from src.agent.events import AgentEvent, EventType
from src.agent.types import LLMResponse, TokenUsage
from src.config.permissions import PermissionManager
from src.config.settings import MitKIISettings
from src.harness.cursor.manager import CursorStateManager
from src.harness.engine import HarnessEngine
from src.indexer.ctags import CtagsIndexResult, CtagsSymbol
from src.indexer.repo_map import build_repo_map
from src.tools.registry import ToolRegistry


def _settings(tmp_path: Path, **overrides: object) -> MitKIISettings:
    values: dict[str, object] = {
        "data_dir": tmp_path / ".mitkii",
        "cursor_evaluation_dir": tmp_path / "eval_json",
        "cursor_evaluation_case_file": None,
        "cursor_inter_enabled": False,
        "cursor_semantic_tags_enabled": False,
        "cursor_max_steps": 3,
        "cursor_validator_command": ["pytest"],
    }
    values.update(overrides)
    return MitKIISettings(**values)


def _harness(tmp_path: Path, settings: MitKIISettings) -> HarnessEngine:
    return HarnessEngine.create(settings, project_root=tmp_path)


def _loop(tmp_path: Path, **settings: object) -> CursorLoop:
    config = _settings(tmp_path, **settings)
    llm = MagicMock()
    loop = CursorLoop(
        llm=llm,
        tools=ToolRegistry(),
        harness=_harness(tmp_path, config),
        context=None,
        permissions=PermissionManager(),
        settings=config,
    )
    loop.query_bridge.generate_raw = AsyncMock(
        return_value=_bridge_json(_bridge_result("sample"))
    )
    return loop


def _loop_with_context(
    tmp_path: Path,
    context: object,
    **settings: object,
) -> CursorLoop:
    config = _settings(tmp_path, **settings)
    loop = CursorLoop(
        llm=MagicMock(),
        tools=ToolRegistry(),
        harness=_harness(tmp_path, config),
        context=context,
        permissions=PermissionManager(),
        settings=config,
    )
    loop.query_bridge.generate_raw = AsyncMock(
        return_value=_bridge_json(_bridge_result("sample"))
    )
    return loop


def _bridge_result(*terms: str) -> QueryBridgeResult:
    return QueryBridgeResult(
        intent="explain",
        expanded_terms=list(terms),
        keywords=[],
        symbols=[],
        file_hints=[],
    )


def _bridge_json(result: QueryBridgeResult) -> str:
    return json.dumps({
        "intent": result.intent,
        "expanded_terms": result.expanded_terms,
        "keywords": result.keywords,
        "symbols": result.symbols,
        "file_hints": result.file_hints,
    })


def test_state_manager_is_bounded_and_cumulative() -> None:
    manager = CursorStateManager(max_bytes=2048)
    state = manager.initial("任务" * 2000)
    for index in range(20):
        state = manager.after_execution(
            state,
            "src/example.py",
            f"patch-{index}-" + "x" * 5000,
            ExecutionResult(success=False, file="src/example.py", error="e" * 5000),
        )
    assert manager.serialized_size(state) <= 2048
    assert "patch-19" in state.last_patch
    assert set(state.to_dict()) == {
        "task", "current_file", "last_patch", "last_observation", "status",
        "current_step", "max_steps", "stage_completion", "execution_traces", "patch_memory",
        "decision_signatures", "retry_bias", "decision_cost_total", "entropy_score",
    }

    assert len(state.patch_memory) == 20
    assert len(state.execution_traces) == 20


def test_patch_applier_exact_and_whitespace_match(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    target.chmod(0o754)
    applier = CursorPatchApplier(tmp_path)
    patch = (
        "<<<<<<< SEARCH\n"
        "def value():\n"
        "  return 1\n"
        "=======\n"
        "def value():\n"
        "    return 2\n"
        ">>>>>>> REPLACE"
    )
    success, error = applier.apply_patch("sample.py", patch)
    assert success, error
    assert target.read_text(encoding="utf-8") == "def value():\n    return 2\n"
    assert os.stat(target).st_mode & 0o777 == 0o754


def test_patch_applier_dry_runs_all_blocks(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    original = "a = 1\nb = 2\n"
    target.write_text(original, encoding="utf-8")
    patch = (
        "<<<<<<< SEARCH\na = 1\n=======\na = 10\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nmissing = 3\n=======\nmissing = 4\n>>>>>>> REPLACE"
    )
    success, error = CursorPatchApplier(tmp_path).apply_patch("sample.py", patch)
    assert not success
    assert "mismatch" in error
    assert target.read_text(encoding="utf-8") == original


def test_patch_applier_rejects_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    patch = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
    success, error = CursorPatchApplier(tmp_path).apply_patch("../outside.py", patch)
    assert not success
    assert "outside project root" in error
    assert outside.read_text(encoding="utf-8") == "x = 1\n"


@pytest.mark.asyncio
async def test_retriever_uses_exact_grep_and_symbols_without_embedding(tmp_path: Path) -> None:
    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("def process_order():\n    return 'needle'\n", encoding="utf-8")
    symbol = SimpleNamespace(
        file_path="src/service.py",
        name="process_order",
        start_line=1,
        end_line=2,
    )
    repo_map = SimpleNamespace(all_symbols=[symbol], symbols=[], file_scores={})
    service = MagicMock()
    service.map = repo_map
    retriever = CursorRetriever(tmp_path, repo_map_service=service)
    result = await retriever.retrieve(("src/service.py", "process_order", "needle"))
    assert result.files == ("src/service.py",)
    assert tuple(symbol.name for symbol in result.symbols) == ("process_order",)


@pytest.mark.asyncio
async def test_query_bridge_rewrites_chinese_before_retrieval(
    tmp_path: Path,
) -> None:
    target = tmp_path / "src" / "validator.py"
    target.parent.mkdir()
    target.write_text(
        "def validate_payload():\n    raise ValueError('invalid')\n",
        encoding="utf-8",
    )
    symbol = SimpleNamespace(
        file_path="src/validator.py",
        name="validate_payload",
        start_line=1,
        end_line=2,
    )
    service = MagicMock()
    service.map = SimpleNamespace(all_symbols=[symbol], symbols=[], file_scores={})
    retriever = CursorRetriever(tmp_path, repo_map_service=service)
    bridge = CursorQueryBridge(llm=None)
    rewritten = bridge.fallback("修复校验错误处理")
    result = await retriever.retrieve(rewritten.search_terms())

    assert "src/validator.py" in result.files
    assert tuple(symbol.name for symbol in result.symbols) == ("validate_payload",)


def test_context_builder_expands_without_selecting(tmp_path: Path) -> None:
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(f"def {name[0]}():\n    return 1\n", encoding="utf-8")
    retrieval = RetrievalResult(files=("a.py", "b.py"))
    pack = CursorContextPackBuilder(tmp_path, max_files=2).build_context(retrieval)
    assert pack.candidate_files == ("a.py", "b.py")
    assert all(window.content for window in pack.windows)


def test_semantic_tagger_only_adds_display_annotations() -> None:
    pack = ContextPack(windows=(ContextWindow(
        file="validator.py",
        start_line=1,
        end_line=2,
        content="def validate_sql():\n    raise ValueError()",
    ),))
    annotations = CursorSemanticTagger().annotate(pack)
    assert annotations.tags_by_file["validator.py"] == (
        "validation", "sql_check", "error_handling"
    )
    assert pack.windows[0].semantic_tags == ()


def test_decision_parser_enforces_single_retrieved_file() -> None:
    valid = json.dumps({
        "action": "edit",
        "answer": "",
        "clarification": "",
        "target_file": "a.py",
        "patch": "patch",
    })
    decision = CursorDecisionLLM.parse(valid, ("a.py", "b.py"))
    assert decision.target_file == "a.py"
    invalid = json.dumps({
        "action": "edit",
        "answer": "",
        "clarification": "",
        "target_file": "outside.py",
        "patch": "patch",
    })
    with pytest.raises(DecisionError, match="not a retrieved candidate"):
        CursorDecisionLLM.parse(invalid, ("a.py",))


def test_decision_parser_rejects_planner_fields() -> None:
    payload = {
        "action": "edit",
        "answer": "",
        "clarification": "",
        "target_file": "a.py",
        "patch": "patch",
        "steps": ["one", "two"],
    }
    with pytest.raises(DecisionError, match="unexpected"):
        CursorDecisionLLM.parse(json.dumps(payload), ("a.py",))


@pytest.mark.asyncio
async def test_decision_llm_is_tool_free() -> None:
    payload = json.dumps({
        "action": "answer",
        "answer": "done",
        "clarification": "",
        "target_file": "",
        "patch": "",
    })
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=LLMResponse(
        content=payload,
        tool_calls=None,
        usage=None,
        model="test",
    ))
    pack = ContextPack(windows=())
    decision, _ = await CursorDecisionLLM(llm).decide(
        state_text="{}",
        context_pack=pack,
    )
    assert decision.action == "answer"
    assert llm.chat.await_args.kwargs["tools"] is None


@pytest.mark.asyncio
async def test_inter_hint_is_strict_and_tool_free() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=LLMResponse(
        content='{"intent":"repair failing behavior","domains":["backend API"],'
        '"concepts":["validation"],"ambiguity":false,"confidence":0.7}',
        tool_calls=None,
        usage=None,
        model="test",
    ))
    hint = await CursorInterLLM(llm).generate("fix it")
    assert hint == InterHint(
        intent="repair failing behavior",
        domains=("backend API",),
        concepts=("validation",),
        ambiguity=False,
        confidence=0.7,
    )
    assert llm.chat.await_args.kwargs["tools"] is None


def test_inter_hint_preserves_open_vocabulary_and_ambiguity() -> None:
    content = json.dumps({
        "intent": "enumerate all meanings of view",
        "domains": ["relational database schema", "server-rendered web UI", "MVC"],
        "concepts": ["SQL VIEW", "page template", "presentation projection"],
        "ambiguity": True,
        "confidence": 0.61,
    })

    hint = CursorInterLLM.parse(content)

    assert hint is not None
    assert hint.domains == (
        "relational database schema",
        "server-rendered web UI",
        "MVC",
    )
    assert hint.concepts == (
        "SQL VIEW",
        "page template",
        "presentation projection",
    )
    assert hint.ambiguity is True


def test_inter_fallback_does_not_guess_domain_from_keywords() -> None:
    hint = CursorInterLLM.fallback("项目里有哪些视图")

    assert hint.domains == ()
    assert hint.concepts == ()
    assert hint.ambiguity is True
    assert hint.confidence == 0.0


@pytest.mark.asyncio
async def test_validator_reports_success_and_bounded_failure(tmp_path: Path) -> None:
    success = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
    )
    assert (await success.validate()).success

    failure = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "import sys; print('x' * 500); sys.exit(2)"),
        max_error_chars=80,
    )
    execution = await failure.validate_execution()
    assert execution["pass"] is False
    assert len(str(execution["error"])) <= 96
    assert "truncated" in str(execution["error"])


@pytest.mark.asyncio
async def test_layered_validator_ast_failure_rolls_back(tmp_path: Path) -> None:
    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
    )

    result = await validator.validate(
        target_file="sample.py",
        patch="patch",
        original_content="def keep():\n    return 1\n",
        patched_content="def keep(:\n    return 1\n",
        user_intent="change implementation",
    )

    assert not result.success
    assert result.decision == "rollback"
    assert result.ast is not None
    assert result.ast["pass"] is False
    assert "python_syntax_error" in result.ast["issues"]


def test_ast_validator_does_not_parse_python_embedded_sql_as_sql(
    tmp_path: Path,
) -> None:
    validator = CursorValidator(tmp_path)
    validator.sql_parser.parse_file = MagicMock(side_effect=AssertionError(
        "python files must not be parsed as raw SQL"
    ))
    patched = '''
from sqlalchemy import text

def build_order_detail_sql(where_clause: str):
    return text("""
        SELECT v.order_id, v.real_name
        FROM view_ticket_report_detail v
        WHERE v.order_id = :order_id
    """)
'''

    result = validator.validate_ast(
        "main.py",
        original_content="def build_order_detail_sql(where_clause: str):\n    pass\n",
        patched_content=patched,
    )

    assert result["pass"] is True
    assert result["issues"] == []
    validator.sql_parser.parse_file.assert_not_called()


def test_ast_validator_still_parses_sql_files(tmp_path: Path) -> None:
    validator = CursorValidator(tmp_path)
    validator.sql_parser.parse_file = MagicMock(return_value=())

    result = validator.validate_ast(
        "db/init/init.sql",
        original_content="",
        patched_content="SELECT * FROM view_ticket_report_detail;",
    )

    assert result["pass"] is True
    validator.sql_parser.parse_file.assert_called_once()


def test_ast_validator_symbol_missing_is_limited_to_patch_scope(tmp_path: Path) -> None:
    validator = CursorValidator(tmp_path)
    original = (
        "def target():\n"
        "    return 1\n\n"
        "def unrelated_legacy():\n"
        "    return 2\n"
    )
    patched = "def target():\n    return 3\n"
    patch = (
        "<<<<<<< SEARCH\n"
        "def target():\n"
        "    return 1\n"
        "=======\n"
        "def target():\n"
        "    return 3\n"
        ">>>>>>> REPLACE"
    )

    result = validator.validate_ast(
        "main.py",
        original_content=original,
        patched_content=patched,
        patch=patch,
    )

    assert result["pass"] is True
    assert result["issues"] == []


@pytest.mark.asyncio
async def test_schema_alias_safety_is_limited_to_patch_sql_fragment(
    tmp_path: Path,
) -> None:
    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
    )
    original = '''
def legacy_sql():
    return """
    SELECT old_alias.id
    FROM view_ticket_report_detail v
    """

def target_sql():
    return """
    SELECT o.order_id
    FROM ticket_order o
    """
'''
    patched = '''
def legacy_sql():
    return """
    SELECT old_alias.id
    FROM view_ticket_report_detail v
    """

def target_sql():
    return """
    SELECT v.order_id
    FROM view_ticket_report_detail v
    """
'''
    patch = (
        "<<<<<<< SEARCH\n"
        "def target_sql():\n"
        "    return \"\"\"\n"
        "    SELECT o.order_id\n"
        "    FROM ticket_order o\n"
        "    \"\"\"\n"
        "=======\n"
        "def target_sql():\n"
        "    return \"\"\"\n"
        "    SELECT v.order_id\n"
        "    FROM view_ticket_report_detail v\n"
        "    \"\"\"\n"
        ">>>>>>> REPLACE"
    )

    result = await validator.validate(
        target_file="main.py",
        patch=patch,
        original_content=original,
        patched_content=patched,
        user_intent="把订单查询的sql改成使用视图查询",
    )

    assert result.success
    assert result.semantic is not None
    alias = result.semantic["details"]["schema"]["checks"]["alias_safety"]
    assert alias["pass"] is True
    assert alias["dead_aliases"] == []


@pytest.mark.asyncio
async def test_layered_validator_execution_failure_is_advisory_signal(
    tmp_path: Path,
) -> None:
    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(2)"),
    )

    result = await validator.validate(
        target_file="sample.py",
        patch="patch",
        original_content="def keep():\n    return 1\n",
        patched_content="def keep():\n    return 2\n",
        user_intent="change implementation",
    )

    assert result.success
    assert result.decision == "commit"
    assert result.ast is not None
    assert result.ast["pass"] is True
    assert result.semantic is not None
    assert result.semantic["pass"] is True
    assert result.execution is not None
    assert result.execution["pass"] is False
    assert result.execution["status"] == "FAIL"


@pytest.mark.asyncio
async def test_layered_validator_no_tests_is_soft_execution_signal(
    tmp_path: Path,
) -> None:
    validator = CursorValidator(
        tmp_path,
        command=(
            sys.executable,
            "-c",
            "print('collected 0 items'); raise SystemExit(5)",
        ),
    )

    result = await validator.validate(
        target_file="sample.py",
        patch="patch",
        original_content="def keep():\n    return 1\n",
        patched_content="def keep():\n    return 2\n",
        user_intent="change implementation",
    )

    assert result.success
    assert result.decision == "commit"
    assert result.score == 1.0
    assert result.execution is not None
    assert result.execution["pass"] is True
    assert result.execution["status"] == "NO_TESTS"
    assert result.execution["warning"] == "pytest has no matching tests"


@pytest.mark.asyncio
async def test_layered_validator_infra_error_is_soft_execution_signal(
    tmp_path: Path,
) -> None:
    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise ModuleNotFoundError('pytest')"),
    )

    result = await validator.validate(
        target_file="sample.py",
        patch="patch",
        original_content="def keep():\n    return 1\n",
        patched_content="def keep():\n    return 2\n",
        user_intent="change implementation",
    )

    assert result.success
    assert result.decision == "commit"
    assert result.execution is not None
    assert result.execution["status"] == "INFRA_ERROR"


@pytest.mark.asyncio
async def test_layered_validator_rejects_dead_sql_alias_inside_python_string(
    tmp_path: Path,
) -> None:
    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
    )
    original = '''
def build_order_detail_sql():
    return f"""
    SELECT o.order_id, p.real_name
    FROM ticket_order o
    JOIN passenger_info p ON p.p_id = o.p_id
    """
'''
    patched = '''
def build_order_detail_sql():
    return f"""
    SELECT o.order_id, p.real_name
    FROM view_ticket_report_detail v
    """
'''

    result = await validator.validate(
        target_file="service.py",
        patch="patch",
        original_content=original,
        patched_content=patched,
        user_intent="把订单查询的sql改成使用视图查询",
    )

    assert not result.success
    assert result.decision == "rollback"
    assert result.ast is not None
    assert result.ast["pass"] is True
    assert result.semantic is not None
    assert result.semantic["pass"] is False
    details = result.semantic["details"]
    assert isinstance(details, dict)
    schema = details["schema"]
    assert isinstance(schema, dict)
    checks = schema["checks"]
    assert isinstance(checks, dict)
    alias_details = checks["alias_safety"]
    assert isinstance(alias_details, dict)
    assert set(alias_details["dead_aliases"]) == {"o", "p"}


@pytest.mark.asyncio
async def test_schema_validator_rejects_select_fields_missing_without_relation_equivalence(
    tmp_path: Path,
) -> None:
    schema_dir = tmp_path / "db" / "init"
    schema_dir.mkdir(parents=True)
    (schema_dir / "init.sql").write_text(
        "CREATE VIEW view_ticket_report_detail AS\n"
        "SELECT order_id, real_name FROM ticket_order;\n",
        encoding="utf-8",
    )
    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
    )
    patched = '''
def build_order_detail_sql():
    return """
    SELECT v.order_id, v.real_name, v.missing_col
    FROM view_ticket_report_detail v
    """
'''

    result = await validator.validate(
        target_file="service.py",
        patch="patch",
        original_content="",
        patched_content=patched,
        user_intent="把订单查询的sql改成使用视图查询",
    )

    assert not result.success
    assert result.decision == "rollback"
    assert result.semantic is not None
    schema = result.semantic["details"]["schema"]
    assert isinstance(schema, dict)
    assert "SELECT_FIELD_NOT_IN_VIEW" in schema["issues"]
    assert schema["missing_fields"] == ["missing_col"]


@pytest.mark.asyncio
async def test_schema_validator_allows_field_mismatch_when_view_relation_equivalent(
    tmp_path: Path,
) -> None:
    schema_dir = tmp_path / "db" / "init"
    schema_dir.mkdir(parents=True)
    (schema_dir / "init.sql").write_text(
        "CREATE VIEW view_ticket_report_detail AS\n"
        "SELECT o.order_id, p.real_name\n"
        "FROM ticket_order o\n"
        "JOIN passenger_info p ON p.p_id = o.p_id\n"
        "JOIN flight_info f ON f.flight_id = o.flight_id;\n",
        encoding="utf-8",
    )
    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
    )
    original = '''
def build_order_detail_sql():
    return """
    SELECT o.order_id, p.real_name, f.flight_no
    FROM ticket_order o
    JOIN passenger_info p ON p.p_id = o.p_id
    JOIN flight_info f ON f.flight_id = o.flight_id
    """
'''
    patched = '''
def build_order_detail_sql():
    return """
    SELECT v.order_id, v.real_name, v.flight_no
    FROM view_ticket_report_detail v
    """
'''

    result = await validator.validate(
        target_file="service.py",
        patch="patch",
        original_content=original,
        patched_content=patched,
        user_intent="把订单查询的sql改成使用视图查询",
    )

    assert result.success
    assert result.decision == "commit"
    schema = result.semantic["details"]["schema"]
    assert isinstance(schema, dict)
    assert schema["missing_fields"] == []
    binding = schema["checks"]["view_semantic_binding"]
    assert isinstance(binding, dict)
    assert "FIELD_MISMATCH_ALLOWED_BY_RELATION_EQUIVALENCE" in binding["warnings"]
    check = binding["checks"][0]
    assert check["relation_equivalent"] is True
    assert check["dependency_equivalent"] is True
    assert check["field_containment"] is False


@pytest.mark.asyncio
async def test_llm_judge_is_advisory_and_cannot_override_schema_failure(
    tmp_path: Path,
) -> None:
    class OverrideLLM:
        async def chat(self, *args: object, **kwargs: object) -> LLMResponse:
            return LLMResponse(
                content=json.dumps({
                    "semantic_analysis": "looks fine",
                    "risk": "low",
                    "pass": True,
                    "score": 1.0,
                }),
                tool_calls=None,
                usage=None,
                model="judge",
            )

    validator = CursorValidator(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(0)"),
        semantic_llm=OverrideLLM(),
    )
    patched = '''
def build_order_detail_sql():
    return """
    SELECT old_alias.order_id
    FROM view_ticket_report_detail v
    """
'''

    result = await validator.validate(
        target_file="service.py",
        patch="patch",
        original_content="",
        patched_content=patched,
        user_intent="把订单查询的sql改成使用视图查询",
    )

    assert not result.success
    assert result.decision == "rollback"
    details = result.semantic["details"]
    assert details["llm_judge"]["risk"] == "low"


@pytest.mark.asyncio
async def test_loop_applies_patch_and_validates(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    loop = _loop(tmp_path)
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(files=("sample.py",)))
    patch = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
    loop.llm.chat = AsyncMock(return_value=_decision_response(
        Decision(action="edit", target_file="sample.py", patch=patch)
    ))
    loop.validator.validate = AsyncMock(return_value=ValidationResult(success=True))

    events = [event async for event in loop.run("change value")]
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert loop.state.status == "success"
    assert EventType.FINAL_ANSWER in {event.type for event in events}
    loop.validator.validate.assert_awaited_once()
    metrics_file = tmp_path / "eval_json" / "harness_evaluation_metrics.jsonl"
    payload = json.loads(metrics_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["target_file"] == "sample.py"
    assert payload["layer2"]["patch_correctness"] == 1.0
    assert payload["layer2"]["execution_success"] == 1.0
    assert payload["layer2"]["code_diff_correctness"] == 1.0
    assert payload["layer2"]["task_passed"] is True
    event_text = "\n".join(str(event.content or "") for event in events)
    assert "[Evaluation] Layer 1 unavailable: no RetrievalTestCase GT is bound." in event_text
    assert "LAYER 2: TASK SUCCESS METRICS" in event_text
    assert "Patch Correctness" in event_text


@pytest.mark.asyncio
async def test_loop_retries_clarify_when_context_is_available(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("value = 1\n", encoding="utf-8")
    loop = _loop(tmp_path, cursor_reranker_enabled=False)
    loop.retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(files=("sample.py",))
    )
    loop.llm.chat = AsyncMock(side_effect=[
        _decision_response(
            Decision(action="ask_clarify", clarification="unnecessary clarification")
        ),
        _decision_response(Decision(action="answer", answer="grounded answer")),
    ])

    events = [event async for event in loop.run("explain sample")]

    assert loop.llm.chat.await_count == 2
    assert loop.state.retry_bias == 1
    assert loop.state.current_step == 2
    assert loop.state.status == "success"
    assert any(event.content == "grounded answer" for event in events)


@pytest.mark.asyncio
async def test_loop_blocks_decision_when_layer1_recall_is_incomplete(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    loop = _loop_with_context(
        tmp_path,
        SimpleNamespace(
            file_tracker=None,
            repo_map_service=None,
            evaluation_expected_targets=("missing.py:target:1-2",),
        ),
    )
    loop.decision_llm.chat = AsyncMock()
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(files=("sample.py",)))

    events = [event async for event in loop.run("change value")]

    loop.decision_llm.chat.assert_not_called()
    assert loop.state.status == "failed"
    assert any(
        event.type == EventType.FINAL_ANSWER
        and event.content == "Layer 1 retrieval recall below CI threshold"
        for event in events
    )


@pytest.mark.asyncio
async def test_loop_binds_gt_and_records_real_fusion_trace(tmp_path: Path) -> None:
    from src.agent.cursor_evaluator import RetrievalTestCase

    (tmp_path / "sample.py").write_text("value = 1\n", encoding="utf-8")
    loop = _loop_with_context(
        tmp_path,
        SimpleNamespace(
            file_tracker=None,
            repo_map_service=None,
            evaluation_test_case=RetrievalTestCase(
                name="sample-retrieval",
                ground_truth=("sample.py",),
            ),
        ),
        cursor_reranker_enabled=False,
    )
    loop.retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(files=("sample.py",))
    )
    loop.llm.chat = AsyncMock(
        return_value=_decision_response(Decision(action="answer", answer="done"))
    )

    events = [event async for event in loop.run("explain sample")]

    assert loop.eval_harness is not None
    assert len(loop.eval_harness.traces) == 1
    trace = loop.eval_harness.traces[0]
    assert trace.retrieved == ("sample.py",)
    assert trace.fused_files == ("sample.py",)
    assert loop.eval_harness.evaluate().recall == 1.0
    assert any("GT-bound cumulative trace" in str(event.content) for event in events)


@pytest.mark.asyncio
async def test_loop_loads_evaluation_case_from_settings(tmp_path: Path) -> None:
    case_file = tmp_path / "retrieval_test_case.json"
    case_file.write_text(
        '{"name":"other-case","ground_truth":["other.py"]}\n'
        '{"name":"configured-case","ground_truth":["sample.py"]}\n',
        encoding="utf-8",
    )

    loop = _loop(
        tmp_path,
        cursor_evaluation_case_file=case_file,
        cursor_evaluation_case_name="configured-case",
    )

    assert loop.eval_harness is None
    await loop._ensure_evaluation_case()

    assert loop.eval_harness is not None
    assert loop.eval_harness.test_case.name == "configured-case"
    assert loop.eval_harness.test_case.ground_truth == ("sample.py",)


@pytest.mark.asyncio
async def test_loop_notifies_file_tracker_after_successful_patch(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    tracker = MagicMock()
    loop = _loop_with_context(
        tmp_path,
        SimpleNamespace(file_tracker=tracker, repo_map_service=None),
    )
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(files=("sample.py",)))
    patch = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
    loop.llm.chat = AsyncMock(return_value=_decision_response(
        Decision(action="edit", target_file="sample.py", patch=patch)
    ))
    loop.validator.validate = AsyncMock(return_value=ValidationResult(success=True))

    _ = [event async for event in loop.run("change value")]

    tracker.record_edit.assert_called_once_with("sample.py")


@pytest.mark.asyncio
async def test_loop_retries_with_latest_validation_observation(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    loop = _loop(tmp_path)
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(files=("sample.py",)))
    first = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
    second = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 3\n>>>>>>> REPLACE"
    loop.llm.chat = AsyncMock(side_effect=[
        _decision_response(Decision(action="edit", target_file="sample.py", patch=first)),
        _decision_response(Decision(action="edit", target_file="sample.py", patch=second)),
    ])
    loop.validator.validate = AsyncMock(side_effect=[
        ValidationResult(success=False, error="tests failed"),
        ValidationResult(success=True),
    ])

    events = [event async for event in loop.run("change value")]
    assert target.read_text(encoding="utf-8") == "value = 3\n"
    assert loop.state.status == "success"
    assert loop.llm.chat.await_count == 2
    second_messages = loop.llm.chat.await_args_list[1].args[0]
    assert "tests failed" in second_messages[-1]["content"]
    assert EventType.ERROR not in {event.type for event in events}


@pytest.mark.asyncio
async def test_loop_exhaustion_marks_failed(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    loop = _loop(tmp_path, cursor_max_steps=2)
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(files=("sample.py",)))
    loop.llm.chat = AsyncMock(return_value=_decision_response(
        Decision(action="edit", target_file="sample.py", patch="bad patch")
    ))
    events = [event async for event in loop.run("change value")]
    assert loop.state.status == "failed"
    assert EventType.ERROR in {event.type for event in events}
    assert target.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.asyncio
async def test_loop_uses_harness_probe_metrics_and_phase_timing(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("x = 1\n", encoding="utf-8")
    loop = _loop(tmp_path, cursor_inter_enabled=True)
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(files=("sample.py",)))
    loop.llm.chat = AsyncMock(side_effect=[
        LLMResponse(
            content='{"intent":"explain code","domains":["software"],'
            '"concepts":["implementation"],"ambiguity":false,"confidence":0.8}',
            tool_calls=None,
            usage=TokenUsage(10, 3, 13),
            model="inter-model",
        ),
        _decision_response(Decision(action="answer", answer="done")),
    ])
    before = loop.harness.before_llm_call
    after = loop.harness.after_llm_call
    loop.harness.before_llm_call = AsyncMock(wraps=before)
    loop.harness.after_llm_call = AsyncMock(wraps=after)

    events = [event async for event in loop.run("explain this")]
    summary = loop.get_probe_metrics()

    assert loop.harness.before_llm_call.await_count == 2
    assert loop.harness.after_llm_call.await_count == 2
    assert summary["total_calls"] == 2
    assert summary["total_tokens"] == 26
    phases = {item["phase"] for item in summary["phases"]}
    assert {"cursor_inter", "cursor_retrieval", "cursor_context", "cursor_decision"} <= phases
    assert len([event for event in events if event.type == EventType.COST_UPDATE]) == 2


@pytest.mark.asyncio
async def test_loop_routes_inter_and_decision_to_separate_clients(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("x = 1\n", encoding="utf-8")
    config = _settings(
        tmp_path,
        cursor_inter_enabled=True,
        cursor_reranker_enabled=False,
    )
    inter_llm = MagicMock()
    decision_llm = MagicMock()
    inter_llm.chat = AsyncMock(return_value=LLMResponse(
        content='{"intent":"modify validation behavior","domains":["backend"],'
        '"concepts":["validator"],"ambiguity":false,"confidence":0.9}',
        tool_calls=None,
        usage=None,
        model="inter-model",
    ))
    decision_llm.chat = AsyncMock(return_value=_decision_response(
        Decision(action="answer", answer="done")
    ))
    loop = CursorLoop(
        llm=decision_llm,
        inter_llm=inter_llm,
        decision_llm=decision_llm,
        tools=ToolRegistry(),
        harness=_harness(tmp_path, config),
        context=None,
        permissions=PermissionManager(),
        settings=config,
    )
    loop.query_bridge.generate_raw = AsyncMock(
        return_value=_bridge_json(_bridge_result("validator"))
    )
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(files=("sample.py",)))

    events = [event async for event in loop.run("change validator")]

    inter_llm.chat.assert_awaited_once()
    decision_llm.chat.assert_awaited_once()
    assert EventType.FINAL_ANSWER in {event.type for event in events}


@pytest.mark.asyncio
async def test_loop_runs_inter_and_query_bridge_concurrently(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("x = 1\n", encoding="utf-8")
    config = _settings(
        tmp_path,
        cursor_inter_enabled=True,
        cursor_reranker_enabled=False,
    )
    inter_llm = MagicMock()
    decision_llm = MagicMock()
    inter_started = asyncio.Event()
    bridge_started = asyncio.Event()

    async def inter_chat(*args: object, **kwargs: object) -> LLMResponse:
        inter_started.set()
        await asyncio.wait_for(bridge_started.wait(), timeout=0.5)
        return LLMResponse(
            content='{"intent":"inspect project","domains":["software"],'
            '"concepts":["views"],"ambiguity":true,"confidence":0.7}',
            tool_calls=None,
            usage=None,
            model="inter-model",
        )

    inter_llm.chat = AsyncMock(side_effect=inter_chat)
    decision_llm.chat = AsyncMock(return_value=_decision_response(
        Decision(action="answer", answer="done")
    ))
    loop = CursorLoop(
        llm=decision_llm,
        inter_llm=inter_llm,
        decision_llm=decision_llm,
        tools=ToolRegistry(),
        harness=_harness(tmp_path, config),
        context=None,
        permissions=PermissionManager(),
        settings=config,
    )

    async def bridge_raw(query: str) -> str:
        bridge_started.set()
        await asyncio.wait_for(inter_started.wait(), timeout=0.5)
        return _bridge_json(_bridge_result("sample"))

    loop.query_bridge.generate_raw = AsyncMock(side_effect=bridge_raw)
    loop.retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(files=("sample.py",))
    )

    events = await asyncio.wait_for(
        _collect_events(loop, "项目里有哪些视图"),
        timeout=1.0,
    )

    parallel = [
        event.data
        for event in events
        if event.type == EventType.STATUS
        and event.data
        and event.data.get("parallel_task_id")
    ]
    assert {(item["parallel_task_id"], item["parallel_state"]) for item in parallel} == {
        ("inter", "running"),
        ("query_bridge", "running"),
        ("inter", "done"),
        ("query_bridge", "done"),
    }


@pytest.mark.asyncio
async def test_loop_keeps_graph_bridge_and_retriever_separate(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("def my_profile_page():\n    pass\n", encoding="utf-8")
    loop = _loop(tmp_path)
    loop.graph_bridge.expand_candidates = AsyncMock(return_value=GraphBridgeResult(
        expanded_symbols=("my_profile_page",),
        expanded_files=("app.py",),
        graph_nodes=(
            GraphNode(
                id="app.py:my_profile_page:1",
                name="my_profile_page",
                type="function",
                file="app.py",
                score=0.9,
                distance=1,
            ),
        ),
    ))
    loop.retriever.retrieve = AsyncMock(
        return_value=RetrievalResult()
    )

    _ = [event async for event in loop.run("项目里有哪些视图")]

    loop.graph_bridge.expand_candidates.assert_awaited_once()
    first_batch = loop.retriever.retrieve.await_args_list[0].args[0]
    assert "my_profile_page" not in first_batch
    assert "app.py" not in first_batch


async def _collect_events(loop: CursorLoop, message: str) -> list[AgentEvent]:
    return [event async for event in loop.run(message)]


def _decision_response(decision: Decision) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({
            "action": decision.action,
            "answer": decision.answer,
            "clarification": decision.clarification,
            "target_file": decision.target_file,
            "patch": decision.patch,
        }),
        tool_calls=None,
        usage=TokenUsage(10, 3, 13),
        model="decision-model",
    )


@pytest.mark.asyncio
async def test_query_bridge_retriever_and_fusion_are_separate_stages(tmp_path: Path) -> None:
    from src.agent.cursor_fusion import CursorFusionEngine
    from src.agent.types import LLMResponse

    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("def process_order():\n    return 'needle'\n", encoding="utf-8")

    symbol = SimpleNamespace(
        file_path="src/service.py",
        name="process_order",
        start_line=1,
        end_line=2,
    )
    service = MagicMock()
    service.map = SimpleNamespace(all_symbols=[symbol], symbols=[], file_scores={})

    llm = MagicMock()
    llm.chat = AsyncMock(return_value=LLMResponse(
        content='{"intent":"modify","expanded_terms":["needle"],'
        '"keywords":["process_order"],"symbols":["process_order"],'
        '"file_hints":["service"]}',
        tool_calls=None,
        usage=None,
        model="test",
    ))

    bridge = CursorQueryBridge(llm)
    rewritten = await bridge.generate("修改 process_order")
    retriever = CursorRetriever(tmp_path, repo_map_service=service)
    raw = await retriever.retrieve(rewritten.search_terms())
    assert tuple(symbol.name for symbol in raw.symbols) == ("process_order",)

    repo_lookup = CursorRepoMapLookup(service)
    candidates = repo_lookup.lookup(rewritten)
    ast_nodes = CursorAstStructureLayer(tmp_path).ground(candidates, limit=4)
    raw = retriever.score_candidates(
        ast_nodes=ast_nodes,
        candidates=candidates,
        bridge=rewritten,
        graph=GraphBridgeResult(),
    )
    result = CursorFusionEngine.fuse(raw, rewritten, max_files=12, max_symbols=12)

    assert "src/service.py" in result.files
    assert len(result.symbols) > 0
    assert result.symbols[0].name == "process_order"


@pytest.mark.asyncio
async def test_loop_aborts_early_if_retrieval_empty(tmp_path: Path) -> None:
    loop = _loop(tmp_path, cursor_inter_enabled=False)
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult())
    loop.llm.chat = AsyncMock()

    events = [event async for event in loop.run("test query")]
    assert loop.state.status == "failed"
    assert loop.llm.chat.await_count == 0
    assert EventType.FINAL_ANSWER in {event.type for event in events}
    assert events[-2].content == "No matching code found"


def test_decision_llm_builds_messages_with_evidence_flag() -> None:
    from src.agent.cursor_contracts import ContextPack, ContextWindow
    from src.agent.cursor_decision import CursorDecisionLLM

    llm = MagicMock()
    decision = CursorDecisionLLM(llm)

    # 1. Non-empty context pack
    pack = ContextPack(windows=(
        ContextWindow(
            file="a.py",
            start_line=1,
            end_line=5,
            content="print('hello')",
            semantic_tags=("tag1",),
        ),
    ))
    messages = decision.build_messages(
        state_text="state",
        context_pack=pack,
        hint=None,
    )
    assert len(messages) == 2
    system_msg = messages[0]["content"]
    user_msg = messages[1]["content"]

    assert "strict local action classifier" in system_msg
    assert "Use ONLY CURRENT_CONTEXT" in system_msg
    assert "Do not propose refactors" in system_msg
    assert "target file is absent" in system_msg
    assert "SEARCH/REPLACE" in system_msg
    assert "Patch format example" in system_msg
    assert "<<<<<<< SEARCH" in system_msg
    assert "diff --git" in system_msg
    assert "EVIDENCE_FLAG" in user_msg

    # Assert evidence flag JSON structure
    flag_line = user_msg.split("\n")[1]
    flag_data = json.loads(flag_line)
    assert flag_data["retrieval_results"] == ["a.py"]
    assert flag_data["can_answer"] is True

    # 2. Empty context pack
    empty_pack = ContextPack(windows=())
    empty_messages = decision.build_messages(
        state_text="state",
        context_pack=empty_pack,
        hint=None,
    )
    empty_user_msg = empty_messages[1]["content"]
    empty_flag_line = empty_user_msg.split("\n")[1]
    empty_flag_data = json.loads(empty_flag_line)
    assert empty_flag_data["retrieval_results"] == []
    assert empty_flag_data["can_answer"] is False


def test_query_bridge_robust_parsing() -> None:
    from src.agent.cursor_query_bridge import CursorQueryBridge, repair_json

    bridge = CursorQueryBridge(llm=None)

    # 1. Test parsing valid dual-output format
    dual_output_content = """
    ```json
    {
      "raw_thought": "I should list views",
      "structured_json": {
        "intent": "list",
        "expanded_terms": ["view", "flight"],
        "keywords": ["视图"],
        "symbols": ["v_flight"],
        "file_hints": ["db"]
      }
    }
    ```
    """
    res = bridge.parse(dual_output_content)
    assert res.intent == "query"
    assert res.expanded_terms == ["view", "flight"]
    assert res.keywords == ["视图"]
    assert res.symbols == []
    assert res.file_hints == ["db"]

    # 2. Test JSON repair: single quotes, unquoted keys, trailing commas
    dirty_json = """
    {
      intent: 'modify',
      'expanded_terms': ["term1", "term2",],
      "keywords": ["key"],
      "symbols": [],
      "file_hints": ["api"],
    }
    """
    repaired = repair_json(dirty_json)
    assert repaired["intent"] == "modify"
    assert repaired["expanded_terms"] == ["term1", "term2"]

    # 3. Test Regex fallback for totally broken content
    broken_content = """
    Thinking: I want to list views.
    intent: list
    expanded_terms: ["term_a", "term_b"]
    keywords: ["视图"]
    symbols: ["func_a"]
    file_hints: ["ui"]
    """
    res_broken = bridge.parse(broken_content)
    assert res_broken.intent == "query"
    assert res_broken.expanded_terms == ["term_a", "term_b"]
    assert res_broken.keywords == ["视图"]
    assert res_broken.symbols == []
    assert res_broken.file_hints == ["ui"]


def test_query_bridge_receives_only_original_query() -> None:
    bridge = CursorQueryBridge(llm=None)

    messages = bridge.build_messages("项目里有哪些视图")
    payload = json.loads(messages[1]["content"])

    assert payload == {"user_query": "项目里有哪些视图"}
    system_prompt = messages[0]["content"].casefold()
    assert "inter_hint" not in system_prompt
    assert "inter llm" not in system_prompt


def test_query_bridge_fallback_preserves_ambiguous_view_interpretations() -> None:
    result = CursorQueryBridge(llm=None).fallback("项目里有哪些视图")

    assert {"view", "component", "page", "CREATE VIEW", "schema", "sql"} <= set(
        result.expanded_terms
    )
    assert "view" in result.search_terms()


def test_query_bridge_and_fusion_bound_and_deduplicate_terms() -> None:
    from src.agent.cursor_query_bridge import CursorQueryBridge

    content = json.dumps({
        "intent": "explain",
        "expanded_terms": [f"term_{index}" for index in range(100)],
        "keywords": ["Needle", "needle", "process_order"],
        "symbols": ["process_order", "OrderService"],
        "file_hints": [f"path_{index}" for index in range(20)],
    })

    result = CursorQueryBridge(llm=None).parse(content)
    fused = result.search_terms()

    assert len(result.expanded_terms) == 8
    assert len(result.file_hints) == 6
    assert result.symbols == []
    assert fused[:2] == ("Needle", "process_order")
    assert len(fused) <= 12
    assert len({term.casefold() for term in fused}) == len(fused)


def test_fusion_reranks_files_and_symbols_from_bridge_signals() -> None:
    from src.agent.cursor_fusion import CursorFusionEngine

    bridge = QueryBridgeResult(
        intent="query",
        expanded_terms=["view"],
        keywords=[],
        symbols=[],
        file_hints=["ui"],
    )
    raw = RetrievalResult(
        files=("src/models/view.py", "src/ui/flight_view.py"),
        symbols=(
            RetrievalSymbol("src/models/view.py", "DatabaseView", 1, 2, score=0.4),
            RetrievalSymbol("src/ui/flight_view.py", "FlightView", 1, 2, score=0.9),
        ),
    )

    result = CursorFusionEngine.fuse(raw, bridge, max_files=2, max_symbols=2)

    assert result.files[0] == "src/ui/flight_view.py"
    assert result.symbols[0].name == "FlightView"
    decision = CursorFusionEngine.decide(raw, bridge, max_files=2, max_symbols=2)
    assert decision.final_context[0] == "src/ui/flight_view.py:FlightView:1-2"
    assert decision.confidence == 0.9


def test_retriever_keeps_saturated_candidate_pool_for_reranker(tmp_path: Path) -> None:
    symbols = tuple(
        SimpleNamespace(
            file_path="main.py" if index == 17 else f"noise_{index}.py",
            name="build_order_detail_sql" if index == 17 else f"render_order_card_{index}",
            start_line=index + 1,
            end_line=index + 1,
            kind="dml_select" if index == 17 else "function",
            tables_referenced=("ticket_order",) if index == 17 else (),
        )
        for index in range(60)
    )
    candidates = tuple(CandidateSymbol(symbol, 0.5) for symbol in symbols)
    ast_nodes = tuple(
        AstNode(
            symbol=symbol.name,
            file=symbol.file_path,
            signature=f"def {symbol.name}()",
            calls=(),
            lines=(symbol.start_line, symbol.end_line),
            code_slice="",
            source_symbol=symbol,
        )
        for symbol in symbols
    )
    bridge = QueryBridgeResult(
        intent="modify",
        expanded_terms=["订单查询", "sql", "视图"],
        keywords=[],
        symbols=[],
        file_hints=[],
    )
    retriever = CursorRetriever(
        tmp_path,
        max_symbols=12,
        candidate_symbols=50,
    )

    result = retriever.score_candidates(
        ast_nodes=ast_nodes,
        candidates=candidates,
        bridge=bridge,
        graph=GraphBridgeResult(),
    )

    assert len(result.symbols) == 50
    assert any(symbol.file == "main.py" for symbol in result.symbols)
    assert any(symbol.name == "build_order_detail_sql" for symbol in result.symbols)


@pytest.mark.asyncio
async def test_retriever_rejects_raw_query_strings(tmp_path: Path) -> None:
    retriever = CursorRetriever(tmp_path)

    with pytest.raises(TypeError, match="rewritten search terms"):
        await retriever.retrieve("项目里有什么视图")


@pytest.mark.asyncio
async def test_retriever_enforces_global_timeout(tmp_path: Path) -> None:
    class SlowRetriever(CursorRetriever):
        async def _grep(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
            await asyncio.sleep(1)
            return ()

    retriever = SlowRetriever(
        tmp_path,
        total_timeout=0.01,
    )

    result = await retriever.retrieve(("needle",))

    assert result == RetrievalResult()


@pytest.mark.asyncio
async def test_retriever_falls_back_when_query_bridge_times_out(tmp_path: Path) -> None:
    class SlowLLM:
        async def chat(self, *args: object, **kwargs: object) -> object:
            await asyncio.sleep(1)
            raise AssertionError("query bridge timeout should cancel the LLM call")

    target = tmp_path / "src" / "views.py"
    target.parent.mkdir()
    target.write_text("class ProjectView:\n    pass\n", encoding="utf-8")
    bridge = CursorQueryBridge(SlowLLM(), timeout=0.01)
    rewritten = await bridge.generate("项目里有什么视图")
    retriever = CursorRetriever(tmp_path, total_timeout=1.0)
    result = await retriever.retrieve(rewritten.search_terms())

    assert result == RetrievalResult()


def test_query_bridge_allows_chinese_and_sorts_descending() -> None:
    bridge = CursorQueryBridge(llm=None)
    result = bridge.fallback("登机牌查询")
    # Check that Chinese tokens are kept in keywords/expanded_terms
    assert "登机牌" in result.keywords or "登机牌" in result.expanded_terms

    # Check that search_terms does not filter out Chinese
    terms = result.search_terms()
    assert any("登机牌" in term for term in terms)

    # Check that expanded terms are sorted by length descending, preferring "boarding_pass" (len 13)
    assert result.expanded_terms[0] == "boarding_pass"


def test_retriever_symbol_matches_subword() -> None:
    # Test _symbol_matches for exact and subword matching
    assert CursorRetriever._symbol_matches("make_boarding_pass_pdf", ("boarding_pass",))
    assert CursorRetriever._symbol_matches("make_boarding_pass_pdf", ("boarding",))
    assert CursorRetriever._symbol_matches("v_boarding_pass", ("boarding",))
    assert not CursorRetriever._symbol_matches("my_profile_page", ("boarding",))


@pytest.mark.asyncio
async def test_retriever_saturates_order_sql_constructor(tmp_path: Path) -> None:
    target = tmp_path / "service.py"
    target.write_text(
        "def build_order_detail_sql(where_clause):\n"
        "    return 'select * from ticket_order'\n",
        encoding="utf-8",
    )
    symbol = SimpleNamespace(
        file_path="service.py",
        name="build_order_detail_sql",
        kind="function",
        start_line=1,
        end_line=2,
    )
    service = MagicMock()
    service.map = SimpleNamespace(all_symbols=[symbol], symbols=[], file_scores={})

    retriever = CursorRetriever(tmp_path, repo_map_service=service)
    result = await retriever.retrieve(("ticket", "view"))

    assert tuple(sym.name for sym in result.symbols) == ("build_order_detail_sql",)
    assert result.symbols[0].score == 0.92
    assert result.symbols[0].reasons == ("universal_ast_recovery",)


@pytest.mark.asyncio
async def test_retriever_pre_ranking_and_ast_promotion(tmp_path: Path) -> None:
    # Write some files
    f1 = tmp_path / "app.py"
    f1.write_text("def make_boarding_pass_pdf(order):\n    pass\n", encoding="utf-8")
    f2 = tmp_path / "main.py"
    f2.write_text("def run_app():\n    pass\n", encoding="utf-8")
    f3 = tmp_path / "api.py"
    f3.write_text("def api_endpoint():\n    pass\n", encoding="utf-8")

    # Mock repo_map
    symbol = SimpleNamespace(
        file_path="app.py",
        name="make_boarding_pass_pdf",
        start_line=1,
        end_line=2,
    )
    service = MagicMock()
    service.map = SimpleNamespace(all_symbols=[symbol], symbols=[], file_scores={})

    retriever = CursorRetriever(tmp_path, repo_map_service=service, max_files=3)
    # Search with terms. The fallback retriever now saturates from universal AST symbols.
    result = await retriever.retrieve(("boarding", "api"))

    assert result.files == ("app.py",)
    assert tuple(symbol.name for symbol in result.symbols) == ("make_boarding_pass_pdf",)


@pytest.mark.asyncio
async def test_query_repomap_graph_ast_retriever_fusion_context_pipeline(
    tmp_path: Path,
) -> None:
    from src.agent.cursor_fusion import CursorFusionEngine
    from src.agent.cursor_graph_bridge import CursorGraphQueryBridge

    service_dir = tmp_path / "service"
    dao_dir = tmp_path / "dao"
    service_dir.mkdir()
    dao_dir.mkdir()
    (service_dir / "boarding.py").write_text(
        "def query_boarding_pass():\n"
        "    return fetch_ticket()\n",
        encoding="utf-8",
    )
    (dao_dir / "ticket.py").write_text(
        "def fetch_ticket():\n"
        "    return ticket_view_sql()\n"
        "\n"
        "def ticket_view_sql():\n"
        "    return 'select * from v_boarding_pass'\n",
        encoding="utf-8",
    )
    indexed = CtagsIndexResult(
        symbols=[
            CtagsSymbol(
                "service/boarding.py",
                "query_boarding_pass",
                "function",
                1,
                2,
                "def query_boarding_pass()",
            ),
            CtagsSymbol(
                "dao/ticket.py",
                "fetch_ticket",
                "function",
                1,
                2,
                "def fetch_ticket()",
            ),
            CtagsSymbol(
                "dao/ticket.py",
                "ticket_view_sql",
                "function",
                4,
                5,
                "def ticket_view_sql()",
            ),
        ],
        references=[
            ("service/boarding.py:query_boarding_pass:1", "fetch_ticket"),
            ("dao/ticket.py:fetch_ticket:1", "ticket_view_sql"),
        ],
        source="test",
    )
    repo_map = build_repo_map(tmp_path, indexed=indexed)
    service = SimpleNamespace(
        map=repo_map,
        wait_until_ready=lambda timeout=None: True,
    )
    bridge = QueryBridgeResult(
        intent="modify",
        domain="ticketing",
        concepts=["boarding pass", "query api", "view based query"],
        expanded_terms=["boarding_pass", "ticket_lookup", "sql_view"],
        constraints={"layer_hint": ["api", "service", "dao"], "exclude": ["auth unrelated"]},
        keywords=[],
        symbols=[],
        file_hints=[],
    )

    repo_lookup = CursorRepoMapLookup(service)
    candidates = repo_lookup.lookup(bridge)
    candidate_names = {candidate.symbol.name for candidate in candidates}
    assert {"query_boarding_pass", "ticket_view_sql"} <= candidate_names
    assert any("name_token_match" in candidate.reasons for candidate in candidates)

    graph = CursorGraphQueryBridge(service, depth=2, top_symbols=3)
    graph_result = await graph.expand_candidates(candidates)
    assert "fetch_ticket" in graph_result.expanded_symbols
    assert any("query_boarding_pass -> dao:fetch_ticket" in path for path in graph_result.paths)

    ast_candidates = repo_lookup.merge_by_ids(candidates, graph_result.expanded_symbol_ids)
    ast_nodes = CursorAstStructureLayer(tmp_path).ground(ast_candidates, limit=4)
    assert any(node.symbol == "query_boarding_pass" for node in ast_nodes)
    assert all(node.lines[0] <= node.lines[1] for node in ast_nodes)

    retriever = CursorRetriever(tmp_path, repo_map_service=service)
    retrieval = retriever.score_candidates(
        ast_nodes=ast_nodes,
        candidates=ast_candidates,
        bridge=bridge,
        graph=graph_result,
    )
    assert retrieval.symbols[0].name == "query_boarding_pass"
    assert retrieval.symbols[0].score > 0
    assert "fetch_ticket" in retrieval.symbols[0].calls

    fusion = CursorFusionEngine.decide(retrieval, bridge, max_files=4, max_symbols=4)
    assert fusion.final_context[0] == (
        "FOCUS:service/boarding.py:query_boarding_pass:1-2"
    )
    assert fusion.confidence > 0

    pack = CursorContextPackBuilder(tmp_path).build_context(
        fusion.retrieval,
        final_context=fusion.final_context,
    )
    assert "[LAYER_1_CORE_SYMBOL]" in pack.windows[0].content
    assert "1: def query_boarding_pass():" in pack.windows[0].content
    assert "[LAYER_3_GLOBAL_SKELETON]" in pack.windows[0].content


def test_context_builder_keeps_core_symbol_exact(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    # Create a file with 100 lines
    content = "\n".join(f"line_{i}" for i in range(1, 101))
    target.write_text(content, encoding="utf-8")

    retrieval = RetrievalResult(
        files=("app.py",),
        symbols=(RetrievalSymbol("app.py", "my_func", 50, 60),),
    )

    builder = CursorContextPackBuilder(tmp_path)
    pack = builder.build_context(retrieval)

    assert pack.windows[0].start_line == 50
    assert pack.windows[0].end_line == 60


def test_fusion_destructive_routes_top_two_as_exclusive_focus() -> None:
    from src.agent.cursor_fusion import CursorFusionEngine

    bridge = QueryBridgeResult(
        intent="debug",
        expanded_terms=[],
        keywords=[],
        symbols=[],
        file_hints=[],
    )
    raw = RetrievalResult(
        files=("a.py", "b.py"),
        symbols=(
            RetrievalSymbol("a.py", "alpha", 10, 12, score=0.9),
            RetrievalSymbol("b.py", "beta", 20, 22, score=0.8),
            RetrievalSymbol("c.py", "gamma", 30, 32, score=0.7),
        ),
    )

    decision = CursorFusionEngine.decide(
        raw,
        bridge,
        max_files=2,
        max_symbols=3,
        top_k=4,
    )

    assert decision.final_context[:3] == (
        "FOCUS:a.py:alpha:10-12",
        "FOCUS:b.py:beta:20-22",
        "c.py:gamma:30-32",
    )


def test_context_builder_uses_focus_highlight_and_semantic_windows(
    tmp_path: Path,
) -> None:
    lines = [f"# filler {index}" for index in range(1, 71)]
    lines[29] = "def focus():"
    lines[30] = "    return 'focus'"
    lines[49] = "def secondary():"
    lines[50] = "    secret_body = True"
    lines[51] = "    return secret_body"
    (tmp_path / "app.py").write_text("\n".join(lines), encoding="utf-8")
    retrieval = RetrievalResult(
        files=("app.py",),
        symbols=(
            RetrievalSymbol("app.py", "focus", 30, 31),
            RetrievalSymbol("app.py", "secondary", 50, 52),
        ),
    )

    pack = CursorContextPackBuilder(tmp_path).build_context(
        retrieval,
        final_context=(
            "FOCUS:app.py:focus:30-31",
            "app.py:secondary:50-52",
        ),
    )

    assert len(pack.windows) == 1
    focus_window = pack.windows[0]
    assert focus_window.start_line == 30
    assert focus_window.end_line == 52
    assert "[LAYER_1_CORE_SYMBOL]" in focus_window.content
    assert "31:     return 'focus'" in focus_window.content
    assert "secret_body" in focus_window.content
    assert "50: def secondary():" in focus_window.content
    assert focus_window.content.count("[LAYER_3_GLOBAL_SKELETON]") == 1
    assert focus_window.content.count("[LAYER_1_CORE_SYMBOL]") == 1
    assert focus_window.content.count("INTERVAL_CHUNK RANGE") == 2


def test_context_builder_merges_ast_neighborhood_ranges(tmp_path: Path) -> None:
    lines = [f"# filler {index}" for index in range(1, 151)]
    lines[19] = "def parent():"
    lines[20] = "    return child()"
    lines[79] = "def child():"
    lines[80] = "    return helper()"
    lines[119] = "def helper():"
    lines[120] = "    return 1"
    (tmp_path / "main.py").write_text("\n".join(lines), encoding="utf-8")
    retrieval = RetrievalResult(
        files=("main.py",),
        symbols=(
            RetrievalSymbol(
                "main.py",
                "child",
                80,
                81,
                score=0.9,
                calls=("helper",),
            ),
            RetrievalSymbol(
                "main.py",
                "parent",
                20,
                21,
                score=0.7,
                calls=("child",),
            ),
            RetrievalSymbol("main.py", "helper", 120, 121, score=0.6),
        ),
    )

    pack = CursorContextPackBuilder(tmp_path).build_context(
        retrieval,
        final_context=("FOCUS:main.py:child:80-81",),
    )

    window = pack.windows[0]
    assert window.start_line == 80
    assert window.end_line == 81
    assert "[LAYER_3_GLOBAL_SKELETON]" in window.content
    assert "20: def parent():" in window.content
    assert "120: def helper():" in window.content


def test_fusion_emits_schema_alignment_signal_for_query_and_view() -> None:
    from src.agent.cursor_fusion import CursorFusionEngine

    bridge = QueryBridgeResult(
        intent="modify",
        expanded_terms=[],
        keywords=[],
        symbols=[],
        file_hints=[],
    )
    raw = RetrievalResult(
        files=("service.py", "schema.sql"),
        symbols=(
            RetrievalSymbol(
                "service.py",
                "build_order_detail_sql:SELECT:ticket_order",
                10,
                30,
                score=0.95,
                kind="dml_select",
                tables_referenced=("ticket_order", "passenger_info"),
            ),
            RetrievalSymbol(
                "schema.sql",
                "view_ticket_report_detail",
                400,
                430,
                score=0.94,
                kind="ddl_view",
                tables_referenced=("ticket_order", "passenger_info"),
            ),
        ),
    )

    decision = CursorFusionEngine.decide(raw, bridge, max_files=2, max_symbols=2)

    assert decision.final_context[0].startswith("FOCUS:service.py:")
    assert not any(item.startswith("FOCUS:schema.sql:") for item in decision.final_context)


@pytest.mark.asyncio
async def test_fusion_reranker_prefers_backend_sql_over_frontend_noise() -> None:
    from src.agent.cursor_fusion import CursorFusionEngine

    class FakeReranker:
        async def rerank(
            self,
            query: str,
            documents: list[str],
            *,
            top_n: int,
        ) -> dict[int, float]:
            assert "把订单查询的 sql 改成使用视图查询" in query
            scores: dict[int, float] = {}
            for index, document in enumerate(documents):
                if "build_order_detail_sql" in document:
                    scores[index] = 0.99
                elif "view_ticket_report_detail" in document:
                    scores[index] = 0.96
                elif "render_order_card" in document:
                    scores[index] = 0.04
                else:
                    scores[index] = 0.20
            return scores

    bridge = QueryBridgeResult(
        intent="modify",
        expanded_terms=["订单查询", "sql", "视图"],
        keywords=[],
        symbols=[],
        file_hints=[],
    )
    raw = RetrievalResult(
        files=("app.py", "main.py", "db/init/init.sql"),
        symbols=(
            RetrievalSymbol("app.py", "render_order_card", 1, 20, score=0.99),
            RetrievalSymbol(
                "main.py",
                "build_order_detail_sql:SELECT:ticket_order",
                40,
                80,
                score=0.45,
                kind="dml_select",
                tables_referenced=("ticket_order",),
            ),
            RetrievalSymbol(
                "db/init/init.sql",
                "view_ticket_report_detail",
                120,
                150,
                score=0.42,
                kind="ddl_view",
                tables_referenced=("ticket_order",),
            ),
        ),
    )

    fusion = CursorFusionEngine(
        reranker=FakeReranker(),
        rerank_enabled=True,
        rerank_top_n=50,
    )
    decision = await fusion.decide_async(
        raw,
        bridge,
        max_files=3,
        max_symbols=3,
        user_intent="把订单查询的 sql 改成使用视图查询",
    )

    assert decision.retrieval.symbols[0].name.startswith("build_order_detail_sql")
    assert decision.retrieval.symbols[1].name == "view_ticket_report_detail"
    assert decision.retrieval.symbols[-1].name == "render_order_card"
    assert decision.final_context[0].startswith(
        "FOCUS:main.py:build_order_detail_sql"
    )


@pytest.mark.asyncio
async def test_fusion_slot_defense_keeps_sql_when_reranker_scores_noise_high() -> None:
    from src.agent.cursor_fusion import CursorFusionEngine

    class NoisyReranker:
        async def rerank(
            self,
            query: str,
            documents: list[str],
            *,
            top_n: int,
        ) -> dict[int, float]:
            return {
                index: (0.99 if "render_order_card" in document else 0.10)
                for index, document in enumerate(documents)
            }

    bridge = QueryBridgeResult(
        intent="modify",
        expanded_terms=["订单查询", "sql", "视图"],
        keywords=[],
        symbols=[],
        file_hints=[],
    )
    raw = RetrievalResult(
        files=("app.py", "main.py", "db/init/init.sql"),
        symbols=(
            RetrievalSymbol("app.py", "render_order_card", 1, 20, score=0.99),
            RetrievalSymbol("app.py", "render_order_page", 21, 35, score=0.98),
            RetrievalSymbol(
                "main.py",
                "build_order_detail_sql:SELECT:ticket_order",
                40,
                80,
                score=0.40,
                kind="dml_select",
                tables_referenced=("ticket_order",),
            ),
            RetrievalSymbol(
                "db/init/init.sql",
                "view_ticket_report_detail",
                120,
                150,
                score=0.39,
                kind="ddl_view",
                tables_referenced=("ticket_order",),
            ),
        ),
    )

    fusion = CursorFusionEngine(
        reranker=NoisyReranker(),
        rerank_enabled=True,
        rerank_top_n=50,
    )
    decision = await fusion.decide_async(
        raw,
        bridge,
        max_files=3,
        max_symbols=3,
        user_intent="把订单查询的 sql 改成使用视图查询",
    )

    selected = tuple(symbol.name for symbol in decision.retrieval.symbols)
    assert selected[0].startswith("build_order_detail_sql")
    assert "view_ticket_report_detail" in selected


@pytest.mark.asyncio
async def test_loop_logs_rerank_summary_and_phase_timing(tmp_path: Path) -> None:
    class FakeReranker:
        model = "Qwen/Qwen3-Reranker-8B"

        async def rerank(
            self,
            query: str,
            documents: list[str],
            *,
            top_n: int,
        ) -> dict[int, float]:
            return {
                index: (0.99 if "build_order_detail_sql" in document else 0.05)
                for index, document in enumerate(documents)
            }

    target = tmp_path / "main.py"
    target.write_text(
        "def build_order_detail_sql():\n    return 'select 1'\n",
        encoding="utf-8",
    )
    loop = _loop(tmp_path, cursor_reranker_enabled=False)
    loop.fusion.rerank_enabled = True
    loop.fusion.reranker = FakeReranker()
    loop.fusion.rerank_top_n = 50
    loop.query_bridge.generate_raw = AsyncMock(
        return_value=_bridge_json(QueryBridgeResult(
            intent="modify",
            expanded_terms=["订单查询", "sql", "视图"],
            keywords=[],
            symbols=[],
            file_hints=[],
        ))
    )
    loop.retriever.retrieve = AsyncMock(return_value=RetrievalResult(
        files=("main.py",),
        symbols=(RetrievalSymbol(
            "main.py",
            "build_order_detail_sql:SELECT:ticket_order",
            1,
            2,
            score=0.4,
            kind="dml_select",
            tables_referenced=("ticket_order",),
        ),),
    ))
    loop.llm.chat = AsyncMock(return_value=_decision_response(
        Decision(action="answer", answer="done")
    ))

    events = [event async for event in loop.run("把订单查询的 sql 改成使用视图查询")]
    messages = "\n".join(str(event.content) for event in events)
    summary = loop.get_probe_metrics()
    rerank_phase = next(
        item for item in summary["phases"] if item["phase"] == "cursor_rerank"
    )

    assert "[Reranker] applied model=Qwen/Qwen3-Reranker-8B" in messages
    assert "main.py:build_order_detail_sql" in messages
    assert rerank_phase["verdict"] == "applied"
    assert rerank_phase["scored"] > 0
    assert rerank_phase["rerank_duration_ms"] >= 0


def test_context_builder_injects_alignment_signal_into_exclusive_window(
    tmp_path: Path,
) -> None:
    schema = tmp_path / "schema.sql"
    schema.write_text(
        "CREATE VIEW view_ticket_report_detail AS\n"
        "SELECT p.real_name AS passenger_name\n"
        "FROM ticket_order o JOIN passenger_info p ON p.p_id = o.p_id;\n",
        encoding="utf-8",
    )
    retrieval = RetrievalResult(
        files=("schema.sql",),
        symbols=(
            RetrievalSymbol(
                "schema.sql",
                "view_ticket_report_detail",
                1,
                3,
                kind="ddl_view",
            ),
        ),
    )
    final_context = (
        "FOCUS:schema.sql:view_ticket_report_detail:1-3\n"
        "[SCHEMA_EVIDENCE]\n"
        "shared_tables: ['passenger_info', 'ticket_order']\n"
        "overlap_score: 1.00\n"
        "confidence: high\n"
        "reason: all referenced tables overlap"
    )

    pack = CursorContextPackBuilder(tmp_path).build_context(
        retrieval,
        final_context=(final_context,),
    )

    assert pack.windows[0].content.startswith("[LAYER_1_CORE_SYMBOL]")
    assert "CREATE VIEW view_ticket_report_detail" in pack.windows[0].content


def test_context_builder_caps_exclusive_sql_route(tmp_path: Path) -> None:
    schema = tmp_path / "schema.sql"
    lines = [f"-- filler {index}" for index in range(1, 201)]
    lines[99] = "CREATE TABLE important_orders (id integer primary key);"
    schema.write_text("\n".join(lines), encoding="utf-8")
    retrieval = RetrievalResult(
        files=("schema.sql",),
        symbols=(RetrievalSymbol("schema.sql", "schema.sql", 100, 100),),
    )

    pack = CursorContextPackBuilder(tmp_path).build_context(
        retrieval,
        final_context=("FOCUS:schema.sql:schema.sql:100-100",),
    )

    assert len(pack.windows[0].content) <= 12_000
    assert "[LAYER_1_CORE_SYMBOL]" in pack.windows[0].content
    assert "CREATE TABLE important_orders" in pack.windows[0].content


def test_decision_prompt_is_a_strict_local_action_classifier() -> None:
    from src.agent.cursor_decision import CursorDecisionLLM

    messages = CursorDecisionLLM(None).build_messages(
        state_text="state",
        context_pack=ContextPack(),
        hint=None,
    )

    system_text = messages[0]["content"]
    assert "strict local action classifier" in system_text
    assert "Use ONLY CURRENT_CONTEXT" in system_text
    assert "Do not propose refactors" in system_text
    assert "target file is absent" in system_text
    assert "reward" not in system_text.casefold()


def test_inter_llm_mismatched_quotes() -> None:
    from src.agent.cursor_inter_llm import CursorInterLLM
    malformed = (
        '{"intent":"查询项目中的视图","domains":["项目管理\'],'
        '"concepts":["项目","视图"],"ambiguity":false,"confidence":0.9}'
    )
    hint = CursorInterLLM.parse(malformed)
    assert hint is not None
    assert hint.intent == "查询项目中的视图"
    assert hint.domains == ("项目管理",)
    assert hint.concepts == ("项目", "视图")
    assert not hint.ambiguity
    assert hint.confidence == 0.9


@pytest.mark.asyncio
async def test_cursor_harness_transactional_sandbox_and_defenses(tmp_path: Path) -> None:
    from src.agent.cursor_harness_context import CursorHarnessContext
    from src.agent.cursor_executor import CursorExecutor
    from src.agent.cursor_decision import CursorDecisionLLM

    # 1. Test is_new_file rollback and clean deletion
    new_file = "new_file.py"
    async with CursorHarnessContext(tmp_path, new_file) as harness:
        # Create new file
        (tmp_path / new_file).write_text("print('hello')", encoding="utf-8")
        # Do not commit, triggering rollback on exit
    assert not (tmp_path / new_file).exists()

    # 2. Test copy-on-write backup rollback for existing files
    existing_file = "existing.py"
    original_content = "x = 1\n"
    (tmp_path / existing_file).write_text(original_content, encoding="utf-8")

    async with CursorHarnessContext(tmp_path, existing_file) as harness:
        (tmp_path / existing_file).write_text("x = 2\n", encoding="utf-8")
        # Do not commit, triggering rollback on exit
    assert (tmp_path / existing_file).read_text(encoding="utf-8") == original_content

    # 3. Test copy-on-write commit for existing files
    async with CursorHarnessContext(tmp_path, existing_file) as harness:
        (tmp_path / existing_file).write_text("x = 3\n", encoding="utf-8")
        harness.commit()
    assert (tmp_path / existing_file).read_text(encoding="utf-8") == "x = 3\n"

    # 4. Test error log truncation
    huge_error = "error\n" * 200
    truncated = CursorExecutor._truncate_error(huge_error)
    assert len(truncated) <= 850
    assert "... [TRUNCATED STACKTRACE] ..." in truncated

    # 5. Test robust suggested_completion casting in parser
    dec = CursorDecisionLLM(None)
    # Parse 40% percentage string
    payload1 = '{"action":"edit","answer":"","clarification":"","target_file":"sample.py","patch":"patch","suggested_completion":"40%"}'
    parsed1 = dec.parse(payload1, ("sample.py",))
    assert parsed1.suggested_completion == 0.4

    # Parse raw integer 50
    payload2 = '{"action":"edit","answer":"","clarification":"","target_file":"sample.py","patch":"patch","suggested_completion":50}'
    parsed2 = dec.parse(payload2, ("sample.py",))
    assert parsed2.suggested_completion == 0.5

    # Parse float 0.8
    payload3 = '{"action":"edit","answer":"","clarification":"","target_file":"sample.py","patch":"patch","suggested_completion":0.8}'
    parsed3 = dec.parse(payload3, ("sample.py",))
    assert parsed3.suggested_completion == 0.8

    # Parse malformed fallback
    payload4 = '{"action":"edit","answer":"","clarification":"","target_file":"sample.py","patch":"patch","suggested_completion":"invalid"}'
    parsed4 = dec.parse(payload4, ("sample.py",))
    assert parsed4.suggested_completion == 0.0


def test_state_manager_kanban_formatting_and_truncation() -> None:
    from src.harness.cursor.manager import CursorStateManager
    from src.agent.cursor_state import CursorState
    
    manager = CursorStateManager()
    
    # 1. Test basic Kanban state formatting
    state = CursorState(
        task="refactor auth service",
        current_file="src/auth.py",
        last_patch="<<<<<<< SEARCH\ndef login(username):\n=======\ndef login_user(username):\n>>>>>>> REPLACE",
        last_observation="Invalid credentials error",
        status="running",
        current_step=3,
        max_steps=10,
        stage_completion=0.6
    )
    formatted = manager.format_for_prompt(state)
    assert "### [KANBAN LOOP GLOBAL STATE] ###" in formatted
    assert "Requirement  : refactor auth service" in formatted
    assert "Active File  : src/auth.py" in formatted
    assert "Loop Progress: Step 3 / 10" in formatted
    assert "Completion   : 60%" in formatted
    assert "Prior Patch  : [Modified around -> def login(username):]" in formatted
    assert "--- LAST RUNTIME OBSERVATION ---" in formatted
    assert "Invalid credentials error" in formatted
    assert "#################################" in formatted

    # 2. Test patch summary fallback (no def/class/@app found)
    state_no_signature = CursorState(
        task="update database schema",
        current_file="schema.sql",
        last_patch="<<<<<<< SEARCH\nCREATE TABLE users;\n=======\nCREATE TABLE accounts;\n>>>>>>> REPLACE",
        last_observation="success",
        status="running",
        current_step=4,
        max_steps=5,
        stage_completion=0.8
    )
    formatted_no_sig = manager.format_for_prompt(state_no_signature)
    assert "Prior Patch  : [Modified around -> In-line diff adjustment]" in formatted_no_sig

    # 3. Test observation log truncation in formatting
    huge_obs = "Traceback line 1\n" + "Error occurred\n" * 50 + "Final failure line"
    state_huge_obs = CursorState(
        task="fix test failure",
        current_file="tests.py",
        last_patch="",
        last_observation=huge_obs,
        status="running",
        current_step=1,
        max_steps=5,
        stage_completion=0.0
    )
    formatted_huge_obs = manager.format_for_prompt(state_huge_obs)
    assert "[TRUNCATED SYSTEM LOGS]" in formatted_huge_obs
    assert len(formatted_huge_obs) <= 1000
    assert "Traceback line 1" in formatted_huge_obs
    assert "Final failure line" in formatted_huge_obs


def test_retriever_frequency_penalty(tmp_path: Path) -> None:
    from src.agent.cursor_retriever import CursorRetriever
    from src.agent.cursor_repo_map_lookup import CandidateSymbol
    from src.agent.cursor_ast_structure import AstNode
    from src.agent.cursor_query_bridge import QueryBridgeResult
    from src.indexer.repo_map import RankedSymbol, RepoMap

    query_sym = RankedSymbol(
        file_path="app.py",
        name="api_get",
        kind="function",
        start_line=10,
        end_line=13,
        signature="def api_get():",
        score=0.2,
        symbol_id="app.py:api_get:10",
    )
    other_query_syms = [
        RankedSymbol(
            file_path="app.py",
            name=f"api_get_{i}",
            kind="function",
            start_line=20 + i,
            end_line=23 + i,
            signature="def api_get():",
            score=0.01,
            symbol_id=f"app.py:api_get_{i}:{20+i}",
        )
        for i in range(10)
    ]
    rare_sym = RankedSymbol(
        file_path="app.py",
        name="make_boarding_pass_pdf",
        kind="function",
        start_line=50,
        end_line=53,
        signature="def make_boarding_pass_pdf():",
        score=0.2,
        symbol_id="app.py:make_boarding_pass_pdf:50",
    )

    symbols = [query_sym] + other_query_syms + [rare_sym]
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={"app.py": symbols},
        symbols_by_id={s.symbol_id: s for s in symbols},
        reference_edges=[],
    )

    class FakeService:
        def __init__(self, rm):
            self.map = rm

    retriever = CursorRetriever(tmp_path, repo_map_service=FakeService(repo_map))

    candidates = (
        CandidateSymbol(query_sym, 1.0),
        CandidateSymbol(rare_sym, 1.0),
    )
    ast_nodes = (
        AstNode(
            symbol="api_get",
            file="app.py",
            signature="def api_get()",
            calls=(),
            lines=(10, 13),
            code_slice="",
            source_symbol=query_sym,
        ),
        AstNode(
            symbol="make_boarding_pass_pdf",
            file="app.py",
            signature="def make_boarding_pass_pdf()",
            calls=(),
            lines=(50, 53),
            code_slice="",
            source_symbol=rare_sym,
        ),
    )

    bridge = QueryBridgeResult(
        intent="modify",
        expanded_terms=["api_get", "make_boarding_pass_pdf"],
        keywords=[],
        symbols=[],
        file_hints=[],
    )

    result = retriever.score_candidates(
        ast_nodes=ast_nodes,
        candidates=candidates,
        bridge=bridge,
        graph=GraphBridgeResult(),
    )

    scores = {sym.name: sym.score for sym in result.symbols}
    assert scores["make_boarding_pass_pdf"] > scores["api_get"]


def test_sql_parser_anchor_slicing(tmp_path: Path) -> None:
    from src.agent.cursor_sql_parser import UniversalSqlParser

    sql_lines = [f"/* line {i} */" for i in range(1, 50)]
    sql_lines.append("SELECT v_boarding_pass FROM tickets;")
    sql_lines.extend(f"/* line {i} */" for i in range(51, 101))

    sql_file = tmp_path / "init.sql"
    sql_file.write_text("\n".join(sql_lines), encoding="utf-8")

    sql_symbols = UniversalSqlParser().parse_text_block(
        sql_file.read_text(encoding="utf-8"),
        "init.sql",
        1,
    )

    assert len(sql_symbols) == 1
    sym = sql_symbols[0]
    assert sym.file_path == "init.sql"
    assert sym.name == "SELECT:tickets"
    assert sym.kind == "dml_select"
    assert sym.start_line == 50
    assert sym.end_line == 50
    assert sym.tables_referenced == ("tickets",)


def test_extension_aware_folding(tmp_path: Path) -> None:
    # Set up files
    py_file = tmp_path / "main.py"
    py_file.write_text("def top_1():\n    pass\n\ndef other_py():\n    x = 1\n    y = 2\n", encoding="utf-8")
    
    go_file = tmp_path / "helper.go"
    go_file.write_text("package main\n\nfunc otherGo() {\n\tprintln(1)\n\tprintln(2)\n}\n", encoding="utf-8")

    js_file = tmp_path / "script.js"
    js_file.write_text("function otherJs() {\n  const a = 1;\n  const b = 2;\n}\n", encoding="utf-8")

    # Symbols: top_1 is top-1, so it shouldn't fold. other_py, otherGo, otherJs should fold.
    retrieval = RetrievalResult(
        files=("main.py", "helper.go", "script.js"),
        symbols=(
            RetrievalSymbol("main.py", "top_1", 1, 2),
            RetrievalSymbol("main.py", "other_py", 4, 6),
            RetrievalSymbol("helper.go", "otherGo", 3, 6),
            RetrievalSymbol("script.js", "otherJs", 1, 4),
        ),
    )

    builder = CursorContextPackBuilder(tmp_path)
    pack = builder.build_context(retrieval)

    # Three-layer assembly keeps one exact core per candidate file.
    py_win = next(w for w in pack.windows if w.file == "main.py")
    go_win = next(w for w in pack.windows if w.file == "helper.go")
    js_win = next(w for w in pack.windows if w.file == "script.js")

    assert "[LAYER_1_CORE_SYMBOL]" in py_win.content
    assert "[LAYER_3_GLOBAL_SKELETON]" in go_win.content
    assert "[LAYER_3_GLOBAL_SKELETON]" in js_win.content


def test_dynamic_context_budget() -> None:
    builder = CursorContextPackBuilder(Path("/tmp"))
    
    # Default budget setting is 12000 from constructor
    assert builder.max_chars_per_file == 12000
    
    # "explain" -> 64KB
    builder.adjust_budget("explain")
    assert builder.max_chars_per_file == 64 * 1024
    
    # "modify" -> 16KB
    builder.adjust_budget("modify")
    assert builder.max_chars_per_file == 16 * 1024
    
    # others -> 16KB
    builder.adjust_budget("other_intent")
    assert builder.max_chars_per_file == 16 * 1024
    builder.adjust_budget(None)
    assert builder.max_chars_per_file == 16 * 1024


def test_patch_sql_fragments_escaped_quotes_and_comments() -> None:
    from src.agent.cursor_validator import _patch_sql_fragments

    # Test case 1: single quotes with comments and other python single quotes
    replace_1 = """
        # Let's run it
        res = conn.execute(
            text('SELECT COUNT(*) AS cnt FROM ticket_order WHERE name = \\'John\\'')
        )
        passenger.mappings()
    """
    fragments_1 = _patch_sql_fragments((("", replace_1),))
    assert len(fragments_1) == 1
    assert fragments_1[0] == "SELECT COUNT(*) AS cnt FROM ticket_order WHERE name = 'John'"

    # Test case 2: double quotes with method calls
    replace_2 = """
        res = conn.execute(
            text("SELECT COUNT(*) AS cnt FROM ticket_order ")
        )
        passenger.mappings()
    """
    fragments_2 = _patch_sql_fragments((("", replace_2),))
    assert len(fragments_2) == 1
    assert fragments_2[0] == "SELECT COUNT(*) AS cnt FROM ticket_order "

    # Test case 3: f-string SQL parts
    replace_3 = """
        query = f"SELECT * FROM passenger WHERE id = {passenger_id}"
    """
    fragments_3 = _patch_sql_fragments((("", replace_3),))
    assert len(fragments_3) == 1
    assert "SELECT * FROM passenger WHERE id = " in fragments_3[0]


def test_parse_context_route_nested_colons_and_enclosing_range(tmp_path: Path) -> None:
    from src.agent.cursor_context_pack_builder import _parse_context_route, _select_core_routes
    from src.agent.cursor_contracts import RetrievalSymbol
    import sys

    # Create dummy files
    service_dir = tmp_path / "service"
    service_dir.mkdir()
    (service_dir / "boarding.py").write_text(
        "def query_boarding_pass():\n"
        "    return fetch_ticket()\n",
        encoding="utf-8",
    )

    # Test route parsing with nested colons
    item = "FOCUS:service/boarding.py:query_boarding_pass:SELECT:ticket_order:1-2"
    route = _parse_context_route(item)
    assert route is not None
    assert route.file == "service/boarding.py"
    assert route.name == "query_boarding_pass:SELECT:ticket_order"
    assert route.start_line == 1
    assert route.end_line == 2

    # Test enclosing range promotion fallback using AST
    symbols = ()
    index = {}
    routes = (route,)
    core_routes = _select_core_routes(tmp_path, routes, index, symbols)
    assert len(core_routes) == 1
    promoted_route, enclosing_sym = core_routes[0]
    assert promoted_route.file == "service/boarding.py"
    assert promoted_route.name == "query_boarding_pass"
    assert promoted_route.start_line == 1
    assert promoted_route.end_line == 2


@pytest.mark.asyncio
async def test_clarify_escape_dynamic_seed_extraction() -> None:
    from src.agent.cursor_controller import ReactiveController
    from types import SimpleNamespace

    # Mock GraphEngine compile dependency subgraph
    called_seed = None
    async def mock_compile(seed, depth=2):
        nonlocal called_seed
        called_seed = seed
        return SimpleNamespace(retrieval=SimpleNamespace(symbols=[], files=[]))

    controller = ReactiveController(
        graph_engine=SimpleNamespace(compile_dependency_subgraph=mock_compile),
        context_builder=SimpleNamespace(merge_interval_subgraph=lambda cp, r: cp),
        state_manager=SimpleNamespace(
            observe_failure_signature=lambda s, **kw: s,
            mark_retry=lambda s, c: s,
            apply_time_decay=lambda s: s
        )
    )

    # Payload has clarification asking for archive_passenger
    payload = {"clarification": "Please provide the complete code for archive_passenger function."}
    state = SimpleNamespace(task="并发优化 archive_passenger 接口")

    updated_cp, next_state, handler_skip = await controller._handle_clarify_escape(
        payload=payload,
        severity=0.7,
        context_pack="mock_cp",
        state=state
    )

    # Inferred seed should be extracted from clarification text/task
    assert called_seed == "archive_passenger"
    assert handler_skip is True
