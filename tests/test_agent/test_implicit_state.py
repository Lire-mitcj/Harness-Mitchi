from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agent.types import ToolResult
from src.state.decision_gravity import (
    calculate_retrieval_gravity,
    evaluate_search_intent,
    detect_structural_blind_spots,
    DecisionGravityController,
)


class DummySymbol:
    def __init__(self, file_path: str, name: str, start_line: int, score: float) -> None:
        self.file_path = file_path
        self.name = name
        self.start_line = start_line
        self.score = score


def test_calculate_retrieval_gravity() -> None:
    # High coverage, high step count -> should decay but stay at min 0.1
    g1 = calculate_retrieval_gravity(coverage_score=0.0, step_count=0)
    assert 0.9 <= g1 <= 1.0

    g2 = calculate_retrieval_gravity(coverage_score=0.8, step_count=5)
    assert g2 < g1

    g3 = calculate_retrieval_gravity(coverage_score=1.0, step_count=20)
    assert g3 == 0.1  # Max decay capped at 0.1


def test_evaluate_search_intent() -> None:
    # 1. High novelty, low uncertainty -> PROCEED_WITH_FULL_DATA
    history = []
    res1 = evaluate_search_intent(
        new_pattern="target_func",
        target_file="app.py",
        execution_history=history,
        has_compile_error=False,
    )
    assert res1["action"] == "PROCEED_WITH_FULL_DATA"
    assert res1["novelty"] == 1.0
    assert res1["uncertainty"] == 0.2

    # 2. Low novelty (repeat search), low uncertainty -> PROCEED_WITH_TRUNCATED_DATA
    history = [{"file": "app.py", "pattern": "target_func"}]
    res2 = evaluate_search_intent(
        new_pattern="target_func",
        target_file="app.py",
        execution_history=history,
        has_compile_error=False,
    )
    assert res2["action"] == "PROCEED_WITH_TRUNCATED_DATA"
    assert res2["novelty"] == 0.0
    assert res2["uncertainty"] == 0.2

    # 3. Low novelty (repeat search) but high uncertainty (error exists) -> PROCEED_WITH_FULL_DATA
    res3 = evaluate_search_intent(
        new_pattern="target_func",
        target_file="app.py",
        execution_history=history,
        has_compile_error=True,
    )
    assert res3["action"] == "PROCEED_WITH_FULL_DATA"
    assert res3["novelty"] == 0.0
    assert res3["uncertainty"] == 1.0


def test_detect_structural_blind_spots() -> None:
    # Use DummySymbol instead of MagicMock to prevent auto-mocking attribute issues
    mock_sym_a = DummySymbol(file_path="a.py", name="func_a", start_line=1, score=0.8)
    mock_sym_b = DummySymbol(file_path="b.py", name="func_b", start_line=10, score=0.6)
    mock_sym_c = DummySymbol(file_path="c.py", name="func_c", start_line=20, score=0.2)  # Score too low

    repo_map = MagicMock()
    repo_map.all_symbols = [mock_sym_a, mock_sym_b, mock_sym_c]
    repo_map.reference_edges = [
        ("a.py:func_a:1", "b.py:func_b:10"),
        ("a.py:func_a:1", "c.py:func_c:20"),
    ]

    # Viewed symbol A, B and C not viewed. B is score 0.6 (>0.4), C is score 0.2
    viewed = {"a.py:func_a:1"}
    spots = detect_structural_blind_spots(viewed, repo_map)

    assert len(spots) == 1
    assert spots[0]["name"] == "func_b"
    assert spots[0]["file"] == "b.py"
    assert spots[0]["score"] == 0.6


def test_decision_gravity_controller() -> None:
    controller = DecisionGravityController()
    
    # Mock context_store (AssembledState) and env_state (StateAssembledLoop)
    context_store = MagicMock()
    context_store.search_cache = {"coverage": {"a.py": 0.9, "b.py": 0.8}}
    context_store.run_state = MagicMock(step=10)
    context_store.context_anchors = MagicMock(code=[])

    env_state = MagicMock()
    env_state._last_novelty_value = 0.1
    env_state._validation_error.return_value = None  # Low uncertainty

    info = controller.coordinate_next_turn(context_store, env_state)
    assert info["strict_truncation"] is True
    assert "System entropy is low. Code evidence is saturated." in info["gravity_prompt"]


def test_grep_search_truncation_logic() -> None:
    # Verify that post-processing truncates search results to top 3
    loop = MagicMock()
    loop._validation_error.return_value = None  # Low uncertainty
    loop._grep_search_history = []
    
    # 5 matches in grep output
    matches = [
        {"file": "app.py", "symbol": "foo", "span": [1, 1], "match_line": "def foo()"},
        {"file": "app.py", "symbol": "bar", "span": [2, 2], "match_line": "def bar()"},
        {"file": "app.py", "symbol": "baz", "span": [3, 3], "match_line": "def baz()"},
        {"file": "app.py", "symbol": "qux", "span": [4, 4], "match_line": "def qux()"},
        {"file": "app.py", "symbol": "quux", "span": [5, 5], "match_line": "def quux()"},
    ]
    payload = {
        "matches": matches,
        "returned_matches": 5,
        "total_matches": 5,
        "truncated": False,
    }
    
    original_result = ToolResult(
        success=True,
        output=json.dumps(payload),
        metadata={"raw_evidence_store": matches, "match_count": 5},
    )

    # First search: novelty is 1.0 -> should not truncate
    from src.agent.state_assembled_loop import StateAssembledLoop
    processed1 = StateAssembledLoop._post_process_tool_result(
        loop,
        "grep_search",
        {"pattern": "target_func", "path": "app.py"},
        original_result,
    )
    assert processed1.success is True
    payload1 = json.loads(processed1.output)
    assert len(processed1.metadata["raw_evidence_store"]) == 5

    # Second search with same pattern -> novelty is 0.0 -> should truncate to 3 matches
    processed2 = StateAssembledLoop._post_process_tool_result(
        loop,
        "grep_search",
        {"pattern": "target_func", "path": "app.py"},
        original_result,
    )
    assert processed2.success is True
    # Parse payload
    output_lines = processed2.output.split("\n")
    # find where JSON ends (ends before the disclaimer string)
    json_part = "\n".join([line for line in output_lines if not line.startswith("[...")])
    payload2 = json.loads(json_part)
    
    assert payload2["returned_matches"] == 3
    assert len(payload2["matches"]) == 3
    assert payload2["truncated"] is True
    assert "matches hidden to reduce context noise" in processed2.output


import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_convergence_signals_inspect_tool_request_async() -> None:
    from src.hooks.before_tool import inspect_tool_request_async

    # 1. Lexical duplicate test (Blocked when has_compile_error is False)
    history = [{"file": "app.py", "pattern": "auth_me"}]
    args = {"pattern": "auth_me", "path": "app.py"}
    err = await inspect_tool_request_async(
        "grep_search",
        args,
        allowed_tools={"grep_search"},
        has_compile_error=False,
        search_history=history,
    )
    assert err is not None
    assert "BLOCK: Redundant search query" in err
    assert args["_lexical_similarity"] == 1.0
    assert args["_novelty_score"] == 0.0

    # 1b. Lexical duplicate test (Allowed when has_compile_error is True)
    args2 = {"pattern": "auth_me", "path": "app.py"}
    err2 = await inspect_tool_request_async(
        "grep_search",
        args2,
        allowed_tools={"grep_search"},
        has_compile_error=True,
        search_history=history,
    )
    assert err2 is None

    # 1c. Redundant grep allowed during edit recovery after a blocked edit
    err_recovery = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "auth_me", "path": "app.py"},
        allowed_tools={"grep_search"},
        has_compile_error=False,
        search_history=history,
        edit_recovery=True,
    )
    assert err_recovery is None

    # 2. Semantic overlap test (Blocked when has_compile_error is False)
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(side_effect=lambda x: [1.0, 0.0] if x == "auth_me" else [0.95, 0.05])
    history = [{"file": "app.py", "pattern": "auth_me"}]
    args3 = {"pattern": "decode_access_token", "path": "app.py"}
    err3 = await inspect_tool_request_async(
        "grep_search",
        args3,
        allowed_tools={"grep_search"},
        has_compile_error=False,
        search_history=history,
        embedder=mock_embedder,
    )
    assert err3 is not None
    assert "BLOCK: Redundant search query" in err3
    assert args3["_semantic_similarity"] > 0.85
    assert args3["_novelty_score"] < 0.15

    # 3. Structural connection test (Blocked under Symbol Dominance when similarity is high)
    repo_map = MagicMock()
    repo_map.all_symbols = []
    repo_map.reference_edges = [
        ("auth.py:func_auth:1", "token.py:func_token:10"),
    ]
    history = [{"file": "auth.py", "pattern": "login"}]
    args4 = {"pattern": "login", "path": "token.py"} # High Jaccard similarity (1.0) vs high connection
    err4 = await inspect_tool_request_async(
        "grep_search",
        args4,
        allowed_tools={"grep_search"},
        has_compile_error=False,
        search_history=history,
        repo_map=repo_map,
    )
    assert err4 is not None
    assert "BLOCK: Symbol dominance detected" in err4
    assert args4["_structural_connection"] == 1.0
    assert args4["_novelty_score"] == 0.0

    # 4. Gravity no longer gates retrieval: a novel, low-similarity search is
    #    allowed even when gravity is low. Tool-opening policy lives in
    #    reallocate_tools / the manifest, not in fact locking.
    args5 = {"pattern": "different_pattern", "path": "app.py"}
    mock_gravity_controller = MagicMock()
    mock_gravity_controller.last_gravity = 0.25
    err5 = await inspect_tool_request_async(
        "grep_search",
        args5,
        allowed_tools={"grep_search"},
        has_compile_error=False,
        search_history=history,
        gravity_controller=mock_gravity_controller,
    )
    assert err5 is None


@pytest.mark.asyncio
async def test_plan_lock_execution_mode() -> None:
    from src.hooks.before_tool import inspect_tool_request_async

    # 1. A checklist alone does not disable codebase_retrieve; RunState owns authorization.
    err = await inspect_tool_request_async(
        "codebase_retrieve",
        {"query": "find auth router"},
        allowed_tools={"codebase_retrieve"},
        checklist=["[ ] Fix auth", "[x] Read DB"],
    )
    assert err is None

    # 2. A checklist alone does not disable grep_search either.
    err = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "auth_me", "mode": "default"},
        allowed_tools={"grep_search"},
        checklist=["[ ] Fix auth"],
    )
    assert err is None

    # 3. grep_search symbol mode is allowed for missing symbols
    err = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "auth_me", "mode": "symbol"},
        allowed_tools={"grep_search"},
        checklist=["[ ] Fix auth"],
        context_anchors_code=[],
    )
    assert err is None

    # 4. grep_search symbol mode is blocked if symbol is already in context
    err = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "auth_me", "mode": "symbol"},
        allowed_tools={"grep_search"},
        checklist=["[ ] Fix auth"],
        context_anchors_code=[{
            "symbol": "auth_me",
            "file": "auth.py",
            "span": [1, 2],
            "code": "def auth_me():\n    return True",
        }],
    )
    assert err is not None
    assert err.startswith("BLOCK: Redundant search.")

    # 5. view_symbol_code is blocked if symbol is already in context
    err = await inspect_tool_request_async(
        "view_symbol_code",
        {"symbol": "auth_me", "target_file": "auth.py"},
        allowed_tools={"view_symbol_code"},
        checklist=["[ ] Fix auth"],
        context_anchors_code=[{
            "symbol": "auth_me",
            "file": "auth.py",
            "span": [1, 2],
            "code": "def auth_me():\n    return True",
        }],
    )
    assert err is not None
    assert err.startswith("BLOCK:")
    assert "already present" in err

    # 6. view_symbol_code is blocked if symbol is already loaded in previous step
    err = await inspect_tool_request_async(
        "view_symbol_code",
        {"symbol": "auth_me", "target_file": "auth.py"},
        allowed_tools={"view_symbol_code"},
        checklist=["[ ] Fix auth"],
        context_anchors_code=[],
        raw_evidence_store=[{
            "symbol": "auth_me",
            "file": "auth.py",
            "span": [1, 2],
            "code": "def auth_me():\n    return True",
        }],
    )
    assert err is not None
    assert err.startswith("BLOCK:")
    assert "already loaded previously" in err

    # 7. view_symbol_code is blocked globally even without a checklist
    err = await inspect_tool_request_async(
        "view_symbol_code",
        {"symbol": "auth_me", "target_file": "auth.py"},
        allowed_tools={"view_symbol_code"},
        checklist=[],
        context_anchors_code=[{
            "symbol": "auth_me",
            "file": "auth.py",
            "span": [1, 2],
            "code": "def auth_me():\n    return True",
        }],
    )
    assert err is not None
    assert err.startswith("BLOCK:")
    assert "already present" in err

    # Locator-only grep anchors do not prove that the symbol body was loaded.
    err = await inspect_tool_request_async(
        "view_symbol_code",
        {"symbol": "auth_me", "target_file": "auth.py"},
        allowed_tools={"view_symbol_code"},
        checklist=[],
        context_anchors_code=[{
            "symbol": "auth_me",
            "file": "auth.py",
            "span": [1, 1],
            "match_line": "def auth_me():",
        }],
    )
    assert err is None

    # 8. view_symbol_code is ALLOWED if the target file is in modified_files (verification bypass)
    err = await inspect_tool_request_async(
        "view_symbol_code",
        {"symbol": "auth_me", "target_file": "auth.py"},
        allowed_tools={"view_symbol_code"},
        checklist=[],
        context_anchors_code=[{"symbol": "auth_me", "file": "auth.py"}],
        modified_files=["auth.py"],
    )
    assert err is None


@pytest.mark.asyncio
async def test_decision_edit_preflight_validation() -> None:
    from src.hooks.before_tool import inspect_tool_request_async

    # 1. Missing target_file
    err = await inspect_tool_request_async(
        "decision_edit",
        {"intent": "fix bug"},
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: missing required 'target_file'" in err

    # 2. Missing intent
    err = await inspect_tool_request_async(
        "decision_edit",
        {"target_file": "list.py"},
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: missing required 'intent'" in err

    # 2a. Read-only intent is forwarded to DecisionLLM (harness does not block).
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "db/init/init.sql",
            "intent": "Read the full file to understand the schema",
            "context_window": [],
        },
        allowed_tools={"decision_edit", "view_symbol_code"},
    )
    assert err is None

    # 2b. An edit may mention inspection when it also specifies a mutation.
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "Inspect the handler and add validation",
            "context_window": [],
        },
        allowed_tools={"decision_edit"},
    )
    assert err is None

    # 3. Invalid focus_symbols type
    err = await inspect_tool_request_async(
        "decision_edit",
        {"target_file": "list.py", "intent": "fix", "focus_symbols": "not-a-list"},
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: 'focus_symbols' must be a JSON array" in err

    # 4. Invalid focus_symbols element type
    err = await inspect_tool_request_async(
        "decision_edit",
        {"target_file": "list.py", "intent": "fix", "focus_symbols": [123]},
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: 'focus_symbols' index 0 must be a string" in err

    # 5. Invalid context_window type
    err = await inspect_tool_request_async(
        "decision_edit",
        {"target_file": "list.py", "intent": "fix", "context_window": "not-a-list"},
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: 'context_window' must be a JSON array" in err

    # 6. Missing context_window file field
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "fix",
            "context_window": [{"span": [10, 20]}]
        },
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: 'context_window' index 0 is missing a valid 'file'" in err

    # 7. Invalid context_window span values
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "fix",
            "context_window": [{"file": "ref.py", "span": [10]}]
        },
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: 'context_window' index 0 'span' must be a list containing [start_line, end_line]" in err

    # 8. Start line greater than end line
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "fix",
            "context_window": [{"file": "ref.py", "span": [50, 20]}]
        },
        allowed_tools={"decision_edit"}
    )
    assert err is not None
    assert "Invalid decision_edit: 'context_window' index 0 'span' start_line 50 cannot be greater than end_line 20" in err

    # 9. Valid parameters pass successfully
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "fix",
            "focus_symbols": ["auth"],
            "context_window": [{"file": "ref.py", "span": [10, 20], "reason": "auth"}]
        },
        allowed_tools={"decision_edit"}
    )
    assert err is None


@pytest.mark.asyncio
async def test_verification_search_blocker() -> None:
    from src.hooks.before_tool import inspect_tool_request_async

    dummy_diff = (
        "--- a/list.py\n"
        "+++ b/list.py\n"
        "@@ -10,3 +10,3 @@\n"
        "-def old_passenger_method():\n"
        "+def archive_passenger(passenger_id):\n"
        "+    pen_orders = 0\n"
    )

    # 1. Allow view_symbol_code for archive_passenger (bypass verification gate for modified target file)
    err = await inspect_tool_request_async(
        "view_symbol_code",
        {"symbol": "archive_passenger", "target_file": "list.py"},
        allowed_tools={"view_symbol_code"},
        git_diff=dummy_diff,
        modified_files=["list.py"],
    )
    assert err is None

    # 2. Block grep_search targeting modified line pattern
    err = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "pen_orders", "path": "list.py"},
        allowed_tools={"grep_search"},
        git_diff=dummy_diff,
        modified_files=["list.py"],
    )
    assert err is not None
    assert "Verification search detected" in err
    assert "pen_orders" in err

    # 3. Allow grep_search on unmodified file or unrelated pattern
    err = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "other_pattern", "path": "list.py"},
        allowed_tools={"grep_search"},
        git_diff=dummy_diff,
        modified_files=["list.py"],
    )
    assert err is None

    err = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "pen_orders", "path": "main.py"},
        allowed_tools={"grep_search"},
        git_diff=dummy_diff,
        modified_files=["list.py"],
    )
    assert err is None


def test_retrieval_disabled_schema_pruning() -> None:
    from src.state.decision_gravity import DecisionGravityController

    controller = DecisionGravityController()
    # Initial state: healthy
    assert not controller.retrieval_disabled

    # Trigger override (low gravity)
    from unittest.mock import MagicMock
    context_mock = MagicMock()
    # Set step to a higher value to decay gravity
    context_mock.run_state.step = 10
    # Set high coverage
    context_mock.search_cache = {"coverage": {"file1.py": 1.0, "file2.py": 1.0}}

    env_mock = MagicMock()
    env_mock._validation_error.return_type = ""
    env_mock._last_novelty_value = 0.0

    res = controller.coordinate_next_turn(context_mock, env_mock)
    assert controller.retrieval_disabled is True
    assert "DECISION OVERRIDE LAYER" in res["gravity_prompt"]
