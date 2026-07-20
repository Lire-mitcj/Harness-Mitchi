from __future__ import annotations

from dataclasses import replace

from src.agent.context_assembly import ContextAssembly
from src.agent.context_compress import (
    filter_summary_anchors_for_prompt,
    format_code_locators_block,
    hot_files_for_turn,
    microcompact_assistant_content,
    should_keep_verbatim,
)
from src.agent.run_state import start_run
from src.agent.state_assembled_loop import (
    AssembledState,
    ContextAnchors,
    _build_deduped_loaded_anchors_block,
    _demote_aged_anchors_to_locators,
    _search_cache_view,
)

def test_age_demotion_keeps_hot_strips_cold_to_locator() -> None:
    old = {
        "file": "old.py",
        "span": [1, 2],
        "code": "def old():\n    return 1",
        "symbol": "old",
    }
    hot = {
        "file": "hot.py",
        "span": [4, 5],
        "code": "def hot():\n    return 2",
        "symbol": "hot",
    }
    state = AssembledState(
        run_state=replace(start_run("", edit_mode=False), step=5),
        context_anchors=ContextAnchors(
            code=(old, hot),
            created_steps={"old.py:1-2": 1, "hot.py:4-5": 5},
            last_updated_step=5,
        ),
    )
    demoted = _demote_aged_anchors_to_locators(
        state,
        hot_files=hot_files_for_turn(active_files=["hot.py"]),
        hot_age=1,
        reason="age",
    )
    assert [item["file"] for item in demoted.context_anchors.code] == ["hot.py"]
    assert any(loc.get("id") == "old.py:1-2" for loc in demoted.context_anchors.locators)
    view = _search_cache_view(demoted)
    block = _build_deduped_loaded_anchors_block(view)
    assert "def hot" in block
    assert "def old" not in block
    assert "CODE LOCATORS" in block
    assert "old.py:1-2" in block


def test_summary_suppressed_when_loaded_or_locator(tmp_path) -> None:
    assembly = ContextAssembly(tmp_path)
    cache = {
        "raw_evidence_store": [
            {"file": "a.py", "span": [1, 2], "code": "def a(): pass", "symbol": "a"},
        ],
        "code_locators": [
            {"id": "b.py:3-4", "file": "b.py", "span": [3, 4], "symbol": "b"},
        ],
        "summary_anchors": {
            "a.py:1-2": "[CONTEXT COLLAPSE - a.py:1-2] should hide",
            "b.py:3-4": "[CONTEXT COLLAPSE - b.py:3-4] should hide",
            "c.py:1-1": "[CONTEXT COLLAPSE - c.py:1-1] cold keep",
        },
    }
    text = assembly.get_user_context(["a.py", "b.py", "c.py"], cache)
    assert "should hide" not in text
    assert "cold keep" in text


def test_filter_summary_anchors_helper() -> None:
    kept = filter_summary_anchors_for_prompt(
        {"x.py:1-2": "sum", "y.py:1-1": "keep"},
        raw_evidence=[{"file": "x.py", "span": [1, 2], "code": "pass"}],
        locators=[],
    )
    assert set(kept) == {"y.py:1-1"}


def test_microcompact_assistant_strips_let_me_look() -> None:
    long = (
        "Let me look at the codebase and examine how strip_chat_noise works in detail "
        "before I decide anything. " * 5
    )
    compacted = microcompact_assistant_content(long, has_tool_calls=True)
    assert len(compacted) < len(long)
    assert compacted.startswith("Let me look")


def test_should_keep_verbatim_hot_file() -> None:
    item = {"file": "a.py", "span": [1, 2], "code": "x"}
    assert should_keep_verbatim(
        item, current_step=10, created_step=1, hot_files={"a.py"}, hot_age=1
    )
    assert not should_keep_verbatim(
        item, current_step=10, created_step=1, hot_files=set(), hot_age=1
    )


def test_format_locators_block() -> None:
    block = format_code_locators_block(
        [{"id": "f.py:1-2", "file": "f.py", "span": [1, 2], "symbol": "foo", "step": 3}]
    )
    assert "CODE LOCATORS" in block
    assert "`foo`" in block


def test_file_contract_is_hash_only(tmp_path) -> None:
    assembly = ContextAssembly(tmp_path)
    text = assembly.get_user_context(
        ["main.py"],
        {
            "file_contracts": {
                "main.py": {
                    "hash": "abc123",
                    "imports": ["import os", "import sys", "from x import y"],
                }
            },
            "raw_evidence_store": [
                {"file": "main.py", "span": [1, 1], "code": "x = 1"},
            ],
        },
    )
    assert "`main.py`@abc123" in text
    assert "import os" not in text
