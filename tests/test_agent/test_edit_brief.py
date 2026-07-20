"""Tests for EditBrief plan-step helpers."""

from __future__ import annotations

from pathlib import Path

from src.agent.edit_brief import (
    format_applied_diff_summary,
    inspect_edit_brief_spec,
    intent_needs_split,
)


def _win(file: str = "a.py", span=(1, 10)) -> list[dict]:
    return [{"file": file, "span": [span[0], span[1]]}]


def test_intent_needs_split_multi_clause() -> None:
    fat = (
        "Add bot_nicknames field to NoisePolicy dataclass, update YAML loading, "
        "and add mention-related helper methods and also wire grpc_server callers"
    )
    assert intent_needs_split(fat)
    # Two related same-file changes are allowed (Edit ≤3 SITE / auto-follow-on).
    related = (
        "Update strip_chat_noise to only strip bot nicknames, and add a new "
        "function is_bot_mentioned with word boundary"
    )
    assert not intent_needs_split(related)
    assert not intent_needs_split("Add bot_nicknames field to NoisePolicy")


def test_inspect_requires_focus_symbols() -> None:
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": "Add one field",
            "context_window": _win(),
        },
    )
    assert err is not None
    assert "E4_SPEC" in err
    assert "focus_symbols" in err


def test_inspect_requires_context_window() -> None:
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": "Add one field",
            "focus_symbols": ["NoisePolicy"],
        },
    )
    assert err is not None
    assert "E4_SPEC" in err
    assert "context_window" in err


def test_tighten_oversized_context_to_symbol_span(tmp_path: Path) -> None:
    from src.agent.edit_brief import (
        expand_decision_edit_to_steps,
        tighten_decision_edit_context,
    )

    body = "\n".join(
        [
            "class NoisePolicy:",
            "    pass",
            "",
            *[f"# pad {i}" for i in range(230)],
            "def load_noise_policy_from_path(path):",
            "    return path",
            "",
            "def strip_chat_noise(raw):",
            "    return raw",
        ]
    )
    (tmp_path / "noise_policy.py").write_text(body + "\n", encoding="utf-8")
    args = {
        "target_file": "noise_policy.py",
        "intent": (
            "Add bot_nicknames to NoisePolicy, update load_noise_policy_from_path, "
            "and add strip helper"
        ),
        "focus_symbols": [
            "NoisePolicy",
            "load_noise_policy_from_path",
            "strip_chat_noise",
        ],
        "context_window": [{"file": "noise_policy.py", "span": [1, 250]}],
    }
    assert inspect_edit_brief_spec("decision_edit", args, project_root=tmp_path)
    steps = expand_decision_edit_to_steps(args, project_root=tmp_path)
    assert len(steps) == 3
    for step in steps:
        span = step["context_window"][0]["span"]
        assert span[1] - span[0] + 1 <= 220
        err = inspect_edit_brief_spec(
            "decision_edit", step, project_root=tmp_path
        )
        assert err is None, err

    tight = tighten_decision_edit_context(
        {
            "target_file": "noise_policy.py",
            "intent": "Add field",
            "focus_symbols": ["NoisePolicy"],
            "context_window": [{"file": "noise_policy.py", "span": [1, 250]}],
        },
        project_root=tmp_path,
    )
    assert tight["context_window"][0]["span"][1] < 50


def test_inspect_allows_multi_focus_overload_for_auto_split() -> None:
    """Multi-focus compound intents are expanded by harness — not E4."""
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": (
                "Add bot_nicknames field to NoisePolicy dataclass, update "
                "load_noise_policy_from_path to parse it from YAML, and add a "
                "strip_mentions helper"
            ),
            "focus_symbols": [
                "NoisePolicy",
                "load_noise_policy_from_path",
                "strip_chat_noise",
            ],
            "context_window": [
                {"file": "a.py", "span": [79, 146], "reason": "NoisePolicy — add field"},
                {
                    "file": "a.py",
                    "span": [158, 223],
                    "reason": "load_noise_policy_from_path — parse YAML",
                },
                {
                    "file": "a.py",
                    "span": [24, 33],
                    "reason": "strip_chat_noise — add mention stripping",
                },
            ],
        },
    )
    assert err is None


def test_inspect_blocks_huge_single_focus_intent() -> None:
    fat = "Add field and update loader and add helper and also " + ("x" * 400)
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": fat,
            "focus_symbols": ["NoisePolicy"],
            "context_window": _win(),
        },
    )
    assert err is not None
    assert "E4_SPEC" in err
    assert "too long" in err


def test_expand_decision_edit_by_focus_symbols() -> None:
    from src.agent.edit_brief import (
        derive_edit_queue_followons,
        expand_decision_edit_to_steps,
    )

    args = {
        "target_file": "noise_policy.py",
        "intent": (
            "Add bot_nicknames field to NoisePolicy dataclass, update "
            "load_noise_policy_from_path to parse it from YAML, and add a "
            "strip_mentions helper"
        ),
        "focus_symbols": [
            "NoisePolicy",
            "load_noise_policy_from_path",
            "strip_chat_noise",
        ],
        "context_window": [
            {
                "file": "noise_policy.py",
                "span": [79, 146],
                "reason": "NoisePolicy dataclass — add bot_nicknames field",
            },
            {
                "file": "noise_policy.py",
                "span": [158, 223],
                "reason": "load_noise_policy_from_path — parse bot_nicknames from YAML",
            },
            {
                "file": "noise_policy.py",
                "span": [24, 33],
                "reason": "strip_chat_noise — existing noise stripping",
            },
        ],
    }
    steps = expand_decision_edit_to_steps(args)
    assert len(steps) == 3
    assert steps[0]["focus_symbols"] == ["NoisePolicy"]
    assert "bot_nicknames" in steps[0]["intent"]
    assert steps[0]["context_window"][0]["span"] == [79, 146]
    assert steps[1]["focus_symbols"] == ["load_noise_policy_from_path"]
    assert steps[2]["focus_symbols"] == ["strip_chat_noise"]

    residual = derive_edit_queue_followons([args])
    assert len(residual) == 2
    assert residual[0]["focus_symbols"] == ["load_noise_policy_from_path"]


def test_split_intent_followons_and_derive_queue() -> None:
    from src.agent.edit_brief import derive_edit_queue_followons, split_intent_followons

    primary, followons = split_intent_followons(
        "Update strip_chat_noise to only strip bot nicknames, and add a new "
        "function is_bot_mentioned that checks mentions"
    )
    assert "strip_chat_noise" in primary
    assert followons
    assert "is_bot_mentioned" in followons[0]

    # Two-verb same-file change stays one EditLLM step — no residual queue.
    residual = derive_edit_queue_followons(
        [
            {
                "target_file": "noise_policy.py",
                "intent": (
                    "Update strip_chat_noise to only strip bot nicknames, "
                    "and add is_bot_mentioned helper"
                ),
                "focus_symbols": ["strip_chat_noise"],
                "context_window": _win("noise_policy.py", (24, 40)),
            }
        ]
    )
    assert residual == []


def test_inspect_blocks_too_many_focus(tmp_path: Path) -> None:
    names = [f"f{i}" for i in range(9)]
    body = "\n".join(f"def {n}():\n    pass" for n in names) + "\n"
    (tmp_path / "a.py").write_text(body, encoding="utf-8")
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": "touch nine symbols",
            "focus_symbols": names,
            "context_window": _win(),
        },
        project_root=tmp_path,
    )
    assert err is not None
    assert "focus_symbols" in err
    assert "8" in err

def test_inspect_blocks_all_hallucinated_focus(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "def strip_chat_noise(raw):\n    return raw\n",
        encoding="utf-8",
    )
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": "Add mention helpers",
            "focus_symbols": ["strip_text_mention", "is_bot_mentioned"],
            "context_window": _win(),
        },
        project_root=tmp_path,
    )
    assert err is not None
    assert "E4_SPEC" in err
    assert "none of focus_symbols" in err or "unknown focus_symbols" in err


def test_inspect_allows_remappable_typo(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "def load_noise_policy_from_path(path):\n    return path\n",
        encoding="utf-8",
    )
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": "Wire YAML field into loader",
            "focus_symbols": ["_load_noise_policy_from_dict"],
            "context_window": _win(),
        },
        project_root=tmp_path,
    )
    assert err is None


def test_inspect_rejects_huge_context_window(tmp_path: Path) -> None:
    # 250 real lines so EOF-clipping cannot shrink the span under the cap.
    body = "\n".join(f"line{i}" for i in range(1, 251)) + "\n"
    (tmp_path / "a.py").write_text(body, encoding="utf-8")
    err = inspect_edit_brief_spec(
        "decision_edit",
        {
            "target_file": "a.py",
            "intent": "Add one field",
            "focus_symbols": ["NoisePolicy"],
            "context_window": [{"file": "a.py", "span": [1, 250]}],
        },
        project_root=tmp_path,
    )
    assert err is not None
    assert "context_window" in err


def test_applied_diff_summary_counts() -> None:
    summary = format_applied_diff_summary(
        "a\nb\n",
        "a\nc\nd\n",
        target_file="x.py",
    )
    assert "applied_diff_summary:" in summary
    assert "+2" in summary or "+1" in summary
    assert "-1" in summary
    assert "x.py" in summary
