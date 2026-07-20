"""Tests for REPLACE-only → SEARCH/REPLACE materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.edit_materialize import (
    MaterializeError,
    materialize_edit_patch,
    parse_replace_sites,
)


def test_parse_replace_sites_with_symbol_header() -> None:
    patch = (
        "SITE: symbol=enabled\n"
        "<<<<<<< REPLACE\n"
        "def enabled():\n"
        "    return True\n"
        ">>>>>>> REPLACE"
    )
    sites = parse_replace_sites(patch)
    assert len(sites) == 1
    assert sites[0]["symbol"] == "enabled"
    assert sites[0]["body"].startswith("def enabled()")


def test_materialize_symbol_site_fills_search_from_disk(tmp_path: Path) -> None:
    target = "policy.py"
    (tmp_path / target).write_text(
        "def enabled():\n"
        "    return False\n"
        "\n"
        "def other():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    patch = (
        "SITE: symbol=enabled\n"
        "<<<<<<< REPLACE\n"
        "def enabled():\n"
        "    return True\n"
        ">>>>>>> REPLACE"
    )

    concrete = materialize_edit_patch(
        tmp_path,
        target,
        patch,
        focus_symbols=["enabled"],
    )

    assert "<<<<<<< SEARCH" in concrete
    assert "return False" in concrete
    assert "return True" in concrete
    assert concrete.count("<<<<<<< SEARCH") == 1


def test_materialize_uses_focus_symbol_when_site_bare(tmp_path: Path) -> None:
    target = "a.py"
    (tmp_path / target).write_text("def foo():\n    return 1\n", encoding="utf-8")
    patch = (
        "<<<<<<< REPLACE\n"
        "def foo():\n"
        "    return 2\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(
        tmp_path,
        target,
        patch,
        focus_symbols=["foo"],
    )
    assert "return 1" in concrete.split("=======")[0]
    assert "return 2" in concrete.split("=======")[1]


def test_materialize_span_site(tmp_path: Path) -> None:
    target = "a.py"
    (tmp_path / target).write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    patch = (
        "SITE: span=2-2\n"
        "<<<<<<< REPLACE\n"
        "b = 99\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(tmp_path, target, patch)
    assert "b = 2" in concrete
    assert "b = 99" in concrete


def test_materialize_missing_symbol_raises(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    patch = (
        "SITE: symbol=missing\n"
        "<<<<<<< REPLACE\n"
        "def missing():\n"
        "    pass\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(MaterializeError, match="E2_LOCATE"):
        materialize_edit_patch(tmp_path, "a.py", patch)


def test_materialize_leaves_legacy_search_replace_unchanged(tmp_path: Path) -> None:
    legacy = (
        "<<<<<<< SEARCH\n"
        "x = 1\n"
        "=======\n"
        "x = 2\n"
        ">>>>>>> REPLACE"
    )
    assert materialize_edit_patch(tmp_path, "a.py", legacy) == legacy


def test_materialize_rejects_noop_replace_with_after_edit_hint(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    patch = (
        "SITE: symbol=foo\n"
        "<<<<<<< REPLACE\n"
        "def foo():\n"
        "    return 1\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(MaterializeError, match="every SITE|AFTER-edit"):
        materialize_edit_patch(tmp_path, "a.py", patch, focus_symbols=["foo"])


def test_materialize_skips_noop_site_when_sibling_has_change(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "def foo():\n"
        "    return 1\n"
        "\n"
        "def bar():\n"
        "    return 2\n",
        encoding="utf-8",
    )
    patch = (
        "SITE: symbol=foo\n"
        "<<<<<<< REPLACE\n"
        "def foo():\n"
        "    return 99\n"
        ">>>>>>> REPLACE\n"
        "SITE: symbol=bar\n"
        "<<<<<<< REPLACE\n"
        "def bar():\n"
        "    return 2\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(
        tmp_path,
        "a.py",
        patch,
        focus_symbols=["foo", "bar"],
    )
    assert concrete.count("<<<<<<< SEARCH") == 1
    assert "return 99" in concrete
    assert "def bar()" not in concrete.split("=======")[0]


def test_materialize_missing_symbol_suggests_close_name(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "def load_noise_policy_from_path(path):\n"
        "    return path\n",
        encoding="utf-8",
    )
    patch = (
        "SITE: symbol=_load_noise_policy_from_dict\n"
        "<<<<<<< REPLACE\n"
        "def load_noise_policy_from_path(path):\n"
        "    return path\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(MaterializeError, match="Did you mean.*load_noise_policy_from_path"):
        materialize_edit_patch(tmp_path, "a.py", patch)


def test_sanitize_focus_symbols_keeps_remaps_and_drops(tmp_path: Path) -> None:
    from src.agent.edit_materialize import sanitize_focus_symbols

    (tmp_path / "a.py").write_text(
        "class NoisePolicy:\n"
        "    pass\n"
        "\n"
        "def strip_chat_noise(raw):\n"
        "    return raw\n"
        "\n"
        "def load_noise_policy_from_path(path):\n"
        "    return path\n",
        encoding="utf-8",
    )
    kept, dropped, remapped = sanitize_focus_symbols(
        tmp_path,
        "a.py",
        [
            "NoisePolicy",
            "_load_noise_policy_from_dict",
            "strip_chat_noise",
            "strip_text_mention",
            "is_bot_mentioned",
        ],
    )
    assert kept == ["NoisePolicy", "load_noise_policy_from_path", "strip_chat_noise"]
    assert remapped["_load_noise_policy_from_dict"] == "load_noise_policy_from_path"
    assert "strip_text_mention" in dropped
    assert "is_bot_mentioned" in dropped


def test_e2_error_omits_hallucinated_focus_symbols(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "def strip_chat_noise(raw):\n"
        "    return raw\n",
        encoding="utf-8",
    )
    patch = (
        "SITE: symbol=strip_text_mention\n"
        "<<<<<<< REPLACE\n"
        "def strip_text_mention(raw):\n"
        "    return raw\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(MaterializeError, match="valid_focus_symbols=\\[strip_chat_noise\\]") as exc:
        materialize_edit_patch(
            tmp_path,
            "a.py",
            patch,
            focus_symbols=["strip_chat_noise", "strip_text_mention", "is_bot_mentioned"],
        )
    assert "strip_text_mention" not in str(exc.value).split("valid_focus_symbols=")[1]
    assert "ignore invented" in str(exc.value)


def test_materialize_strips_context_line_prefixes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    patch = (
        "SITE: symbol=foo\n"
        "<<<<<<< REPLACE\n"
        "1: def foo():\n"
        "2:     return 2\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(tmp_path, "a.py", patch, focus_symbols=["foo"])
    assert "1: def foo()" not in concrete
    assert "return 2" in concrete.split("=======")[1]


def test_materialize_splits_legacy_equals_inside_replace(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    patch = (
        "SITE: symbol=foo\n"
        "<<<<<<< REPLACE\n"
        "def foo():\n"
        "    return 1\n"
        "=======\n"
        "def foo():\n"
        "    return 2\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(tmp_path, "a.py", patch, focus_symbols=["foo"])
    replace_half = concrete.split("=======")[1]
    assert "return 2" in replace_half
    assert "=======" not in replace_half.strip()


def test_materialize_surgical_hunk_for_partial_replace(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    patch = (
        "SITE: symbol=foo\n"
        "<<<<<<< REPLACE\n"
        "    b = 99\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(tmp_path, "a.py", patch, focus_symbols=["foo"])
    search_half, replace_half = concrete.split("=======")
    assert "b = 2" in search_half
    assert "def foo()" not in search_half
    assert "b = 99" in replace_half
    assert "return a + b" not in replace_half


def test_materialize_insert_after_delta(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "@dataclass(frozen=True)\n"
        "class NoisePolicy:\n"
        "    deictic_followup_patterns: tuple[str, ...]\n"
        "\n"
        "    def ok(self) -> bool:\n"
        "        return True\n",
        encoding="utf-8",
    )
    patch = (
        "SITE: symbol=NoisePolicy\n"
        "MODE: insert_after\n"
        "ANCHOR:     deictic_followup_patterns: tuple[str, ...]\n"
        "<<<<<<< REPLACE\n"
        "    bot_nicknames: frozenset[str]\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(tmp_path, "a.py", patch, focus_symbols=["NoisePolicy"])
    search_half, replace_half = concrete.split("=======")
    assert "deictic_followup_patterns: tuple[str, ...]" in search_half
    assert "bot_nicknames: frozenset[str]" in replace_half
    assert "class NoisePolicy" not in search_half


def test_materialize_replace_anchor_delta(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    patch = (
        "SITE: span=1-3\n"
        "MODE: replace_anchor\n"
        "ANCHOR: y = 2\n"
        "<<<<<<< REPLACE\n"
        "y = 99\n"
        ">>>>>>> REPLACE"
    )
    concrete = materialize_edit_patch(tmp_path, "a.py", patch)
    assert concrete.split("=======")[0].endswith("y = 2\n") or "y = 2" in concrete.split("=======")[0]
    assert "y = 99" in concrete.split("=======")[1]
