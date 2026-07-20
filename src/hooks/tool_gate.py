"""Unified tool-gate policy for the assembled loop.

Owns the cross-cutting rules that used to be re-derived in
``reallocate_tools``, ``fact_locking``, ``tool_scorer``, and post-edit
invalidation:

* **invalidate scope** — which files become STALE after an edit
* **actionable stale** — when STALE may reopen retrieval
* **edit_ready closed** — prefer ``decision_edit`` only
* **view symbol normalize** — ``*`` / ``(full file)`` aliases

Hard allowlist waterfall stays in ``reallocate_tools.determine_allowed_tools``
but must consume these helpers instead of inventing parallel stale semantics.
Fact-lock applies exemptions; it must not invent when tools reopen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.agent.manifest import (
    StepManifest,
    Sufficiency,
    _norm_file,
    stale_needing_refresh,
)

# Config / docs must never blanket-invalidate unrelated code anchors.
_CONFIG_SUFFIXES = frozenset({
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".csv",
})

_FULL_FILE_SYMBOL_ALIASES = frozenset({
    "*",
    "all",
    "file",
    "(full file)",
    "full file",
    "full_file",
    "<full file>",
    "entire file",
    "whole file",
    "whole_file",
    "the whole file",
    "entire_file",
})


@dataclass(frozen=True, slots=True)
class FactLockPolicy:
    """Exemptions fact_locking should honor (gate-owned, not invented locally)."""

    allow_reread_files: frozenset[str] = frozenset()
    allow_reread_symbols: frozenset[str] = frozenset()
    suppress_grep_novelty: bool = False


def normalize_view_symbol(symbol: str) -> str:
    """Map LLM full-file aliases onto ``*`` (handled by view_symbol_code)."""
    text = str(symbol or "").strip()
    if not text:
        return text
    key = text.casefold().strip("\"'`")
    compact = key.replace(" ", "").replace("_", "")
    if (
        key in _FULL_FILE_SYMBOL_ALIASES
        or compact in {"(fullfile)", "fullfile", "entirefile", "wholefile"}
    ):
        return "*"
    return text


def files_to_invalidate(
    edited_file: str,
    manifest: StepManifest | None = None,
) -> frozenset[str]:
    """Files whose anchors become STALE after a successful decision_edit.

    Only the edited file is invalidated. Cross-file partners are **not**
    auto-staled — that used to mark every observed py file STALE after a yaml
    edit and reopen VIEW while ``edit_ready: yes``. Partner refresh is driven by
    wiring_gap / verification mode, not blanket partner STALE.
    """
    _ = manifest  # reserved for future explicit dependency edges
    edited = _norm_file(edited_file)
    if not edited:
        return frozenset()
    return frozenset({edited})


def is_config_path(path: str) -> bool:
    suffix = Path(_norm_file(path)).suffix.casefold()
    return suffix in _CONFIG_SUFFIXES


def edited_files_set(run_state: Any) -> frozenset[str]:
    changes = getattr(run_state, "changes", None)
    files = getattr(changes, "files", ()) if changes is not None else ()
    return frozenset(_norm_file(path) for path in files if path)


def self_stale_only(run_state: Any, manifest: StepManifest | None) -> bool:
    """True when every stale-needing-refresh item lives on an already-edited file."""
    if manifest is None:
        return False
    stale = stale_needing_refresh(manifest)
    if not stale:
        return False
    edited = edited_files_set(run_state)
    if not edited:
        return False
    return all(
        item.file and _norm_file(item.file) in edited
        for item in stale
    )


def actionable_stale(
    run_state: Any,
    manifest: StepManifest | None,
    *,
    sufficiency: str | None = None,
    has_missing: bool = False,
) -> tuple[Any, ...]:
    """STALE items that may reopen retrieval.

    When evidence is already ``edit_ready`` (sufficient + no missing), STALE is
    **advisory** — Core should pass spans via ``context_window``, not re-open
    VIEW. Self-stale on edited files is never actionable mid edit_burst.
    """
    if manifest is None:
        return ()
    stale = stale_needing_refresh(manifest)
    if not stale:
        return ()
    if self_stale_only(run_state, manifest):
        return ()
    suf = sufficiency or str(getattr(manifest, "sufficiency", Sufficiency.INSUFFICIENT))
    if (
        suf
        in {
            Sufficiency.SUFFICIENT_FOR_EDIT,
            Sufficiency.SUFFICIENT_FOR_VERIFY,
        }
        and not has_missing
    ):
        return ()
    return tuple(stale)


def has_actionable_stale(
    run_state: Any,
    manifest: StepManifest | None,
    *,
    sufficiency: str | None = None,
    has_missing: bool = False,
) -> bool:
    return bool(
        actionable_stale(
            run_state,
            manifest,
            sufficiency=sufficiency,
            has_missing=has_missing,
        )
    )


def edit_ready_closed(
    *,
    task_mode: str,
    sufficiency: str,
    has_missing: bool,
    has_actionable_stale_flag: bool,
    blocks_edit: bool,
) -> bool:
    """True when hard gate should be ``decision_edit`` only."""
    if task_mode != "edit":
        return False
    if blocks_edit or has_missing or has_actionable_stale_flag:
        return False
    return sufficiency in {
        Sufficiency.SUFFICIENT_FOR_EDIT,
        Sufficiency.SUFFICIENT_FOR_VERIFY,
    }


def fact_lock_policy_for(
    run_state: Any,
    *,
    modified_files: Iterable[str] | None = None,
) -> FactLockPolicy:
    """Build fact-lock exemptions from gate taxonomy (not raw ``has_stale``)."""
    manifest = getattr(run_state, "manifest", None)
    has_missing = bool(getattr(manifest, "missing_items", ())) if manifest else False
    sufficiency = (
        str(getattr(manifest, "sufficiency", Sufficiency.INSUFFICIENT))
        if manifest is not None
        else Sufficiency.INSUFFICIENT
    )
    stale_items = actionable_stale(
        run_state,
        manifest,
        sufficiency=sufficiency,
        has_missing=has_missing,
    )
    allow_files = {
        _norm_file(item.file) for item in stale_items if getattr(item, "file", None)
    }
    allow_symbols = {
        str(item.symbol).strip()
        for item in stale_items
        if getattr(item, "symbol", None)
    }
    for path in modified_files or ():
        norm = _norm_file(path)
        if norm:
            allow_files.add(norm)
    edited = edited_files_set(run_state)
    allow_files |= set(edited)
    return FactLockPolicy(
        allow_reread_files=frozenset(allow_files),
        allow_reread_symbols=frozenset(s for s in allow_symbols if s),
        suppress_grep_novelty=bool(edited) or bool(stale_items),
    )


def scorer_should_boost_view_for_stale(ctx_like: Any) -> bool:
    """Scorer: do not push VIEW on stale when already edit_ready."""
    sufficiency = str(getattr(ctx_like, "sufficiency", "") or "")
    has_missing = bool(getattr(ctx_like, "has_missing", False))
    if (
        sufficiency
        in {
            Sufficiency.SUFFICIENT_FOR_EDIT,
            Sufficiency.SUFFICIENT_FOR_VERIFY,
        }
        and not has_missing
    ):
        return False
    return float(getattr(ctx_like, "stale_ratio", 0.0) or 0.0) > 0.0
