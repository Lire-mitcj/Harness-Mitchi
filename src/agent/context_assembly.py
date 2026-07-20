from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agent.context_compress import filter_summary_anchors_for_prompt
from src.agent.context_loader import ContextLoader
from src.agent.types import Message
from src.context.prompt_resources import load_internal_prompt

log = logging.getLogger(__name__)

# FILE CONTRACT is locator-strength only (hash); import dumps do not help edit decisions.
_FILE_CONTRACT_INCLUDE_IMPORTS = False
_MAX_FILE_CONTRACT_IMPORTS = 4
_MAX_FILE_FACTS = 6
_MAX_SUMMARY_CHARS = 900


def _norm_contract_file(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _file_contract_allowlist(
    active_files: list[str] | None,
    search_cache: dict[str, Any] | None,
) -> set[str]:
    """Files whose FILE CONTRACT / FACTS may appear in PROJECT RULES.

    Prefer active files + files that currently have loaded code anchors. Never
    dump every historically touched file's import list (main.py / grpc / …).
    """
    allowed: set[str] = set()
    for path in active_files or ():
        norm = _norm_contract_file(str(path))
        if norm:
            allowed.add(norm)
    if not search_cache:
        return allowed
    for item in search_cache.get("raw_evidence_store") or ():
        if isinstance(item, dict):
            norm = _norm_contract_file(str(item.get("file") or ""))
            if norm:
                allowed.add(norm)
    for item in search_cache.get("symbol_projections") or ():
        if isinstance(item, dict):
            norm = _norm_contract_file(str(item.get("file") or ""))
            if norm:
                allowed.add(norm)
    return allowed


def build_runtime_state_block(
    *,
    active_files: list[str],
    git_diff: str,
    validation_error: str | None,
    last_tool_result: str | None,
    last_error: dict[str, Any] | None,
    edit_plan_card: str = "",
) -> str:
    """Volatile per-turn state injected near the top of the user message.

    Checklist is no longer injected as control state — use ``edit_plan_card``.
    """
    git_diff_str = (
        f"```text\n{git_diff}\n```" if git_diff else "Clean working tree."
    )
    build_errors_str = (
        f"```\n{validation_error}\n```"
        if validation_error
        else "No compile or build errors."
    )
    last_error_str = json.dumps(last_error, ensure_ascii=False) if last_error else "None"
    plan_block = (edit_plan_card or "").strip() or "plan: empty (no frozen edit_plan)"
    return "\n".join([
        "### RUNTIME STATE (this turn)",
        f"- Active files: {', '.join(active_files) or 'none'}",
        "- Git working-tree state:",
        git_diff_str,
        "- Latest compile/build/test errors:",
        build_errors_str,
        f"- Last tool execution: {last_tool_result or 'No tools executed yet.'}",
        f"- Last structured error: {last_error_str}",
        "- edit_plan:",
        plan_block,
    ])


def build_loaded_code_anchor_block(context_block: str) -> str:
    context = context_block.strip()
    if not context:
        context = "No loaded code anchors in the current prompt."
    return (
        "### LOADED CODE ANCHORS (verified; reuse, do not reload) ###\n"
        "These are the grounded code/schema snippets available for the next action. "
        "Use them directly; a file listed elsewhere is only a locator unless it appears here "
        "or in STEP EVIDENCE as loaded.\n\n"
        f"{context}"
    )


def build_turn_context_block(
    *,
    loaded_anchors: str,
    execution_card_text: str,
) -> str:
    """Assemble per-turn context with recency-weighted ordering for the Core LLM.

    STEP EVIDENCE is placed after LOADED CODE ANCHORS so the decision card
    receives the strongest recency weight in the user message.
    """
    blocks: list[str] = []
    if loaded_anchors.strip():
        blocks.append(build_loaded_code_anchor_block(loaded_anchors))
    if execution_card_text.strip():
        blocks.append(execution_card_text)
    if not blocks:
        return ""
    return "\n\n" + "\n\n".join(blocks)


class ContextAssembly:
    """Assembles environment state, active files, and history into the coordinating LLM prompt."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.context_loader = ContextLoader(project_root)

    def load_system_prompt(self) -> str:
        return load_internal_prompt(
            "assembled_system_prompt.md",
            fallback="You are the lead coordinating agent for a coding task.",
        )

    def get_user_context(
        self,
        active_files: list[str],
        search_cache: dict[str, Any] | None = None,
    ) -> str:
        """Layer 3: hierarchical user instructions plus durable read summaries."""
        sections = []
        rules = self.context_loader.get_hierarchical_rules(active_files)
        if rules:
            sections.append(rules)
        file_contracts = search_cache.get("file_contracts") if search_cache else None
        if isinstance(file_contracts, dict) and file_contracts:
            allowed = _file_contract_allowlist(active_files, search_cache)
            blocks = []
            for file_path, contract in sorted(file_contracts.items()):
                norm = str(file_path).replace("\\", "/").lstrip("./")
                if allowed and norm not in allowed and str(file_path) not in allowed:
                    continue
                # Weak FILE CONTRACT: hash locator only (imports rarely help edits).
                line = f"- `{file_path}`@{contract.get('hash') or 'unknown'}"
                if _FILE_CONTRACT_INCLUDE_IMPORTS:
                    imports = list(contract.get("imports") or [])
                    shown = imports[:_MAX_FILE_CONTRACT_IMPORTS]
                    if shown:
                        line += " imports: " + ", ".join(f"`{value}`" for value in shown)
                        if len(imports) > _MAX_FILE_CONTRACT_IMPORTS:
                            line += f" (+{len(imports) - _MAX_FILE_CONTRACT_IMPORTS})"
                blocks.append(line)
            if blocks:
                sections.append(
                    "## 文件契约（hash only；正文见 LOADED ANCHORS）\n\n" + "\n".join(blocks)
                )
        file_facts = search_cache.get("file_facts") if search_cache else None
        if isinstance(file_facts, dict) and file_facts:
            allowed = _file_contract_allowlist(active_files, search_cache)
            blocks = []
            for file_path, facts in sorted(file_facts.items()):
                if not facts:
                    continue
                norm = str(file_path).replace("\\", "/").lstrip("./")
                if allowed and norm not in allowed and str(file_path) not in allowed:
                    continue
                blocks.append(
                    f"[FILE FACTS - {file_path}]\n"
                    + "\n".join(f"- {fact}" for fact in facts[:_MAX_FILE_FACTS])
                )
            if blocks:
                sections.append("## 文件事实\n\n" + "\n\n".join(blocks))
        schemas = search_cache.get("schema_contracts") if search_cache else None
        if isinstance(schemas, dict) and schemas:
            sections.append(
                "## Schema Contracts\n\n"
                + "\n".join(f"- `{key}`: {value}" for key, value in sorted(schemas.items()))
            )
        # Cold summaries only — never duplicate LOADED ANCHORS / CODE LOCATORS.
        summary_anchors = search_cache.get("summary_anchors") if search_cache else None
        if isinstance(summary_anchors, dict) and summary_anchors:
            filtered = filter_summary_anchors_for_prompt(
                summary_anchors,
                raw_evidence=list(search_cache.get("raw_evidence_store") or [])
                if search_cache
                else None,
                locators=list(search_cache.get("code_locators") or [])
                if search_cache
                else None,
            )
            if filtered:
                clipped: list[str] = []
                for value in filtered.values():
                    text = str(value)
                    if len(text) > _MAX_SUMMARY_CHARS:
                        text = text[: _MAX_SUMMARY_CHARS - 1].rstrip() + "…"
                    clipped.append(text)
                sections.append(
                    "## 已读代码摘要（冷；禁止与 LOADED/LOCATORS 同证据双通道）\n\n"
                    + "\n\n".join(clipped)
                )
        return "\n\n".join(sections)

    def build_context_block(
        self,
        active_files: list[str],
        search_cache: dict[str, Any] | None = None,
        modified_files: list[str] | None = None,
    ) -> str:
        blocks = []

        # 1. Inject projected retrieval evidence only. Durable artifacts remain in
        # search_cache for tools and are not exposed wholesale to the coordinator.
        retrieval_projection = None
        if search_cache:
            retrieval_projection = search_cache.get("context_projection")
        if retrieval_projection:
            blocks.append(
                "### RETRIEVAL CODE ANCHOR (current turn only) ###\n"
                f"{retrieval_projection}"
            )
        return "\n\n---\n\n".join(blocks)

    def assemble(
        self,
        *,
        user_query: str,
        active_files: list[str],
        git_diff: str,
        validation_error: str | None,
        messages_history: list[Message],
        search_cache: dict[str, Any] | None = None,
        last_tool_result: str | None = None,
        last_error: dict[str, Any] | None = None,
        modified_files: list[str] | None = None,
        edit_plan_card: str = "",
    ) -> list[dict[str, Any]]:
        system_base = self.load_system_prompt()

        rules_text = self.get_user_context(active_files, search_cache)
        rules_block = f"\n\n### PROJECT RULES & USER CONTEXT ###\n{rules_text}\n" if rules_text else ""

        context_block = self.build_context_block(
            active_files,
            search_cache=search_cache,
            modified_files=modified_files,
        )

        runtime_state_block = build_runtime_state_block(
            active_files=active_files,
            git_diff=git_diff,
            validation_error=validation_error,
            last_tool_result=last_tool_result,
            last_error=last_error,
            edit_plan_card=edit_plan_card,
        )

        context_text = (
            f"\n\n{build_loaded_code_anchor_block(context_block)}"
            if context_block.strip()
            else ""
        )
        user_instruction_block = (
            f"Original Request: {user_query}"
            f"{rules_block}\n\n{runtime_state_block}"
            f"{context_text}"
        )

        formatted_messages = []
        formatted_messages.append({"role": "system", "content": system_base})

        for msg in messages_history:
            formatted_messages.append(msg.to_dict())

        formatted_messages.append({"role": "user", "content": user_instruction_block})

        return formatted_messages
