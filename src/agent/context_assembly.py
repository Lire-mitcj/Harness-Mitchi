from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agent.context_loader import ContextLoader
from src.agent.types import Message
from src.context.prompt_resources import load_internal_prompt

log = logging.getLogger(__name__)


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
            blocks = []
            for file_path, contract in sorted(file_contracts.items()):
                imports = contract.get("imports") or []
                import_text = "\n".join(f"  - `{value}`" for value in imports) or "  - 无"
                blocks.append(
                    f"[FILE CONTRACT - {file_path}@{contract.get('hash') or 'unknown'}]\n"
                    f"- Imports:\n{import_text}"
                )
            sections.append("## 文件契约（每文件唯一）\n\n" + "\n\n".join(blocks))
        file_facts = search_cache.get("file_facts") if search_cache else None
        if isinstance(file_facts, dict) and file_facts:
            blocks = [
                f"[FILE FACTS - {file_path}]\n" + "\n".join(f"- {fact}" for fact in facts)
                for file_path, facts in sorted(file_facts.items()) if facts
            ]
            if blocks:
                sections.append("## 文件事实\n\n" + "\n\n".join(blocks))
        schemas = search_cache.get("schema_contracts") if search_cache else None
        if isinstance(schemas, dict) and schemas:
            sections.append(
                "## Schema Contracts\n\n"
                + "\n".join(f"- `{key}`: {value}" for key, value in sorted(schemas.items()))
            )
        summary_anchors = search_cache.get("summary_anchors") if search_cache else None
        if isinstance(summary_anchors, dict) and summary_anchors:
            sections.append(
                "## 已读代码摘要（禁止重复检索相同 file+span）\n\n"
                + "\n\n".join(str(value) for value in summary_anchors.values())
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
        checklist: list[str],
        git_diff: str,
        validation_error: str | None,
        messages_history: list[Message],
        search_cache: dict[str, Any] | None = None,
        last_tool_result: str | None = None,
        last_error: dict[str, Any] | None = None,
        modified_files: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        system_base = self.load_system_prompt()

        rules_text = self.get_user_context(active_files, search_cache)
        rules_block = f"\n\n### PROJECT RULES & USER CONTEXT ###\n{rules_text}\n" if rules_text else ""

        checklist_str = "\n".join(f"- {item}" for item in checklist) or "- No checklist items"
        context_block = self.build_context_block(active_files, search_cache=search_cache, modified_files=modified_files)

        # Check if new placeholders exist in system prompt
        has_new_slots = (
            "{{STATE.ACTIVE_FILES_LIST}}" in system_base or
            "{{STATE.CHECKLIST}}" in system_base or
            "{{STATE.GIT_DIFFS}}" in system_base or
            "{{STATE.BUILD_ERRORS}}" in system_base or
            "{{STATE.ACTIVE_FILES_BLOCKS}}" in system_base or
            "{{STATE.LAST_TOOL_RESULT}}" in system_base or
            "{{STATE.LAST_ERROR}}" in system_base
        )

        if has_new_slots:
            active_files_list_str = ", ".join(active_files)
            git_diff_str = (
                f"```text\n{git_diff}\n```" if git_diff else "Clean working tree."
            )
            build_errors_str = f"```\n{validation_error}\n```" if validation_error else "No compile or build errors."
            active_files_blocks_str = context_block or "No active files in context yet."
            last_tool_result_str = last_tool_result or "No tools executed yet."
            import json
            last_error_str = json.dumps(last_error, ensure_ascii=False) if last_error else "None"

            substituted = system_base
            substituted = substituted.replace("{{STATE.ACTIVE_FILES_LIST}}", active_files_list_str)
            substituted = substituted.replace("{{STATE.CHECKLIST}}", checklist_str)
            substituted = substituted.replace("{{STATE.GIT_DIFFS}}", git_diff_str)
            substituted = substituted.replace("{{STATE.BUILD_ERRORS}}", build_errors_str)
            substituted = substituted.replace("{{STATE.ACTIVE_FILES_BLOCKS}}", active_files_blocks_str)
            substituted = substituted.replace("{{STATE.LAST_TOOL_RESULT}}", last_tool_result_str)
            substituted = substituted.replace("{{STATE.LAST_ERROR}}", last_error_str)

            system_content = substituted
            user_instruction_block = f"{rules_block}\nOriginal Request: {user_query}" if rules_block else f"Original Request: {user_query}"
        else:
            # Fallback format if template doesn't contain placeholders
            system_content = system_base

            state_parts = [
                "### CURRENT STATE ###",
                f"Active Checklist:\n{checklist_str}",
            ]
            if last_tool_result:
                state_parts.append(f"Last Tool Result: {last_tool_result}")
            if last_error:
                import json
                state_parts.append(f"Last Error (Structured):\n```json\n{json.dumps(last_error, ensure_ascii=False, indent=2)}\n```")
            if git_diff:
                state_parts.append(f"Git Working Tree State:\n```text\n{git_diff}\n```")
            if validation_error:
                state_parts.append(f"Validation/Compiler Failures:\n```\n{validation_error}\n```")

            state_text = "\n\n".join(state_parts)
            context_text = f"### CURRENT CONTEXT ###\n\n{context_block}" if context_block else "### CURRENT CONTEXT ###\n\nNo active files in context yet."

            user_instruction_block = (
                f"{state_text}\n\n"
            )
            if rules_block:
                user_instruction_block += f"{rules_block}\n"
            user_instruction_block += (
                f"{context_text}\n\n"
                f"Original Request: {user_query}"
            )

        formatted_messages = []
        formatted_messages.append({"role": "system", "content": system_content})

        for msg in messages_history:
            formatted_messages.append(msg.to_dict())

        formatted_messages.append({"role": "user", "content": user_instruction_block})

        return formatted_messages
