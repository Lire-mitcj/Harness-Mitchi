from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agent.types import Message

log = logging.getLogger(__name__)


class ContextAssembly:
    """Assembles environment state, active files, and history into the coordinating LLM prompt."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._prompt_path = project_root / "prompts" / "assembled_system_prompt.md"

    def load_system_prompt(self) -> str:
        if self._prompt_path.exists():
            try:
                return self._prompt_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                log.warning("Failed to load assembled system prompt: %s", exc)
        return "You are the lead coordinating agent for a coding task."

    def build_context_block(self, active_files: list[str]) -> str:
        blocks = []
        for file in active_files:
            abs_path = (self.project_root / file).resolve()
            if not abs_path.exists():
                continue
            try:
                content = abs_path.read_text(encoding="utf-8")
                blocks.append(
                    f'<file path="{file}">\n{content}\n</file>'
                )
            except OSError as exc:
                log.warning("ContextAssembly: failed to read %s: %s", file, exc)
        return "\n\n".join(blocks)

    def assemble(
        self,
        *,
        user_query: str,
        active_files: list[str],
        checklist: list[str],
        git_diff: str,
        validation_error: str | None,
        messages_history: list[Message],
    ) -> list[dict[str, Any]]:
        system_base = self.load_system_prompt()

        project_rules_path = self.project_root / ".mitkii" / "rules.md"
        rules_text = ""
        if project_rules_path.exists():
            try:
                rules_text = f"\n\n<project_rules>\n{project_rules_path.read_text(encoding='utf-8').strip()}\n</project_rules>"
            except OSError:
                pass

        checklist_str = "\n".join(f"- {item}" for item in checklist) or "- No checklist items"
        context_block = self.build_context_block(active_files)

        # Check if new placeholders exist in system prompt
        has_new_slots = (
            "{{STATE.ACTIVE_FILES_LIST}}" in system_base or
            "{{STATE.CHECKLIST}}" in system_base or
            "{{STATE.GIT_DIFFS}}" in system_base or
            "{{STATE.BUILD_ERRORS}}" in system_base or
            "{{STATE.ACTIVE_FILES_BLOCKS}}" in system_base
        )

        if has_new_slots:
            active_files_list_str = ", ".join(active_files)
            git_diff_str = f"```diff\n{git_diff}\n```" if git_diff else "No unsaved changes."
            build_errors_str = f"```\n{validation_error}\n```" if validation_error else "No compile or build errors."
            active_files_blocks_str = context_block or "No active files in context yet."

            substituted = system_base
            substituted = substituted.replace("{{STATE.ACTIVE_FILES_LIST}}", active_files_list_str)
            substituted = substituted.replace("{{STATE.CHECKLIST}}", checklist_str)
            substituted = substituted.replace("{{STATE.GIT_DIFFS}}", git_diff_str)
            substituted = substituted.replace("{{STATE.BUILD_ERRORS}}", build_errors_str)
            substituted = substituted.replace("{{STATE.ACTIVE_FILES_BLOCKS}}", active_files_blocks_str)

            system_content = f"{substituted}{rules_text}"
            user_instruction_block = f"Original Request: {user_query}"
        else:
            # Fallback format if template doesn't contain placeholders
            system_content = f"{system_base}{rules_text}"

            state_parts = [
                "### CURRENT STATE ###",
                f"Active Checklist:\n{checklist_str}",
            ]
            if git_diff:
                state_parts.append(f"Git Diff:\n```diff\n{git_diff}\n```")
            if validation_error:
                state_parts.append(f"Validation/Compiler Failures:\n```\n{validation_error}\n```")

            state_text = "\n\n".join(state_parts)
            context_text = f"### CURRENT CONTEXT ###\n\n{context_block}" if context_block else "### CURRENT CONTEXT ###\n\nNo active files in context yet."

            user_instruction_block = (
                f"{state_text}\n\n"
                f"{context_text}\n\n"
                f"Original Request: {user_query}"
            )

        formatted_messages = []
        formatted_messages.append({"role": "system", "content": system_content})

        for msg in messages_history:
            formatted_messages.append(msg.to_dict())

        formatted_messages.append({"role": "user", "content": user_instruction_block})

        return formatted_messages
