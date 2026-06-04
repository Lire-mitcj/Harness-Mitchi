from __future__ import annotations

import re
from enum import StrEnum


class Intent(StrEnum):
    """Classified intent of a user message."""

    CODING_TASK = "coding_task"
    QUESTION = "question"
    FILE_OPERATION = "file_operation"
    SHELL_COMMAND = "shell_command"
    CONVERSATION = "conversation"


_FILE_PATTERNS = re.compile(
    r"\b(create|edit|write|delete|rename|move|read|open|show)\b.*\b(file|module|class|function)\b",
    re.IGNORECASE,
)
_SHELL_PATTERNS = re.compile(
    r"\b(run|execute|install|build|test|deploy|start|stop|npm|pip|docker|git|make|cargo)\b",
    re.IGNORECASE,
)
_CODING_PATTERNS = re.compile(
    r"\b(implement|refactor|fix|add|remove|change|update|debug|optimize|migrate)\b",
    re.IGNORECASE,
)
_QUESTION_PATTERNS = re.compile(
    r"^(what|how|why|where|when|which|who|can you explain|tell me about|describe)\b",
    re.IGNORECASE,
)


class IntentRouter:
    """Lightweight intent classifier for routing user messages.

    Uses pattern matching to determine whether a message requires tool use
    or is a simple conversational exchange. This avoids burning an LLM call
    for trivial messages.
    """

    def __init__(self, custom_patterns: dict[Intent, re.Pattern[str]] | None = None) -> None:
        self._patterns: list[tuple[Intent, re.Pattern[str]]] = [
            (Intent.FILE_OPERATION, _FILE_PATTERNS),
            (Intent.SHELL_COMMAND, _SHELL_PATTERNS),
            (Intent.CODING_TASK, _CODING_PATTERNS),
            (Intent.QUESTION, _QUESTION_PATTERNS),
        ]
        if custom_patterns:
            for intent, pattern in custom_patterns.items():
                self._patterns.insert(0, (intent, pattern))

    def classify(self, message: str) -> Intent:
        """Determine the intent of a user message.

        Checks patterns in priority order: file_operation > shell_command >
        coding_task > question > conversation (fallback).
        """
        stripped = message.strip()
        if not stripped:
            return Intent.CONVERSATION

        for intent, pattern in self._patterns:
            if pattern.search(stripped):
                return intent

        if len(stripped.split()) <= 3 and not stripped.endswith("?"):
            return Intent.CONVERSATION

        if stripped.endswith("?"):
            return Intent.QUESTION

        return Intent.CONVERSATION

    def needs_tools(self, intent: Intent) -> bool:
        """Return True if the intent typically requires tool execution."""
        return intent in {
            Intent.CODING_TASK,
            Intent.FILE_OPERATION,
            Intent.SHELL_COMMAND,
        }
