from __future__ import annotations

import logging
import traceback
from typing import Any

from src.agent.events import AgentEvent, error_event

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 529})

_MAX_ERROR_CONTEXT_LEN = 1500


class ErrorRecovery:
    """Centralized error handling and recovery for the agent loop.

    Formats errors into LLM-digestible context and determines whether
    a failed operation should be retried.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._retry_counts: dict[str, int] = {}

    def handle_tool_error(self, error: Exception, tool_name: str) -> str:
        """Format a tool execution error into context the LLM can learn from.

        Returns a string suitable for injecting into the conversation as a
        tool result message.
        """
        error_type = type(error).__name__
        error_msg = str(error)

        if len(error_msg) > _MAX_ERROR_CONTEXT_LEN:
            error_msg = error_msg[:_MAX_ERROR_CONTEXT_LEN] + "... [truncated]"

        tb = _safe_traceback(error)

        parts = [
            f"Tool '{tool_name}' failed with {error_type}: {error_msg}",
        ]

        if isinstance(error, FileNotFoundError):
            parts.append("Hint: verify the file path exists before retrying.")
        elif isinstance(error, PermissionError):
            parts.append("Hint: check file permissions or try a different approach.")
        elif isinstance(error, TimeoutError):
            parts.append("Hint: the operation took too long. Try a simpler approach.")
        elif isinstance(error, ValueError):
            parts.append("Hint: check the parameters you passed to the tool.")

        if tb:
            parts.append(f"Traceback (last 5 frames):\n{tb}")

        return "\n".join(parts)

    def handle_llm_error(self, error: Exception) -> AgentEvent:
        """Convert an LLM API error into a user-visible AgentEvent."""
        status = getattr(error, "status_code", None)
        error_type = type(error).__name__

        if status == 429:
            msg = "Rate limited by the LLM provider. Please wait a moment."
        elif status == 401:
            msg = "Authentication failed. Check your API key configuration."
        elif status == 400:
            msg = f"Bad request to LLM: {error}"
        elif status in (500, 502, 503):
            msg = "LLM provider is experiencing issues. Retrying may help."
        else:
            msg = f"LLM error ({error_type}): {error}"

        log.error("LLM error (status=%s): %s", status, error)
        return error_event(msg, {"error_type": error_type, "status_code": status})

    def should_retry(self, error: Exception) -> bool:
        """Determine whether the failed operation warrants a retry."""
        status = getattr(error, "status_code", None)
        if status in _RETRYABLE_STATUS_CODES:
            return True

        error_type = type(error).__name__
        count = self._retry_counts.get(error_type, 0)
        if count >= self.max_retries:
            return False

        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            self._retry_counts[error_type] = count + 1
            return True

        return False

    def record_retry(self, error: Exception) -> int:
        """Increment and return the retry count for this error type."""
        key = type(error).__name__
        self._retry_counts[key] = self._retry_counts.get(key, 0) + 1
        return self._retry_counts[key]

    def reset(self) -> None:
        self._retry_counts.clear()


def _safe_traceback(error: Exception, limit: int = 5) -> str:
    """Extract a short traceback string, suppressing failures."""
    try:
        frames = traceback.format_exception(type(error), error, error.__traceback__)
        lines = "".join(frames).strip().splitlines()
        if len(lines) > limit * 3:
            lines = lines[-(limit * 3):]
        return "\n".join(lines)
    except Exception:
        return ""
