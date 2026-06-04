from __future__ import annotations

"""Detect repeated or stagnant shell_exec loops within a subtask/turn."""


def normalize_shell_command(command: str) -> str:
    return " ".join(command.strip().split())


class ShellCommandTracker:
    """Track identical shell commands and failure streaks per executor scope."""

    def __init__(self, *, dedup_limit: int = 2, stagnant_limit: int = 3) -> None:
        self.dedup_limit = max(1, dedup_limit)
        self.stagnant_limit = max(1, stagnant_limit)
        self._run_counts: dict[str, int] = {}
        self._failure_streak: dict[str, int] = {}

    def check(self, command: str) -> str | None:
        key = normalize_shell_command(command)
        if not key:
            return "Empty shell command."

        runs = self._run_counts.get(key, 0)
        if runs >= self.dedup_limit:
            return (
                f"Blocked duplicate shell_exec: identical command already run "
                f"{runs} time(s) this subtask (limit {self.dedup_limit}). "
                "Try a different approach or summarize the blocker."
            )

        streak = self._failure_streak.get(key, 0)
        if streak >= self.stagnant_limit:
            return (
                f"Blocked stagnant shell loop: command failed {streak} consecutive "
                f"time(s) (limit {self.stagnant_limit}). Stop retrying and report "
                "the root cause in your final answer."
            )
        return None

    def record_run(self, command: str) -> None:
        key = normalize_shell_command(command)
        if key:
            self._run_counts[key] = self._run_counts.get(key, 0) + 1

    def record_outcome(self, command: str, *, success: bool) -> None:
        key = normalize_shell_command(command)
        if not key:
            return
        if success:
            self._failure_streak.pop(key, None)
            return
        self._failure_streak[key] = self._failure_streak.get(key, 0) + 1
