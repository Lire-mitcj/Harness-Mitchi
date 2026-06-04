from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.agent.explore_guard import (
    ExploreCommandTracker,
    normalize_grep_query,
    normalize_map_query,
    normalize_read_range,
    parse_read_path_with_lines,
)
from src.executor.exploration_digest import build_exploration_digest

from src.harness.gates.types import TruncationPolicy

_CACHE_HEADER = (
    "[Harness cached explore — same query already ran this subtask; "
    "use this output, then edit_file if ready]\n"
)


@dataclass
class ExploreSessionMemory:
    """Per-subtask explore cache, digest, and dedup tracker."""

    tracker: ExploreCommandTracker
    running_digest: str = ""
    policy: TruncationPolicy | None = None
    project_root: Path | None = None
    _outputs: dict[str, str] = field(default_factory=dict)
    _digest_scanned_len: int = 0
    tool_calls: int = 0
    cache_hits: int = 0

    @classmethod
    def create(
        cls,
        *,
        prior_exploration: str | None = None,
        policy: TruncationPolicy | None = None,
        project_root: Path | None = None,
    ) -> ExploreSessionMemory:
        tracker = ExploreCommandTracker()
        digest = (prior_exploration or "").strip()
        if digest:
            tracker.seed_from_digest(digest)
        return cls(
            tracker=tracker,
            running_digest=digest,
            policy=policy,
            project_root=project_root,
        )

    def explore_key(self, tool_name: str, args: dict) -> str | None:
        if tool_name == "grep_search":
            pattern = args.get("pattern") or args.get("query") or ""
            if not isinstance(pattern, str) or not pattern.strip():
                return None
            path_arg = args.get("path")
            include_arg = args.get("include") or args.get("glob")
            return normalize_grep_query(
                pattern,
                path=path_arg if isinstance(path_arg, str) else None,
                include=include_arg if isinstance(include_arg, str) else None,
            )
        if tool_name == "map_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return None
            return normalize_map_query(query)
        if tool_name in {"read_file", "read_files"}:
            if tool_name == "read_file":
                paths = [args.get("path")] if isinstance(args.get("path"), str) else []
            else:
                paths = [p for p in (args.get("paths") or []) if isinstance(p, str)]
            if not paths:
                return None
            start = args.get("start_line")
            end = args.get("end_line")
            parsed_path, ps, pe = parse_read_path_with_lines(str(paths[0]))
            if not isinstance(start, int) and ps is not None:
                start = ps
            if not isinstance(end, int) and pe is not None:
                end = pe
            return normalize_read_range(
                parsed_path.replace("\\", "/").lstrip("./"),
                start_line=start if isinstance(start, int) else None,
                end_line=end if isinstance(end, int) else None,
            )
        return None

    def get_output(self, key: str) -> str | None:
        return self._outputs.get(key)

    def put_output(self, key: str, output: str) -> None:
        if key and output:
            self._outputs[key] = output

    def format_cached(self, body: str) -> str:
        return _CACHE_HEADER + body

    def digest_fallback(self, key: str) -> str:
        if self.running_digest.strip():
            return (
                "Summary excerpt (full tool output was folded earlier):\n"
                + self.running_digest.strip()[:4000]
            )
        return f"(No cached body for {key}; see session exploration summary if present.)"

    def serve_read_from_preload(
        self,
        rel: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str | None:
        if self.project_root is None or self.policy is None:
            return None
        from src.executor.preload_read import format_cached_read_from_policy
        return format_cached_read_from_policy(
            self.project_root,
            rel,
            start_line=start_line,
            end_line=end_line,
            policy=self.policy,
        )

    def is_duplicate_explore(self, tool_name: str, args: dict) -> bool:
        if tool_name == "grep_search":
            pattern = args.get("pattern") or args.get("query") or ""
            if not isinstance(pattern, str):
                return False
            return self.tracker.check_grep(
                pattern,
                path=args.get("path") if isinstance(args.get("path"), str) else None,
                include=(
                    (args.get("include") or args.get("glob"))
                    if isinstance(args.get("include") or args.get("glob"), str)
                    else None
                ),
            ) is not None
        if tool_name == "map_search":
            query = args.get("query")
            if not isinstance(query, str):
                return False
            return self.tracker.check_map(query) is not None
        if tool_name in {"read_file", "read_files"}:
            if tool_name == "read_file":
                paths = [args.get("path")] if isinstance(args.get("path"), str) else []
            else:
                paths = [p for p in (args.get("paths") or []) if isinstance(p, str)]
            if not paths:
                return False
            start = args.get("start_line")
            end = args.get("end_line")
            parsed_path, ps, pe = parse_read_path_with_lines(str(paths[0]))
            if not isinstance(start, int) and ps is not None:
                start = ps
            if not isinstance(end, int) and pe is not None:
                end = pe
            rel = parsed_path.replace("\\", "/").lstrip("./")
            return self.tracker.check_read(
                rel,
                start_line=start if isinstance(start, int) else None,
                end_line=end if isinstance(end, int) else None,
            ) is not None
        return False

    def record_explore(self, tool_name: str, args: dict) -> None:
        if tool_name == "grep_search":
            pattern = args.get("pattern") or args.get("query") or ""
            if isinstance(pattern, str):
                self.tracker.record_grep(
                    pattern,
                    path=args.get("path") if isinstance(args.get("path"), str) else None,
                    include=(
                        (args.get("include") or args.get("glob"))
                        if isinstance(args.get("include") or args.get("glob"), str)
                        else None
                    ),
                )
        elif tool_name == "map_search":
            query = args.get("query")
            if isinstance(query, str) and query.strip():
                self.tracker.record_map(query)
        elif tool_name in {"read_file", "read_files"} and tool_name == "read_file":
            path_arg = args.get("path")
            if isinstance(path_arg, str):
                parsed_path, ps, pe = parse_read_path_with_lines(path_arg)
                start = args.get("start_line")
                end = args.get("end_line")
                if not isinstance(start, int) and ps is not None:
                    start = ps
                if not isinstance(end, int) and pe is not None:
                    end = pe
                rel = parsed_path.replace("\\", "/").lstrip("./")
                self.tracker.record_read(
                    rel,
                    start_line=start if isinstance(start, int) else None,
                    end_line=end if isinstance(end, int) else None,
                )

    def merge_digest_from_messages(self, messages: list) -> None:
        from src.executor.context_compress import merge_exploration_digests

        if len(messages) < self._digest_scanned_len:
            self._digest_scanned_len = 0
        if len(messages) <= self._digest_scanned_len:
            return
        delta = messages[self._digest_scanned_len :]
        self.running_digest = merge_exploration_digests(self.running_digest, delta)
        self._digest_scanned_len = len(messages)
        self.tracker.seed_from_digest(self.running_digest)

    def reset_digest_scan(self, message_count: int) -> None:
        """After fold/compact replaces the message list, avoid re-scanning merged history."""
        self._digest_scanned_len = max(0, message_count)

    def append_tool_digest(self, messages: list, *, max_chars: int = 10_000) -> None:
        fresh = build_exploration_digest(messages, max_chars=max_chars // 2)
        if not fresh:
            return
        if self.running_digest.strip():
            self.running_digest = f"{self.running_digest.strip()}\n\n{fresh}"
        else:
            self.running_digest = fresh
        if len(self.running_digest) > max_chars:
            self.running_digest = self.running_digest[: max_chars - 24] + "\n…[digest truncated]"
        self.tracker.seed_from_digest(self.running_digest)

    @staticmethod
    def truncate_output(output: str, *, max_chars: int = 14_000) -> str:
        if len(output) <= max_chars:
            return output
        return output[: max_chars - 36] + "\n…[harness truncated tool output]\n"
