from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.agent.types import Message
from src.executor.policy import resolve_executor_tools
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.harness.subtask.prompt_builder import count_messages_tokens, estimate_messages_tokens
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


@dataclass
class ExecutorContextConfig:
    root_task: str
    task_tree: TaskTree
    subtask: SubTaskNode
    project_root: Path
    policy: TruncationPolicy
    prior_summaries: dict[str, str] | None
    whitelist_files: list[str]
    whitelist_norm: frozenset[str]
    diag_handoff: bool
    compact_token_threshold: int


@dataclass
class ExecutorRuntimeState:
    paths_only_mode: bool
    use_paths_only: bool
    preloaded_paths: frozenset[str]
    truncated_paths: frozenset[str]
    active_runtime_tools: frozenset[str]
    explore_restricted: bool
    context_compacted: bool = False
    edit_read_fallback: bool = False


@dataclass
class ContextPipelineEvent:
    kind: str
    content: str


@dataclass
class ContextPipelineResult:
    messages: list[Message]
    runtime: ExecutorRuntimeState
    events: list[ContextPipelineEvent] = field(default_factory=list)
    changed: bool = False
    token_est: int | None = None


class ExecutorContextSession:
    """Harness-owned fold/compact for Executor message lists."""

    def __init__(
        self,
        *,
        config: ExecutorContextConfig,
        runtime: ExecutorRuntimeState,
        memory: ExploreSessionMemory,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.memory = memory
        self._prefix_len: int = 0
        self._prefix_tokens: int | None = None
        self._pinned_layers: list[Message] = []
        self._bootstrap_messages: list[Message] = []
        self._paths_only_scope_cache: Message | None = None
        self._paths_only_scope_key: frozenset[str] | None = None
        self._base_cache_key_stored: tuple | None = None
        self._base_messages_cache: list[Message] | None = None

    def seed_prefix(self, messages: list[Message]) -> None:
        """Pin S1/S2/S3 + bootstrap user from handoff; only the ReAct tail grows per turn."""
        if (
            len(messages) >= 3
            and messages[0].role == "system"
            and messages[1].role == "system"
            and messages[2].role == "system"
        ):
            self._pinned_layers = list(messages[:3])
            self._bootstrap_messages = list(messages[3:])
        else:
            self._pinned_layers = []
            self._bootstrap_messages = []
        self._prefix_len = len(messages)
        self._prefix_tokens = count_messages_tokens(messages)

    def _estimate_tokens(self, messages: list[Message]) -> int:
        return estimate_messages_tokens(
            messages,
            prefix_len=self._prefix_len,
            prefix_tokens=self._prefix_tokens,
        )

    def _reset_prefix(self, messages: list[Message]) -> None:
        self._prefix_len = len(messages)
        self._prefix_tokens = count_messages_tokens(messages)

    def estimate_tokens(self, messages: list[Message]) -> int:
        return self._estimate_tokens(messages)

    def should_compact(self, token_est: int) -> bool:
        return token_est > self.config.compact_token_threshold

    def _maybe_apply_paths_only(self) -> None:
        """Drop preload to paths-only list — skip for diagnose slice handoff (keep S3)."""
        if not self.config.diag_handoff:
            self._apply_paths_only()

    def prepare_before_llm(
        self,
        messages: list[Message],
        error_trace: list[str],
    ) -> ContextPipelineResult:
        """Digest fold/compact stage (runs before generic probe trim)."""
        token_est = self._estimate_tokens(messages)
        if not self.should_compact(token_est):
            return ContextPipelineResult(
                messages=messages,
                runtime=self.runtime,
                changed=False,
                token_est=token_est,
            )
        return self.compact_before_turn(messages, error_trace, token_est=token_est)

    def compact_before_turn(
        self,
        messages: list[Message],
        error_trace: list[str],
        *,
        token_est: int | None = None,
    ) -> ContextPipelineResult:
        if token_est is None:
            token_est = self.estimate_tokens(messages)
        self.memory.merge_digest_from_messages(messages)
        self._maybe_apply_paths_only()
        folded = self._rebuild_after_fold(
            error_trace,
            compact_reason="context size",
        )
        self.memory.reset_digest_scan(len(folded))
        self._reset_prefix(folded)
        return ContextPipelineResult(
            messages=folded,
            runtime=self.runtime,
            events=[
                ContextPipelineEvent(
                    kind="compact",
                    content=(
                        f"compressing context ({token_est} tok) — "
                        "session summary preserved"
                    ),
                )
            ],
            changed=True,
            token_est=self._prefix_tokens,
        )

    def after_tool_round(
        self,
        messages: list[Message],
        error_trace: list[str],
        *,
        explore_used: bool,
        explore_ok: bool,
    ) -> ContextPipelineResult:
        from src.executor.context_compress import merge_exploration_digests
        from src.executor.edit_guard import should_skip_explore_fold

        events: list[ContextPipelineEvent] = []
        out_messages = messages

        self.memory.merge_digest_from_messages(messages)

        if (
            explore_used
            and explore_ok
            and self.config.subtask.kind == SubTaskKind.EDIT
            and not should_skip_explore_fold(error_trace)
            and not self.config.diag_handoff
        ):
            new_digest = merge_exploration_digests(
                self.memory.running_digest,
                messages,
            )
            if new_digest.strip():
                self.memory.running_digest = new_digest
                self.memory.tracker.seed_from_digest(new_digest)
                self._apply_paths_only()
                folded = self._rebuild_after_fold(
                    error_trace,
                    compact_reason="after read/grep",
                )
                if folded != messages:
                    out_messages = folded
                    self.memory.reset_digest_scan(len(folded))
                    self._reset_prefix(folded)
                    events.append(
                        ContextPipelineEvent(
                            kind="fold",
                            content="folded read/grep/map into session summary",
                        )
                    )

        token_est = self._estimate_tokens(out_messages)
        if self.should_compact(token_est):
            self._maybe_apply_paths_only()
            out_messages = self._rebuild_after_fold(
                error_trace,
                compact_reason="context size after tools",
            )
            self.memory.reset_digest_scan(len(out_messages))
            self._reset_prefix(out_messages)
            events.append(
                ContextPipelineEvent(
                    kind="compact",
                    content=(
                        f"compressing after tools ({token_est} tok) — "
                        "session summary preserved"
                    ),
                )
            )
            token_est = self._prefix_tokens or token_est

        return ContextPipelineResult(
            messages=out_messages,
            runtime=self.runtime,
            events=events,
            changed=out_messages is not messages or bool(events),
            token_est=token_est,
        )

    def _rebuild_after_fold(
        self,
        error_trace: list[str],
        *,
        compact_reason: str,
    ) -> list[Message]:
        from src.executor.context_compress import rebuild_compacted_executor_messages

        return rebuild_compacted_executor_messages(
            base_messages=self._fold_base_messages(),
            digest=self.memory.running_digest,
            error_trace=error_trace,
            compact_reason=compact_reason,
        )

    def _fold_base_messages(self) -> list[Message]:
        """S1/S2 pinned from handoff; S3 swapped to paths_only when compacting; bootstrap kept."""
        if len(self._pinned_layers) >= 3:
            s1, s2 = self._pinned_layers[0], self._pinned_layers[1]
            use_paths_only_s3 = (
                self.runtime.paths_only_mode and not self.config.diag_handoff
            )
            s3 = (
                self._paths_only_scope_message()
                if use_paths_only_s3
                else self._pinned_layers[2]
            )
            return [s1, s2, s3, *self._bootstrap_messages]
        return self._base_messages()

    def _paths_only_scope_message(self) -> Message:
        tools_key = self.runtime.active_runtime_tools
        if (
            self._paths_only_scope_cache is not None
            and self._paths_only_scope_key == tools_key
        ):
            return self._paths_only_scope_cache

        from src.harness.subtask.prompt_builder import build_executor_scope_message

        msg = build_executor_scope_message(
            subtask=self.config.subtask,
            task_tree=self.config.task_tree,
            project_root=self.config.project_root,
            policy=self.config.policy,
            preload_mode="paths_only",
            runtime_tools=self.runtime.active_runtime_tools,
        )
        self._paths_only_scope_cache = msg
        self._paths_only_scope_key = tools_key
        return msg

    def _base_cache_key(self) -> tuple:
        return (
            self.runtime.paths_only_mode,
            self.runtime.active_runtime_tools,
            self.config.diag_handoff,
        )

    def _invalidate_base_cache(self) -> None:
        self._base_cache_key_stored = None
        self._base_messages_cache = None
        self._paths_only_scope_cache = None
        self._paths_only_scope_key = None

    def _base_messages(self) -> list[Message]:
        """Fallback when handoff layers were not seeded (tests / legacy paths)."""
        key = self._base_cache_key()
        if self._base_messages_cache is not None and self._base_cache_key_stored == key:
            return self._base_messages_cache

        from src.harness.subtask.prompt_builder import build_executor_messages

        preload_mode = "paths_only" if self.runtime.paths_only_mode else "full"
        msgs = build_executor_messages(
            root_task=self.config.root_task,
            task_tree=self.config.task_tree,
            subtask=self.config.subtask,
            project_root=self.config.project_root,
            policy=self.config.policy,
            prior_summaries=self.config.prior_summaries,
            preload_mode=preload_mode,
            runtime_tools=self.runtime.active_runtime_tools,
        )
        self._base_cache_key_stored = key
        self._base_messages_cache = msgs
        return msgs

    def _apply_paths_only(self) -> None:
        self.runtime.paths_only_mode = True
        self.runtime.use_paths_only = True
        self.runtime.preloaded_paths = frozenset()
        self.runtime.truncated_paths = self.config.whitelist_norm
        self.runtime.context_compacted = True
        if self.config.subtask.kind == SubTaskKind.EDIT and self.config.whitelist_files:
            self.runtime.active_runtime_tools = resolve_executor_tools(
                self.config.subtask,
                preloaded_paths=frozenset(),
                truncated_paths=self.runtime.truncated_paths,
                explore_restricted=self.runtime.explore_restricted,
                edit_read_fallback=self.runtime.edit_read_fallback,
            )
        self._invalidate_base_cache()
