from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

_ENV_KEYS_TO_SANITIZE = (
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "MITKII_MODEL",
    "MITKII_PLANNER_MODEL",
    "MITKII_JUDGE_MODEL",
    "ANTHROPIC_API_KEY",
)


def _sanitize_env_values() -> None:
    """Strip Windows CR characters accidentally exported into the shell."""
    for key in _ENV_KEYS_TO_SANITIZE:
        value = os.environ.get(key)
        if value and "\r" in value:
            os.environ[key] = value.replace("\r", "")

class ModelConfig(BaseSettings):
    """Configuration for a single LLM backend."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 8192
    api_base: str | None = None

    model_config = {"env_prefix": "MITKII_MODEL_", "extra": "ignore"}


class MitKIISettings(BaseSettings):
    """Top-level application settings.

    Values are loaded from environment variables prefixed with ``MITKII_``
    (e.g. ``MITKII_MODEL``, ``MITKII_MAX_TURNS``).  A ``.env`` file in the
    working directory is also read automatically.
    """

    # --- LLM ------------------------------------------------------------------
    model: str = "claude-sonnet-4-20250514"
    planner_model: str | None = Field(
        default=None,
        description=(
            "Planner-only model (TaskTree JSON). Defaults to scout_model when unset. "
            "Executor uses `model` (MITKII_MODEL)."
        ),
    )
    planner_max_tokens: int = Field(
        default=1536,
        ge=256,
        le=8192,
        description="Max completion tokens for Planner (TaskTree JSON; 1536 for multi-step plans).",
    )
    planner_json_mode: bool = Field(
        default=True,
        description="Request JSON object response_format from Planner LLM when supported.",
    )
    planner_trace: bool = Field(
        default=False,
        description="If true, Planner emits <planning_trace> before JSON (slower).",
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"

    # --- Context window -------------------------------------------------------
    max_context_tokens: int = 128_000
    context_budget_ratio: float = Field(
        default=0.75,
        ge=0.1,
        le=1.0,
        description="Fraction of max_context_tokens reserved for retrieval context.",
    )
    executor_compact_context_ratio: float = Field(
        default=0.70,
        ge=0.5,
        le=0.95,
        description=(
            "Compact executor message history when estimated tokens exceed this "
            "fraction of max_context_tokens (LLM context window)."
        ),
    )

    # --- Agent behaviour ------------------------------------------------------
    max_turns: int = Field(default=50, ge=1)
    max_retries: int = Field(default=3, ge=0)
    auto_approve_edits: bool = False
    edit_splice_enabled: bool = Field(
        default=True,
        description=(
            "Use Harness splice (replace_symbol) for edit subtasks with diagnose handoff "
            "instead of LLM-driven edit_file string matching."
        ),
    )
    edit_splice_max_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Anchor verify + symbol re-resolve attempts before replace_symbol fails.",
    )

    # --- Orchestrator (Planner-driven ReAct) ----------------------------------
    orchestrator_mode: bool = Field(
        default=True,
        description=(
            "Use Planner → Executor orchestrator (recommended). "
            "Set false for legacy ReAct."
        ),
    )
    orchestrator_executor_max_turns: int = Field(default=5, ge=1, le=15)
    executor_max_turns_diagnose: int = Field(
        default=15,
        ge=1,
        le=15,
        description="Executor turns for diagnose subtasks (explore + summarize).",
    )
    executor_tool_rounds_diagnose: int = Field(
        default=12,
        ge=1,
        le=30,
        description=(
            "Max tool-call rounds allowed inside a diagnose subtask before "
            "summary-only mode."
        ),
    )
    executor_max_turns_edit: int = Field(default=5, ge=1, le=15)
    executor_max_turns_verify: int = Field(default=5, ge=1, le=15)
    executor_max_turns_shell: int = Field(default=5, ge=1, le=15)
    orchestrator_max_replans: int = Field(default=3, ge=0, le=10)
    subtask_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Retry same subtask before escalating to Planner re-plan.",
    )
    subtask_quality_gate_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max L0/L1 FAIL rounds inside one edit subtask run.",
    )

    # --- Plan / Preflight gates -----------------------------------------------
    plan_gate_max_nodes: int = Field(default=4, ge=1, le=20)
    plan_gate_max_replans: int = Field(default=2, ge=0, le=10)
    preflight_yellow_ratio: float = Field(default=0.65, ge=0.1, le=1.0)
    preflight_red_ratio: float = Field(default=0.95, ge=0.1, le=1.0)
    preflight_max_chars_per_file: int = Field(default=12_000, ge=1000)
    preflight_large_file_bytes: int = Field(default=512_000, ge=10_000)
    preflight_turn_reserve_tokens: int = Field(default=8_000, ge=0)
    preflight_slice_padding: int = Field(default=15, ge=0, le=200)
    preflight_slice_max_symbols: int = Field(default=5, ge=1, le=20)

    # --- Scout discovery (pre-Planner AOP) ------------------------------------
    scout_enabled: bool = Field(
        default=False,
        description=(
            "Run read-only Scout phase before Planner "
            "(default off — use diagnose subtask instead)."
        ),
    )
    scout_preflight_grep: bool = Field(
        default=True,
        description=(
            "Run harness ripgrep before Scout LLM and inject results as context. "
            "Does not replace Scout LLM — only supplies hints."
        ),
    )
    scout_model: str = Field(
        default="openai/Qwen/Qwen2.5-7B-Instruct",
        description=(
            "Small model for Harness Scout pre-Planner probe. "
            "SiliconFlow: Qwen2.5-7B is cheap/free vs DeepSeek-V4-Flash."
        ),
    )
    scout_max_turns: int = Field(
        default=4,
        ge=1,
        le=15,
        description="Legacy cap; tool + manifest turns combined.",
    )
    scout_max_tool_turns: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Scout LLM turns allowed for grep/read tools before manifest-only phase.",
    )
    scout_manifest_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Manifest-only LLM attempts (no tools) after tool phase.",
    )
    llm_request_timeout: int = Field(
        default=180,
        ge=30,
        le=600,
        description="Timeout (seconds) for Planner/Scout non-streaming LLM calls.",
    )
    scout_max_tokens: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Max completion tokens per Scout tool-phase call.",
    )
    scout_manifest_max_tokens: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Max completion tokens for Scout manifest-only phase.",
    )
    scout_trace: bool = Field(
        default=False,
        description="If true, Scout emits <discovery_trace> before manifest JSON.",
    )
    scout_auto_approve: bool = Field(
        default=True,
        description="Auto-approve Scout tool calls (read/grep/shell) without prompts.",
    )

    # --- Repo map (static index + PageRank skeleton for Planner) --------------
    repo_map_enabled: bool = Field(
        default=True,
        description="Build ctags/parser symbol index + PageRank skeleton at chat startup.",
    )
    repo_map_top_k: int = Field(
        default=200,
        ge=10,
        le=2000,
        description="Keep top-K symbols by PageRank score in the skeleton.",
    )
    repo_map_max_chars: int = Field(
        default=12_000,
        ge=2000,
        le=50_000,
        description="Max characters for <repo_map> block sent to Planner.",
    )
    repo_map_build_timeout: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="Seconds to wait for background repo map build before Planner.",
    )
    context_retriever_enabled: bool = Field(
        default=True,
        description="Inject request-specific ContextPack into Planner input.",
    )
    patch_plan_enabled: bool = Field(
        default=False,
        description=(
            "Try experimental Planner PatchPlan before TaskTree planning. "
            "Default false because executable patches belong in the Executor skill layer."
        ),
    )
    legacy_react_fallback_enabled: bool = Field(
        default=False,
        description=(
            "Allow fallback from PatchPlan/SkillExecutor to legacy TaskTree/ReAct. "
            "Default false keeps the new architecture as the primary execution path."
        ),
    )
    executor_skill_enabled: bool = Field(
        default=True,
        description="Use deterministic skills for edit subtasks before legacy ReAct.",
    )

    # --- Prompt caching (prefix snapshot) -------------------------------------
    prompt_cache_enabled: bool = Field(
        default=True,
        description="Apply ephemeral prefix caching on supported models (Anthropic, etc.).",
    )
    prompt_cache_min_tokens: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Minimum tokens per cached block (provider minimum is usually 1024).",
    )
    prompt_cache_ttl: str = Field(
        default="5m",
        description='Cache TTL: "5m" (default) or "1h" on Anthropic.',
    )

    # --- Executor guards (Phase 2) ----------------------------------------------
    shell_dedup_limit: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Max identical shell_exec commands per subtask before blocking.",
    )
    shell_stagnant_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max consecutive failures of the same shell command before blocking.",
    )

    # --- Scoring judge (L1 rubric) -------------------------------------------
    judge_model: str | None = None
    judge_temperature: float = 0.0
    judge_max_tokens: int = Field(default=2048, ge=256)

    # --- Safety ---------------------------------------------------------------
    shell_whitelist: list[str] = Field(default_factory=list)
    never_auto: list[str] = Field(
        default_factory=lambda: ["git push", "rm -rf", "rm -r"],
    )

    # --- Storage paths --------------------------------------------------------
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".mitkii")
    project_dir: str = ".mitkii"

    model_config = {
        "env_prefix": "MITKII_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def context_budget(self) -> int:
        """Effective number of tokens available for retrieved context."""
        return int(self.max_context_tokens * self.context_budget_ratio)

    @property
    def effective_planner_model(self) -> str:
        """Model used by Planner (falls back to scout_model, then executor model)."""
        if self.planner_model:
            return self.planner_model
        if self.scout_model:
            return self.scout_model
        return self.model

    def ensure_dirs(self) -> None:
        """Create data / project directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> MitKIISettings:
    """Return a process-wide singleton of :class:`MitKIISettings`."""
    _sanitize_env_values()
    return MitKIISettings()
