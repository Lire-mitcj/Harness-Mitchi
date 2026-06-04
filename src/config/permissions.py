from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.agent.types import RiskLevel


class PermissionLevel(StrEnum):
    """How the system should handle an action that requires approval."""

    AUTO_ALLOW = "auto_allow"
    ASK_ONCE = "ask_once"
    ALWAYS_ASK = "always_ask"


@dataclass(slots=True)
class PermissionRule:
    """A single permission rule matching a tool / action pattern."""

    pattern: str
    level: PermissionLevel
    risk: RiskLevel = RiskLevel.SAFE

    def matches(self, action: str) -> bool:
        """Return True if *action* is matched by this rule's glob pattern."""
        return fnmatch.fnmatch(action, self.pattern)


@dataclass
class PermissionConfig:
    """Full set of permission rules, organized by risk level.

    Rules are evaluated top-down within each risk tier; the first match wins.
    If no rule matches, the default for the risk level is used.
    """

    safe_rules: list[PermissionRule] = field(default_factory=lambda: [
        PermissionRule(pattern="*", level=PermissionLevel.AUTO_ALLOW, risk=RiskLevel.SAFE),
    ])
    moderate_rules: list[PermissionRule] = field(default_factory=lambda: [
        PermissionRule(pattern="*", level=PermissionLevel.ASK_ONCE, risk=RiskLevel.MODERATE),
    ])
    dangerous_rules: list[PermissionRule] = field(default_factory=lambda: [
        PermissionRule(pattern="*", level=PermissionLevel.ALWAYS_ASK, risk=RiskLevel.DANGEROUS),
    ])

    _defaults: dict[RiskLevel, PermissionLevel] = field(
        init=False,
        repr=False,
        default_factory=lambda: {
            RiskLevel.SAFE: PermissionLevel.AUTO_ALLOW,
            RiskLevel.MODERATE: PermissionLevel.ASK_ONCE,
            RiskLevel.DANGEROUS: PermissionLevel.ALWAYS_ASK,
        },
    )

    def rules_for(self, risk: RiskLevel) -> list[PermissionRule]:
        mapping: dict[RiskLevel, list[PermissionRule]] = {
            RiskLevel.SAFE: self.safe_rules,
            RiskLevel.MODERATE: self.moderate_rules,
            RiskLevel.DANGEROUS: self.dangerous_rules,
        }
        return mapping[risk]

    def resolve(self, action: str, risk: RiskLevel) -> PermissionLevel:
        """Return the permission level for *action* at the given *risk*."""
        for rule in self.rules_for(risk):
            if rule.matches(action):
                return rule.level
        return self._defaults[risk]


class PermissionManager:
    """Session-scoped permission manager.

    Tracks which actions have been approved under the ``ASK_ONCE`` policy so
    the user is not prompted repeatedly for the same tool within a session.
    """

    def __init__(self, config: PermissionConfig | None = None) -> None:
        self._config = config or PermissionConfig()
        self._session_approvals: dict[str, bool] = {}

    @property
    def config(self) -> PermissionConfig:
        return self._config

    def check(self, action: str, risk: RiskLevel) -> PermissionCheckResult:
        """Evaluate whether *action* at *risk* is allowed, denied, or needs a prompt.

        Returns a :class:`PermissionCheckResult` describing the decision.
        """
        level = self._config.resolve(action, risk)

        if level is PermissionLevel.AUTO_ALLOW:
            return PermissionCheckResult(allowed=True, needs_prompt=False, action=action)

        if level is PermissionLevel.ASK_ONCE:
            previous = self._session_approvals.get(action)
            if previous is not None:
                return PermissionCheckResult(
                    allowed=previous, needs_prompt=False, action=action
                )
            return PermissionCheckResult(allowed=False, needs_prompt=True, action=action)

        # ALWAYS_ASK
        return PermissionCheckResult(allowed=False, needs_prompt=True, action=action)

    def record_decision(self, action: str, approved: bool) -> None:
        """Record the user's approval or rejection for an ``ASK_ONCE`` action."""
        self._session_approvals[action] = approved

    def session_decision(self, action: str) -> bool | None:
        """Return the session decision for *action*, or ``None`` if not decided yet."""
        if action not in self._session_approvals:
            return None
        return self._session_approvals[action]

    def is_approved(self, action: str) -> bool:
        """Check if an action was previously approved in this session."""
        return self._session_approvals.get(action, False)

    def reset(self) -> None:
        """Clear all remembered session approvals."""
        self._session_approvals.clear()

    def approved_actions(self) -> list[str]:
        """Return a list of all actions approved so far in this session."""
        return [a for a, ok in self._session_approvals.items() if ok]

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state (for checkpoint persistence)."""
        return {"session_approvals": dict(self._session_approvals)}

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], config: PermissionConfig | None = None
    ) -> PermissionManager:
        """Restore from a previously serialized dict."""
        mgr = cls(config=config)
        mgr._session_approvals = dict(data.get("session_approvals", {}))
        return mgr


@dataclass(frozen=True, slots=True)
class PermissionCheckResult:
    """Outcome of a permission check."""

    allowed: bool
    needs_prompt: bool
    action: str
