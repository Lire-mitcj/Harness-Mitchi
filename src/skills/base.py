from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.context.pack import ContextPack
from src.planner.patch_plan import PatchPlan


@dataclass(frozen=True)
class SkillContext:
    user_request: str
    context_pack: ContextPack | None = None
    patch_plan: PatchPlan | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    success: bool
    summary: str
    changed_files: tuple[str, ...] = ()
    validation_result: str = ""
    missing_info: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    requires_fallback: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


class Skill(Protocol):
    name: str

    async def run(self, context: SkillContext, **kwargs: Any) -> SkillResult: ...


class SkillExecutor:
    """Registry for deterministic skill execution before ReAct fallback."""

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills or []:
            self.register(skill)

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def has(self, name: str) -> bool:
        return name in self._skills

    async def run(
        self,
        name: str,
        context: SkillContext,
        **kwargs: Any,
    ) -> SkillResult:
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult(
                success=False,
                summary=f"Skill '{name}' is not registered.",
                missing_info=(f"skill:{name}",),
                requires_fallback=True,
            )
        return await skill.run(context, **kwargs)
