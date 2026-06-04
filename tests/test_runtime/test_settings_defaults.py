from __future__ import annotations

from src.config.settings import MitKIISettings


def test_new_architecture_defaults_disable_legacy_react_fallback() -> None:
    settings = MitKIISettings()

    assert settings.context_retriever_enabled is True
    assert settings.patch_plan_enabled is False
    assert settings.legacy_react_fallback_enabled is False
    assert settings.executor_skill_enabled is True
