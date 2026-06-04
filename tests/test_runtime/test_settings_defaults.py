from __future__ import annotations

from src.config.settings import MitKIISettings


def test_new_architecture_defaults_use_skill_executor_without_legacy_fallback() -> None:
    settings = MitKIISettings()

    assert settings.context_retriever_enabled is True
    assert settings.patch_plan_enabled is False
    assert not hasattr(settings, "legacy_react_fallback_enabled")
    assert settings.executor_skill_enabled is True
