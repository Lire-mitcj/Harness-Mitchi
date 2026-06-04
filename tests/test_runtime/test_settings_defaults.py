from __future__ import annotations

from src.config.settings import MitKIISettings


def test_new_architecture_defaults_use_single_executor_agent_without_legacy_fallback() -> None:
    settings = MitKIISettings()

    assert settings.context_retriever_enabled is True
    assert settings.patch_plan_enabled is False
    assert not hasattr(settings, "legacy_react_fallback_enabled")
    assert settings.executor_skill_enabled is False
    assert settings.effective_final_summary_model == "openai/deepseek-ai/DeepSeek-V3"
