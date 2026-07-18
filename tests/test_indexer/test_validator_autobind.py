from pathlib import Path

from src.config.settings import MitKIISettings
from src.harness.engine import HarnessEngine
from src.indexer.project_stack import apply_project_stack_to_settings


def test_apply_project_stack_binds_go_validator(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")

    settings = MitKIISettings(cursor_validator_command=["pytest"], cursor_validator_auto=True)
    updated = apply_project_stack_to_settings(settings, tmp_path)

    assert updated.cursor_validator_command == ["go", "test", "./..."]


def test_apply_project_stack_respects_custom_validator(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")

    settings = MitKIISettings(
        cursor_validator_command=["make", "test"],
        cursor_validator_auto=True,
    )
    updated = apply_project_stack_to_settings(settings, tmp_path)

    assert updated.cursor_validator_command == ["make", "test"]


def test_apply_project_stack_auto_disabled(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")

    settings = MitKIISettings(cursor_validator_command=["pytest"], cursor_validator_auto=False)
    updated = apply_project_stack_to_settings(settings, tmp_path)

    assert updated.cursor_validator_command == ["pytest"]


def test_harness_engine_create_applies_validator(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project><artifactId>demo</artifactId></project>", encoding="utf-8")

    settings = MitKIISettings(cursor_validator_command=["pytest"], cursor_validator_auto=True)
    harness = HarnessEngine.create(settings, project_root=tmp_path)

    assert harness.settings.cursor_validator_command == ["mvn", "-q", "test"]
