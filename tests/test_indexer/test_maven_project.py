from pathlib import Path

from src.indexer.ctags import CtagsSymbol
from src.indexer.maven_project import find_maven_project, module_paths_from_pom, parse_pom
from src.indexer.project_stack import detect_project_stack


def test_parse_pom_modules_and_artifact(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "pom.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<project>
  <artifactId>api</artifactId>
  <packaging>jar</packaging>
</project>
""",
        encoding="utf-8",
    )
    (tmp_path / "pom.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>demo-parent</artifactId>
  <packaging>pom</packaging>
  <modules>
    <module>api</module>
    <module>service</module>
  </modules>
</project>
""",
        encoding="utf-8",
    )

    project = parse_pom(tmp_path / "pom.xml")
    assert project is not None
    assert project.artifact_id == "demo-parent"
    assert project.packaging == "pom"
    assert [mod.path for mod in project.modules] == ["api", "service"]
    assert project.modules[0].artifact_id == "api"
    assert module_paths_from_pom(tmp_path / "pom.xml") == ("api", "service")


def test_find_maven_project_returns_none_without_pom(tmp_path: Path) -> None:
    assert find_maven_project(tmp_path) is None


def test_detect_project_stack_includes_maven_modules(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text(
        """<project>
  <artifactId>root</artifactId>
  <modules><module>core</module></modules>
</project>""",
        encoding="utf-8",
    )
    stack = detect_project_stack(tmp_path)
    assert stack.primary == "java"
    assert stack.maven_modules == ("core",)
