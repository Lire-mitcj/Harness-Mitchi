from __future__ import annotations

from pathlib import Path

import pytest

from src.indexer.language_profiles import (
    GO,
    JAVA,
    PROTO,
    PYTHON,
    build_import_mode_pattern,
    build_symbol_mode_pattern,
    classify_line_with_profiles,
    extract_symbol_with_profiles,
    profile_for_extension,
    profile_for_path,
)
from src.indexer.project_stack import detect_project_stack
from src.tools.grep_match_symbols import classify_match_line, extract_symbol_from_match_line


def test_profile_for_go_extension() -> None:
    assert profile_for_extension(".go") is GO
    assert profile_for_extension(".proto") is PROTO


def test_build_symbol_mode_pattern_go() -> None:
    pattern = build_symbol_mode_pattern("GetOrder", (GO,))
    assert "func" in pattern
    assert "GetOrder" in pattern


def test_build_symbol_mode_pattern_multi_profile() -> None:
    pattern = build_symbol_mode_pattern("Foo", (PYTHON, GO, JAVA))
    assert "def" in pattern
    assert "func" in pattern
    assert "class" in pattern


def test_extract_go_func_symbol() -> None:
    line = "func (s *OrderService) GetOrder(ctx context.Context) error {"
    assert extract_symbol_from_match_line(line, file_path="service.go") == "GetOrder"


def test_extract_java_class_symbol() -> None:
    line = "public class OrderController {"
    assert extract_symbol_from_match_line(line, file_path="OrderController.java") == "OrderController"


def test_extract_proto_service_symbol() -> None:
    line = "service OrderService {"
    assert extract_symbol_from_match_line(line, file_path="order.proto") == "OrderService"


def test_classify_go_definition() -> None:
    kind = classify_match_line("func main() {", file_path="main.go")
    assert kind == "definition"


def test_classify_java_mount() -> None:
    kind = classify_match_line("@RestController", file_path="App.java")
    assert kind in {"mount", "definition"}


def test_detect_go_project_stack(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/app\n\ngo 1.22\n", encoding="utf-8")
    (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    assert stack.primary == "go"
    assert "go" in stack.languages
    assert stack.validator_command[0] == "go"


def test_detect_java_project_stack(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project></project>", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    assert stack.primary == "java"
    assert stack.validator_command[0] == "mvn"


def test_mixed_go_proto_uses_go_validator(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/qq\n\ngo 1.22\n", encoding="utf-8")
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "x.proto").write_text("service Foo {}\n", encoding="utf-8")
    stack = detect_project_stack(tmp_path)
    assert stack.primary == "mixed"
    assert "go" in stack.languages
    assert stack.validator_command == ("go", "test", "./...")


def test_indexable_extensions_for_mixed_go_proto() -> None:
    from src.indexer.project_stack import ProjectStack, indexable_extensions_for_stack

    stack = ProjectStack(
        primary="mixed",
        languages=("go", "proto"),
        default_include_globs=("*.go", "*.proto"),
        validator_command=("go", "test", "./..."),
        discovery_patterns=(),
    )
    assert indexable_extensions_for_stack(stack) == frozenset({".go", ".proto"})


def test_python_still_extracts_def() -> None:
  line = "async def build_router(engine):"
  assert extract_symbol_from_match_line(line, file_path="list.py") == "build_router"


def test_classify_line_proto_definition() -> None:
    kind = classify_line_with_profiles(
        "message OrderRequest {",
        pattern="OrderRequest",
        profiles=(PROTO,),
    )
    assert kind == "definition"
