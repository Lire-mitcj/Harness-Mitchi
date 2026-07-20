from __future__ import annotations

from src.agent.manifest import EvidenceItem, StepManifest, Sufficiency
from src.hooks.before_tool import inspect_tool_request_async
import asyncio


def _manifest_with_list_target() -> StepManifest:
    return StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:main.py:app",
                need="app",
                type="symbol",
                role="observed",
                file="main.py",
                span=(41, 41),
                symbol="app",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )


def test_blocks_read_only_decision_edit_on_main_py() -> None:
    result = asyncio.run(
        inspect_tool_request_async(
            "decision_edit",
            {
                "target_file": "./main.py",
                "intent": "查看 main.py 中 app 初始化及路由挂载区域，确认 list.py 的 router 如何被 include",
                "context_window": [{"file": "./main.py", "span": [1, 100]}],
            },
            allowed_tools={"decision_edit"},
            manifest=_manifest_with_list_target(),
        )
    )
    assert result is not None
    assert result.startswith("BLOCK:")
    assert "list.py" in result


def test_blocks_read_led_intent_even_when_noun_contains_create() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="router",
                type="symbol",
                role="observed",
                file="list.py",
                span=(16, 358),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    intent = (
        "查看 main.py 中 FastAPI 应用创建和路由挂载的代码段，"
        "以确定全局异常处理器的插入位置。"
    )
    result = asyncio.run(
        inspect_tool_request_async(
            "decision_edit",
            {
                "target_file": "./main.py",
                "intent": intent,
                "context_window": [{"file": "./main.py", "span": [1, 50]}],
            },
            allowed_tools={"decision_edit"},
            manifest=manifest,
        )
    )
    assert result is not None
    assert result.startswith("BLOCK:")
    assert "view_symbol_code" in result


def test_blocks_read_only_decision_edit_on_primary_edit_target() -> None:
    """Scheme B edit-only must not let Core smuggle view via decision_edit."""
    result = asyncio.run(
        inspect_tool_request_async(
            "decision_edit",
            {
                "target_file": "./list.py",
                "intent": "查看 list.py 的完整内容，特别是 _load 函数和 YAML 加载逻辑",
                "context_window": [{"file": "./list.py", "span": [1, 300]}],
            },
            allowed_tools={"decision_edit"},
            manifest=_manifest_with_list_target(),
        )
    )
    assert result is not None
    assert result.startswith("BLOCK:")
    assert "patches, not reading" in result


def test_allows_patch_intent_on_list_py() -> None:
    result = asyncio.run(
        inspect_tool_request_async(
            "decision_edit",
            {
                "target_file": "./list.py",
                "intent": "Add imports and wrap database routes with _db_error_handler",
                "focus_symbols": ["build_router"],
                "context_window": [{"file": "./list.py", "span": [16, 80]}],
            },
            allowed_tools={"decision_edit"},
            manifest=_manifest_with_list_target(),
        )
    )
    assert result is None
