from __future__ import annotations

from typing import Any

import pytest

from src.agent.types import LLMResponse
from src.orchestrator.final_summarizer import (
    FinalSummarizer,
    _clean_summary,
    build_deterministic_user_summary,
)
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree


class FakeSummaryLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.kwargs: dict[str, Any] = {}

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        self.messages = messages
        self.kwargs = kwargs
        yield "已完成视图定位。", None
        yield "", LLMResponse(
            content="已完成视图定位。",
            tool_calls=None,
            usage=None,
            model="fake",
        )


@pytest.mark.asyncio
async def test_final_summarizer_uses_structured_run_result() -> None:
    llm = FakeSummaryLLM()
    summarizer = FinalSummarizer(llm, max_tokens=512, timeout=12)
    tree = TaskTree(
        root_task="查找项目中使用的视图",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="搜索项目中使用的视图",
                status=SubTaskStatus.SUCCESS,
            )
        ],
    )

    text = await summarizer.summarize(
        user_request="查找项目中使用的视图",
        task_tree=tree,
        subtask_summaries={"st-1": "Result: 已定位 1 个相关代码位置。"},
    )

    assert text == "已完成视图定位。"
    assert llm.kwargs["max_tokens"] == 512
    assert llm.kwargs["timeout"] == 12
    assert "RUN_RESULT_JSON" in llm.messages[1]["content"]
    assert "搜索项目中使用的视图" in llm.messages[1]["content"]


@pytest.mark.asyncio
async def test_final_summarizer_strips_executor_digest_from_payload() -> None:
    llm = FakeSummaryLLM()
    summarizer = FinalSummarizer(llm)
    tree = TaskTree(
        root_task="查找项目中使用的视图",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="搜索项目中使用的视图",
                status=SubTaskStatus.SUCCESS,
            )
        ],
    )

    await summarizer.summarize(
        user_request="查找项目中使用的视图",
        task_tree=tree,
        subtask_summaries={
            "st-1": (
                "Result: 已定位 1 个相关代码位置。\n"
                "Evidence:\n"
                "- db/init/init.sql:354 | 视图定义 | CREATE VIEW v_flight_monitoring AS\n"
                "Executor evidence digest:\n"
                "[grep_search pattern='view']\n"
                "\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\""
            )
        },
    )

    payload_text = llm.messages[1]["content"]
    assert "db/init/init.sql:354" in payload_text
    assert "Executor evidence digest" not in payload_text
    assert "[grep_search" not in payload_text


def test_final_summarizer_rejects_corrupted_output() -> None:
    assert _clean_summary('完成。""""""""""""""""""""""""""""""""""""""""""""') == ""
    assert _clean_summary("已完成。\n- db/init.sql:354 - �目目标代码") == ""


def test_deterministic_summary_preserves_evidence_locations() -> None:
    tree = TaskTree(
        root_task="查找项目中使用的视图",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="搜索项目中使用的视图",
                status=SubTaskStatus.SUCCESS,
            )
        ],
    )

    text = build_deterministic_user_summary(
        user_request="查找项目中使用的视图",
        task_tree=tree,
        subtask_summaries={
            "st-1": (
                "Result: 已定位 2 个相关代码位置。\n"
                "Evidence:\n"
                "- /mnt/d/project/db/init/init.sql:352 | 目标代码 | -- 视图：航班满座率监控\n"
                "- /mnt/d/project/db/init/init.sql:354 | 目标代码 | CREATE VIEW v_flight_monitoring AS\n"
                "Conclusion: acceptance met。"
            )
        },
    )

    assert "已完成。" in text
    assert "/mnt/d/project/db/init/init.sql:352 - 目标代码: -- 视图：航班满座率监控" in text
    assert "�" not in text
