#!/usr/bin/env python3
"""One-shot E2E runner for mitkii assembled loop."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT = Path("/mnt/d/sqlkeshe/database-course-design")
TASK = (
    "这些新增接口发生数据库异常时，是否应该直接返回默认 500？"
    "你生成一个全局异常处理器，统一记录日志、转换异常并给前端一致的"
    "错误格式"
)
LOG_PATH = Path("/home/csh/harness-mitkii/tmp_e2e_run.log")


def log(line: str) -> None:
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


async def main() -> int:
    import os

    from src.agent.events import EventType
    from src.agent.state_assembled_loop import StateAssembledLoop
    from src.runtime.session_factory import create_mitkii_session

    os.chdir(PROJECT)
    LOG_PATH.write_text("", encoding="utf-8")
    log(f"project_root={PROJECT}")
    log(f"task={TASK!r}")

    session = create_mitkii_session(project_root=PROJECT)
    loop = StateAssembledLoop(
        llm=session.llm,
        tools=session.tools,
        harness=session.harness,
        context=session.context_builder,
        permissions=session.permissions,
        settings=session.settings,
    )

    final_answer = ""
    tool_calls = 0
    async for event in loop.run(TASK):
        etype = event.type.value if hasattr(event.type, "value") else str(event.type)
        if event.type == EventType.TOOL_CALL:
            tool_calls += 1
            name = (event.data or {}).get("tool", event.content or "?")
            args = (event.data or {}).get("params", {})
            log(f"[tool_call] {name} {json.dumps(args, ensure_ascii=False)[:400]}")
        elif event.type == EventType.TOOL_RESULT:
            name = (event.data or {}).get("tool", "?")
            ok = (event.data or {}).get("success", True)
            body = (event.content or "")[:600]
            log(f"[tool_result] {name} success={ok} {body}")
        elif event.type == EventType.FINAL_ANSWER:
            final_answer = event.content or final_answer
            if event.content:
                log(f"[final_answer] {(event.content or '')[:1200]}")
        elif event.type == EventType.ERROR:
            log(f"[error] {event.content}")
        elif event.type == EventType.FILE_EDIT:
            path = (event.data or {}).get("path", "?")
            log(f"[file_edit] {path}")

    log(f"tool_calls={tool_calls}")
    log(f"final_answer={final_answer[:2000]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
