from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agent.cursor_contracts import ContextPack, ContextWindow
from src.agent.cursor_decision import CursorDecisionLLM
from src.agent.cursor_executor import CursorExecutor
from src.agent.cursor_patch_applier import CursorPatchApplier
from src.agent.cursor_validator import CursorValidator
from src.agent.types import RiskLevel, ToolResult
from src.llm.client import LLMClient
from src.tools.base import Tool

log = logging.getLogger(__name__)


class DecisionEditTool(Tool):
    name = "decision_edit"
    description = (
        "Generate and apply a SEARCH/REPLACE patch to a file using the Decision LLM. "
        "Automatically runs linter, tests, and syntax validation. Rolls back if validation fails."
    )
    risk_level = RiskLevel.MODERATE
    parameters = {
        "type": "object",
        "properties": {
            "target_file": {
                "type": "string",
                "description": "目标文件的相对路径。注意：此文件必须存在于你当前上下文的活跃文件(Active Files)列表中。如果你尚未检索它，请先调用 codebase_retrieve。"
            },
            "instruction": {
                "type": "string",
                "description": "针对该文件的具体、单一、明确的代码修改指令。必须使用动作词开头（例如：'将 disabled 的返回值改为 True'，'在第20行引入 logger 模块'）。严禁下达宏观、含糊的含糊命令。"
            }
        },
        "required": ["target_file", "instruction"]
    }

    def __init__(
        self,
        *,
        project_root: Path,
        settings: Any,
        decision_llm: Any,
        harness: Any,
    ) -> None:
        self.project_root = project_root.resolve()
        self.settings = settings
        self.decision_llm = decision_llm
        self.harness = harness

        self.decision = CursorDecisionLLM(self.decision_llm)
        self.patch_applier = CursorPatchApplier(self.project_root)
        self.executor = CursorExecutor(self.project_root, self.patch_applier)

        validator_llm = self.decision_llm
        if getattr(validator_llm, "model", "") != settings.cursor_validator_model:
            validator_llm = LLMClient(
                model=settings.cursor_validator_model,
                request_timeout=float(settings.llm_request_timeout),
                prompt_cache_enabled=settings.prompt_cache_enabled,
                prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
                prompt_cache_ttl=(
                    "1h"
                    if settings.prompt_cache_ttl.strip().lower() in {"1h", "hour", "60m"}
                    else "5m"
                ),
            )
        self.validator = CursorValidator(
            self.project_root,
            command=tuple(settings.cursor_validator_command),
            timeout=settings.cursor_validator_timeout,
            max_error_chars=settings.cursor_observation_max_chars,
            semantic_llm=validator_llm,
            semantic_timeout=settings.cursor_validator_semantic_timeout,
        )
        self.active_files: list[str] = []

    def set_active_files(self, active_files: list[str]) -> None:
        self.active_files = list(active_files)

    def _build_context_pack(self, target_file: str) -> ContextPack:
        windows = []
        all_files = list(self.active_files)
        if target_file not in all_files:
            all_files.append(target_file)

        for file in all_files:
            abs_path = (self.project_root / file).resolve()
            if not abs_path.exists():
                continue
            try:
                content = abs_path.read_text(encoding="utf-8")
                lines = content.splitlines()
                windows.append(
                    ContextWindow(
                        file=file,
                        start_line=1,
                        end_line=len(lines),
                        content=content,
                        symbols=(),
                        semantic_tags=(),
                    )
                )
            except Exception as exc:
                log.warning("Failed to read context file %s: %s", file, exc)

        return ContextPack(windows=tuple(windows))

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        target_file = validated["target_file"]
        instruction = validated["instruction"]

        # Build context pack dynamically containing active files
        context_pack = self._build_context_pack(target_file)

        # Build state text and prompt DecisionLLM to get patch
        state_text = f"Apply the following instruction to target file:\n{instruction}"
        try:
            decision_messages = self.decision.build_messages(
                state_text=state_text,
                context_pack=context_pack,
                hint=None,
            )
            trimmed_messages = await self.harness.before_llm_call(decision_messages)
            response = await self.decision_llm.chat(
                trimmed_messages,
                tools=None,
                stream=False,
            )
            await self.harness.after_llm_call(response, response.usage)
            parsed_decision = self.decision.parse(
                response.content or "",
                context_pack.candidate_files,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"DecisionLLM generation/parsing failed: {exc}",
            )

        if parsed_decision.action != "edit":
            return ToolResult(
                success=False,
                output=f"Decision LLM did not generate an edit. Action taken: {parsed_decision.action}. Message: {parsed_decision.answer or parsed_decision.clarification}",
                error=f"Action is not edit: {parsed_decision.action}",
            )

        # Run transaction
        execution, validation, pipeline_metrics = await self.executor.execute_transaction(
            parsed_decision.target_file,
            parsed_decision.patch,
            self.validator,
            step=1,
            layer1=None,
            user_intent=instruction,
        )

        if not execution.success:
            return ToolResult(
                success=False,
                output=(
                    f"❌ 【补丁生成失败】：DecisionLLM 生成的 SEARCH/REPLACE 块无法匹配到文件 `{target_file}` 的物理内容。\n"
                    f"原因分析：可能因为你在 instruction 中提供的描述有误，或者你参考了过期的代码快照。\n"
                    f"具体底层错误：{execution.error}"
                ),
                error=execution.error,
                metadata={
                    "pipeline_metrics": pipeline_metrics,
                    "execution": execution,
                    "validation": validation,
                }
            )

        if not validation.success:
            # 编译或测试挂了，但是补丁已经被应用过（虽然现在回滚了）
            return ToolResult(
                success=False,
                output=(
                    f"❌ 【自动化代码验证失败】：补丁可以成功应用，但导致系统出现编译、语法或测试错误。\n"
                    f"👉 已经自动将修改全部物理回滚（Rollback）。\n"
                    f"👉 具体的验证器报错如下，请根据此错误重新构思修改指令：\n"
                    f"```\n{validation.error}\n```"
                ),
                error=validation.error,
                metadata={
                    "pipeline_metrics": pipeline_metrics,
                    "execution": execution,
                    "validation": validation,
                }
            )

        return ToolResult(
            success=True,
            output=f"✅ 【应用成功】：针对文件 `{target_file}` 的补丁已通过编译与自动化测试验证，修改已持久化提交（Committed）。",
            metadata={
                "pipeline_metrics": pipeline_metrics,
                "execution": execution,
                "validation": validation,
            }
        )
