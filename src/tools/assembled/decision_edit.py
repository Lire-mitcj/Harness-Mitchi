from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.agent.contracts import ContextPack, ContextWindow
from src.agent.decision import CursorDecisionLLM
from src.agent.executor import CursorExecutor
from src.agent.patch_applier import CursorPatchApplier
from src.agent.types import RiskLevel, ToolResult
from src.agent.validator import CursorValidator
from src.llm.client import LLMClient
from src.tools.base import Tool

log = logging.getLogger(__name__)


class DecisionEditTool(Tool):
    name = "decision_edit"
    description = (
        "Apply a scoped edit to exactly one file when STEP EVIDENCE shows edit_ready:yes "
        "and the change is grounded in loaded code anchors. Generates a SEARCH/REPLACE "
        "patch via the Decision LLM, then runs lint/tests/syntax checks; rolls back on "
        "failure. Required: target_file, intent, focus_symbols, and context_window "
        "entries with frozen file+span evidence from loaded anchors. One call modifies "
        "one file only—do not batch multi-file changes. Do NOT use to inspect, grep, "
        "or load code. Do not call in parallel with retrieval reads the patch depends on."
    )
    risk_level = RiskLevel.MODERATE
    parameters = {
        "type": "object",
        "properties": {
            "target_file": {
                "type": "string",
                "description": "目标文件的相对路径。"
            },
            "intent": {
                "type": "string",
                "description": "对单个目标文件进行修改的意图与要求（如修改逻辑、添加功能等）。修订意图必须局限在当前 target_file 内部。"
            },
            "focus_symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "本次修改重点关注/涉及的符号列表（如函数名、类名）。"
            },
            "context_window": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "参考文件的相对路径"},
                        "span": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "参考代码片段的起止行号，如 [10, 50]"
                        },
                        "reason": {"type": "string", "description": "该参考片段被纳为上下文的原因/用途说明"}
                    },
                    "required": ["file", "span"]
                },
                "description": "为本次编辑显式注入的、已冻结的上下文片段列表。如果提供此字段，编辑工具将严格仅从这些片段提取依赖上下文，绝不自行检索或搜寻其他代码。"
            },
            "task_id": {
                "type": "string",
                "description": "可选的任务包/步骤 ID。"
            },
            "mode": {
                "type": "string",
                "enum": ["edit"],
                "description": "任务模式，固定为 'edit'。"
            },
            "constraints": {
                "type": "object",
                "properties": {
                    "no_further_retrieval": {"type": "boolean", "default": True},
                    "patch_scope": {"type": "string", "default": "single_file_only"}
                },
                "description": "任务执行约束条件。"
            }
        },
        "required": ["target_file", "intent", "context_window"]
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
        configured_timeout = getattr(settings, "cursor_decision_timeout", 90.0)
        self.decision_timeout = (
            float(configured_timeout)
            if isinstance(configured_timeout, (int, float))
            else 90.0
        )

        self.decision = CursorDecisionLLM(self.decision_llm)
        self.patch_applier = CursorPatchApplier(self.project_root)
        self.executor = CursorExecutor(self.project_root, self.patch_applier)

        validator_llm = self.decision_llm
        if not settings.cursor_validator_model or settings.cursor_validator_model.lower() in ("none", "disabled", "false"):
            validator_llm = None
        elif getattr(validator_llm, "model", "") != settings.cursor_validator_model:
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

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "context_window" not in params:
            params = {**params, "context_window": []}
        return super().validate_params(params)

    def _read_file_span(self, file_path: str, start_line: int, end_line: int) -> str | None:
        abs_path = (self.project_root / file_path).resolve()
        if not abs_path.is_file():
            return None
        try:
            lines = abs_path.read_text(encoding="utf-8").splitlines()
            start = max(1, start_line)
            end = min(len(lines), end_line)
            return "\n".join(lines[start - 1 : end])
        except Exception as exc:
            log.warning(
                "Failed to read span %s:%d-%d from disk: %s",
                file_path,
                start_line,
                end_line,
                exc,
            )
            return None

    def _build_context_pack(
        self,
        target_file: str,
        search_cache: dict[str, Any] | None = None,
        context_window: list[dict[str, Any]] | None = None,
    ) -> ContextPack:
        windows = []

        norm_target = target_file.replace("\\", "/").lstrip("./")
        target_spans: list[tuple[str, int, int]] = []
        reference_spans: list[tuple[str, int, int]] = []

        if context_window is not None:
            for item in context_window:
                file_path = item.get("file")
                span = item.get("span")
                if not file_path or not span or len(span) < 2:
                    continue
                norm_file = str(file_path).replace("\\", "/").lstrip("./")
                entry = (str(file_path), int(span[0]), int(span[1]))
                if norm_file == norm_target:
                    target_spans.append(entry)
                else:
                    reference_spans.append(entry)

        if target_spans:
            for index, (file_path, start_line, end_line) in enumerate(target_spans):
                content = self._read_file_span(file_path, start_line, end_line)
                if content is None:
                    continue
                windows.append(
                    ContextWindow(
                        file=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        symbols=(),
                        semantic_tags=(),
                        role="target" if index == 0 else "reference",
                        mode="snippet",
                    )
                )
        else:
            abs_target_path = (self.project_root / target_file).resolve()
            if abs_target_path.exists():
                try:
                    lines = abs_target_path.read_text(encoding="utf-8").splitlines()
                    content = "\n".join(lines)
                    windows.append(
                        ContextWindow(
                            file=target_file,
                            start_line=1,
                            end_line=len(lines),
                            content=content,
                            symbols=(),
                            semantic_tags=(),
                            role="target",
                            mode="full",
                        )
                    )
                except Exception as exc:
                    log.warning("Failed to read target file %s: %s", target_file, exc)

        for file_path, start_line, end_line in reference_spans:
            content = self._read_file_span(file_path, start_line, end_line)
            if content is None:
                continue
            windows.append(
                ContextWindow(
                    file=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    symbols=(),
                    semantic_tags=(),
                    role="reference",
                    mode="snippet",
                )
            )

        if context_window is None:
            # Fall back to historical raw_evidence_store context loading (legacy behavior)
            raw_evidence = []
            if search_cache and "raw_evidence_store" in search_cache:
                raw_evidence = search_cache["raw_evidence_store"]

            for item in raw_evidence:
                file_path = item["file"]
                if file_path == target_file:
                    continue

                abs_path = (self.project_root / file_path).resolve()
                start_line, end_line = item["span"]

                content = None
                if abs_path.is_file():
                    try:
                        lines = abs_path.read_text(encoding="utf-8").splitlines()
                        start = max(1, start_line)
                        end = min(len(lines), end_line)
                        content = "\n".join(lines[start - 1 : end])
                    except Exception as exc:
                        log.warning("Failed to read span %s:%d-%d from disk: %s", file_path, start_line, end_line, exc)

                if content is None:
                    content = item["code"]

                windows.append(
                    ContextWindow(
                        file=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        symbols=(),
                        semantic_tags=(),
                        role="reference",
                        mode="snippet",
                    )
                )

        return ContextPack(windows=tuple(windows))

    @staticmethod
    def _build_evidence_flag(
        target_file: str,
        search_cache: dict[str, Any] | None,
        context_pack: ContextPack,
    ) -> dict[str, Any]:
        anchors: list[dict[str, Any]] = []
        if search_cache:
            raw = search_cache.get("raw_evidence_store") or []
            if isinstance(raw, list):
                anchors.extend(item for item in raw if isinstance(item, dict))
            encoded = search_cache.get("search_output")
            if isinstance(encoded, str) and encoded.strip():
                try:
                    decoded = json.loads(encoded)
                    if isinstance(decoded, list):
                        anchors.extend(item for item in decoded if isinstance(item, dict))
                except (TypeError, json.JSONDecodeError) as exc:
                    log.warning("Failed to parse code-anchor search_output: %s", exc)

        unique: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
        for anchor in anchors:
            file_path = str(anchor.get("file") or "")
            span = tuple(anchor.get("span") or ())
            if file_path and len(span) == 2:
                unique[(file_path, span)] = anchor

        target_anchors = []
        reference_anchors = []
        related: dict[tuple[str, str, tuple[Any, ...]], dict[str, Any]] = {}
        for (file_path, span), anchor in unique.items():
            compact = {"file": file_path, "span": list(span), "status": "loaded"}
            if anchor.get("symbol"):
                compact["symbol"] = anchor["symbol"]
            (target_anchors if file_path == target_file else reference_anchors).append(compact)
            for function in anchor.get("related_functions") or []:
                if not isinstance(function, dict) or not function.get("name"):
                    continue
                function_span = tuple(function.get("span") or ())
                key = (str(function["name"]), str(function.get("file") or ""), function_span)
                related[key] = {
                    "name": function["name"],
                    "file": function.get("file"),
                    "span": list(function_span),
                }

        return {
            "target_file": target_file,
            "target_code_anchors": target_anchors,
            "reference_code_anchors": reference_anchors,
            "first_hop_functions": list(related.values()),
            "available_context_files": list(context_pack.candidate_files),
            "can_edit": bool(context_pack.windows),
        }

    async def execute(self, **params: Any) -> ToolResult:
        started_at = time.monotonic()
        validated = self.validate_params(params)
        target_file = validated["target_file"]
        intent = validated["intent"]
        search_cache = params.get("_search_cache")
        context_window = params.get("context_window")

        # Build context pack dynamically containing target file and context files
        context_pack = self._build_context_pack(
            target_file,
            search_cache,
            context_window=context_window,
        )
        context_chars = sum(len(window.content) for window in context_pack.windows)
        context_lines = sum(len(window.content.splitlines()) for window in context_pack.windows)
        print(
            "[debug][decision-edit][context] "
            f"file={target_file} windows={len(context_pack.windows)} "
            f"lines={context_lines} chars={context_chars} "
            f"elapsed={time.monotonic() - started_at:.2f}s",
            flush=True,
        )

        evidence_flag = self._build_evidence_flag(target_file, search_cache, context_pack)

        # Build state text and prompt DecisionLLM to get patch
        state_text = f"Apply the following modification intent to target file:\n{intent}"
        try:
            decision_messages = self.decision.build_messages(
                state_text=state_text,
                context_pack=context_pack,
                hint=None,
                evidence_flag=evidence_flag,
                edit_only=True,
            )
            trimmed_messages = await self.harness.before_llm_call(decision_messages)

            use_stream = hasattr(self.decision_llm, "chat_stream")
            response = None
            content_chunks = []

            def count_diff_lines(text: str) -> tuple[int, int]:
                normalized = text.replace("\\n", "\n")
                added = 0
                removed = 0
                state = "normal"
                for line in normalized.splitlines():
                    clean_line = line.strip().strip('"\'')
                    if "<<<<<<< SEARCH" in clean_line:
                        state = "search"
                    elif "=======" in clean_line:
                        state = "replace"
                    elif ">>>>>>> REPLACE" in clean_line:
                        state = "normal"
                    else:
                        if state == "search":
                            removed += 1
                        elif state == "replace":
                            added += 1
                return added, removed

            if use_stream:
                try:
                    stream_generator = self.decision_llm.chat_stream(
                        trimmed_messages,
                        tools=None,
                    )
                    if not hasattr(stream_generator, "__aiter__"):
                        if inspect.iscoroutine(stream_generator):
                            stream_generator.close()
                        use_stream = False
                except Exception:
                    use_stream = False

            print(
                "[debug][decision-edit][decision-llm-start] "
                f"file={target_file} timeout={self.decision_timeout:g}s",
                flush=True,
            )
            async with asyncio.timeout(self.decision_timeout):
                if use_stream:
                    if (
                        hasattr(self.harness, "progress_callback")
                        and self.harness.progress_callback
                    ):
                        self.harness.progress_callback(f"正在编辑文件: {target_file}… [+0 -0]")

                    async for content_chunk, final_response in stream_generator:
                        if content_chunk:
                            content_chunks.append(content_chunk)
                            added, removed = count_diff_lines("".join(content_chunks))
                            if (
                                hasattr(self.harness, "progress_callback")
                                and self.harness.progress_callback
                            ):
                                self.harness.progress_callback(
                                    f"正在编辑文件: {target_file}… [+{added} -{removed}]"
                                )
                        if final_response is not None:
                            response = final_response
                else:
                    response = await self.decision_llm.chat(
                        trimmed_messages,
                        tools=None,
                        stream=False,
                    )

            print(
                "[debug][decision-edit][decision-llm-done] "
                f"file={target_file} elapsed={time.monotonic() - started_at:.2f}s",
                flush=True,
            )

            if response:
                await self.harness.after_llm_call(response, response.usage)

            session_id = getattr(self.harness, "session_id", "default_session")
            subtask_id = f"step_{getattr(self.harness, 'current_step', 1)}_edit"
            if hasattr(self.harness, "session_storage"):
                for msg in trimmed_messages:
                    self.harness.session_storage.append_sidechain_message(
                        session_id, subtask_id, msg.get("role", "system"), msg.get("content", "")
                    )
                if response:
                    self.harness.session_storage.append_sidechain_message(
                        session_id, subtask_id, "assistant", response.content or ""
                    )

            parsed_decision = self.decision.parse(
                response.content or "",
                context_pack.candidate_files,
                edit_only=True,
            )
        except TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "DecisionLLM patch generation timed out after "
                    f"{self.decision_timeout:g}s for {target_file}."
                ),
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
            user_intent=intent,
        )
        print(
            "[debug][decision-edit][validation-done] "
            f"file={target_file} success={validation.success} "
            f"elapsed={time.monotonic() - started_at:.2f}s",
            flush=True,
        )

        task_id = params.get("task_id")
        task_label = f" 任务 `{task_id}`" if task_id else ""

        if not execution.success:
            return ToolResult(
                success=False,
                output=(
                    f"❌ 【补丁生成失败】：{task_label}对文件 `{target_file}` 的补丁无法匹配到物理内容。\n"
                    f"原因分析：可能因为你在 intent 中提供的描述有误，或者你参考了过期的代码快照。\n"
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
                    f"❌ 【自动化代码验证失败】：{task_label}对文件 `{target_file}` 的补丁导致编译、语法或测试错误。\n"
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
            output=f"✅ 【应用成功】：针对文件 `{target_file}`{task_label} 的补丁已通过编译与自动化测试验证，修改已持久化提交（Committed）。",
            metadata={
                "pipeline_metrics": pipeline_metrics,
                "execution": execution,
                "validation": validation,
            }
        )
