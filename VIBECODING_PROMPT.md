# MitKII — AI Code Agent Vibecoding Prompt

> **产品定位**: 类 Claude Code 的终端 AI 编码代理。用户在 CLI 中用自然语言描述需求，
> Agent 自主完成代码搜索、文件编辑、命令执行、自我验证的完整闭环。
> 底层由 Harness 引擎提供上下文探针、Checkpoint、评分、Pipeline 编排能力。

---

## 开工前必须确定的底层架构

### 技术栈决策表

| 层级 | 选型 | 理由 | 备选 |
|------|------|------|------|
| **语言** | Python 3.12+ | LLM 生态最成熟、asyncio 原生、快速迭代 | Rust (性能极致但开发慢) |
| **CLI 框架** | Typer + Rich + Prompt Toolkit | Typer 类型安全；Rich 富文本渲染；Prompt Toolkit 做 REPL 输入 | Click + Textual |
| **LLM 调用** | LiteLLM | 统一 OpenAI/Anthropic/本地模型接口，内置 fallback/retry | 直接用各家 SDK |
| **主力模型** | Claude Sonnet 4 (coding) | 工具调用最稳、长上下文、代码理解强 | GPT-4o / DeepSeek V3 |
| **本地存储** | SQLite (via aiosqlite) | 零依赖、单文件、足够快、便携 | DuckDB |
| **向量存储** | SQLite + sqlite-vec 扩展 | 本地向量搜索，不引入额外服务 | Qdrant (如需远程) |
| **Embedding** | text-embedding-3-small (远程) / nomic-embed-text (本地) | 远程精度高；本地零成本可离线 | BGE-M3 |
| **代码解析** | Tree-sitter (via py-tree-sitter) | 增量解析、多语言、提取函数/类/import 结构 | AST 模块(仅Python) |
| **文件搜索** | ripgrep (subprocess) | 极快、尊重 .gitignore、正则支持 | grep |
| **Git 操作** | gitpython + subprocess | 高层 API + 底层命令兼顾 | dulwich (纯Python) |
| **进程管理** | asyncio.subprocess | 非阻塞命令执行、超时控制 | subprocess + threading |
| **配置格式** | TOML (tomli/tomllib) | Python 生态标准、比 YAML 不易出错 | YAML |
| **包管理** | uv | 极快的依赖安装、lockfile、虚拟环境管理 | pip + venv |
| **测试** | pytest + pytest-asyncio | 标准选择 | — |
| **发布** | PyPI (pip install mitkii) | 用户一行命令安装 | Homebrew tap |

### 关键架构决策说明

**为什么选 Python 不选 Rust/Go？**
- AI Agent 的瓶颈在 LLM API 延迟（秒级），不在本地计算
- Python 的 LLM/NLP 生态库最全（litellm, tiktoken, tree-sitter bindings）
- 开发效率是 Rust 的 5-10 倍，项目目标是 4-8 周出 MVP
- 性能敏感的部分（ripgrep/tree-sitter）已经是 Rust 编写的二进制

**为什么零外部服务依赖（无 PostgreSQL/Redis）？**
- CLI 工具的黄金法则：`pip install` 即用，不能要求用户装数据库
- SQLite 单文件搞定关系数据 + 向量索引 + 会话存储
- 和 Claude Code 一样的哲学：轻量、即开即用

**为什么自研 Agent Loop 而不用 LangGraph/AutoGen？**
- Agent Loop 是产品的核心灵魂，必须完全可控
- 框架带来的抽象层会限制探针机制的精细度
- Claude Code 级别的 Agent Loop 核心代码 < 500 行，没必要引框架
- Harness 本身就是"框架"，不需要再套一层

---

## 核心架构设计

### 1. Agent Loop — 心跳循环

这是整个系统的核心，一切都围绕这个循环：

```
┌──────────────────────────────────────────────────────────┐
│                    Agent Main Loop                         │
│                                                           │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐             │
│  │  THINK  │───▶│   ACT   │───▶│ OBSERVE  │──┐          │
│  │         │    │         │    │          │  │          │
│  │ LLM决策  │    │ 执行工具 │    │ 收集结果  │  │          │
│  └─────────┘    └─────────┘    └──────────┘  │          │
│       ▲                                       │          │
│       └───────────────────────────────────────┘          │
│                                                           │
│  退出条件: LLM 返回最终回答 (无工具调用) 或达到最大轮次      │
└──────────────────────────────────────────────────────────┘
```

```python
class AgentLoop:
    """核心 Agent 循环 — 全系统最重要的 500 行代码"""

    async def run(self, user_message: str) -> AsyncIterator[Event]:
        """一次完整的 Agent 执行"""

        # 构建初始上下文
        messages = await self.context.build(user_message)

        for turn in range(self.max_turns):
            # === THINK: 调用 LLM ===
            # Harness 探针在此拦截，管理 token 预算
            messages = await self.harness.probe.before_call(messages)

            response = await self.llm.chat(
                messages=messages,
                tools=self.tools.get_schemas(),
                stream=True,
            )

            # 流式输出思考过程
            async for chunk in response:
                yield Event(type="thinking", content=chunk)

            # === ACT: 执行工具调用 ===
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    # 权限检查
                    if await self.permissions.needs_approval(tool_call):
                        approved = yield Event(type="approval_request", tool_call=tool_call)
                        if not approved:
                            continue

                    # 执行工具 (Harness 沙盒内)
                    result = await self.harness.sandbox.execute(
                        self.tools.call, tool_call
                    )

                    yield Event(type="tool_result", result=result)

                    # === OBSERVE: 结果加入上下文 ===
                    messages.append(tool_call_message(tool_call, result))

            else:
                # 无工具调用 = Agent 给出最终回答
                yield Event(type="final_answer", content=response.content)
                break

            # Harness: 每轮结束后的钩子
            await self.harness.probe.after_call(response)
            await self.harness.checkpoint.auto_save_if_needed(messages)

            # Harness: 评分检查 (如果刚完成代码修改)
            if self._just_edited_code():
                score = await self.harness.scorer.evaluate()
                if score.needs_retry:
                    messages.append(system_message(
                        f"你的修改未通过验证: {score.feedback}。请修正。"
                    ))
```

### 2. Harness 引擎 — 四大核心能力

```
┌─────────────────────── Harness Engine ───────────────────────┐
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐   │
│  │   Probe     │  │ Checkpoint  │  │     Scorer         │   │
│  │  (探针)     │  │  (检查点)   │  │    (评分器)        │   │
│  │             │  │             │  │                    │   │
│  │ ·token预算  │  │ ·状态快照   │  │ ·L0 程序化门控    │   │
│  │ ·动态裁剪   │  │ ·git stash  │  │ ·L1 LLM红线Rubric │   │
│  │ ·用量监控   │  │ ·会话存档   │  │ ·L2 质量建议(可选)│   │
│  │ ·过载告警   │  │ ·回滚恢复   │  │ ·rubrics/*.yaml   │   │
│  └─────────────┘  └─────────────┘  └────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Pipeline                              │ │
│  │  DAG 执行 · 条件分支 · 并行 Stage · 失败重试 · 钩子     │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

#### 2.1 探针 (Probe) — 上下文守门人

```python
class ContextProbe:
    """
    类比:
    - Java Agent 通过 Instrumentation API 拦截字节码加载
    - ContextProbe 通过拦截 LLM 调用管理上下文窗口

    核心职责: 确保每次 LLM 调用都在 token 预算内，
    且上下文质量最高（相关信息优先，噪声被裁掉）
    """

    def __init__(self, config: ProbeConfig):
        self.max_tokens = config.max_context_tokens  # 如 128K
        self.budget_ratio = config.budget_ratio      # 如 0.75 (留 25% 给输出)
        self.budget = int(self.max_tokens * self.budget_ratio)

    async def before_call(self, messages: list[Message]) -> list[Message]:
        """LLM 调用前拦截"""
        current_tokens = self.count_tokens(messages)

        if current_tokens <= self.budget:
            return messages  # 安全，直接放行

        # 超预算 → 启动裁剪策略链
        messages = await self._apply_trim_strategies(messages, current_tokens)
        return messages

    async def _apply_trim_strategies(self, messages, current_tokens):
        """裁剪策略优先级（从温和到激进）"""

        # 策略1: 压缩早期对话轮次为摘要
        if current_tokens > self.budget:
            messages = await self._summarize_old_turns(messages)

        # 策略2: 截断过长的工具返回结果
        if self.count_tokens(messages) > self.budget:
            messages = self._truncate_tool_outputs(messages, max_per_tool=2000)

        # 策略3: 移除已不相关的文件内容
        if self.count_tokens(messages) > self.budget:
            messages = await self._evict_irrelevant_files(messages)

        # 策略4: 硬截断最早的历史 (最后手段)
        if self.count_tokens(messages) > self.budget:
            messages = self._hard_truncate(messages)

        return messages
```

#### 2.2 Checkpoint — 状态快照与回滚

```python
class CheckpointStore:
    """
    Checkpoint 触发时机:
    1. 用户发送新消息时 (保存上一轮完整状态)
    2. 执行危险操作前 (git stash + 状态快照)
    3. 多步骤任务的关键节点
    4. 手动触发 (/checkpoint 命令)

    存储内容:
    - 对话历史 (messages)
    - 文件变更列表 (哪些文件被改了)
    - git diff (可回滚的代码变更)
    - 记忆状态 (working memory snapshot)
    - Agent 内部状态 (当前 plan、进度)
    """

    async def save(self, trigger: str, state: AgentState) -> str:
        snapshot = Snapshot(
            id=generate_id(),
            trigger=trigger,
            timestamp=now(),
            messages=state.messages,
            file_changes=state.file_tracker.get_changes(),
            git_patch=await self._create_git_patch(),
            memory_snapshot=state.memory.export(),
            plan_state=state.planner.export() if state.planner else None,
        )
        await self.store.save(snapshot)
        return snapshot.id

    async def rollback(self, checkpoint_id: str) -> None:
        """回滚: 恢复代码 + 恢复对话状态"""
        snapshot = await self.store.load(checkpoint_id)

        # 恢复文件变更
        await self._apply_reverse_patch(snapshot.git_patch)

        # 恢复 Agent 状态
        return snapshot
```

#### 2.3 评分器 (Scorer) — 代码变更质量门

**设计原则**: 能程序化判定的绝不交给 LLM；LLM 只做「意图/范围/安全」等难以自动化的**二值红线**判定，且必须按 Rubric 表逐条举证，禁止开放式「整体感觉不错」或 1–10 分。

##### 三层评分架构

```
┌─────────────────────────────────────────────────────────────┐
│ L0 硬门控 (程序判定，一票否决，失败则不调用 LLM)              │
│  lint · typecheck · tests · file_guard · resource_limit      │
└─────────────────────────────────────────────────────────────┘
                              ↓ 全过
┌─────────────────────────────────────────────────────────────┐
│ L1 LLM 红线 Rubrics (结构化 JSON，任意 blocker → Fail)       │
│  任务完成度 · 修改范围 · 安全 · Harness 策略 · rules.md      │
└─────────────────────────────────────────────────────────────┘
                              ↓ 全过
┌─────────────────────────────────────────────────────────────┐
│ L2 质量建议 (可选，默认 warning 不阻断)                       │
│  可读性 · 命名 · 过度抽象等                                   │
└─────────────────────────────────────────────────────────────┘
```

| 层级 | 实现模块 | 谁判定 | 失败后果 |
|------|----------|--------|----------|
| L0 | `code_quality.py`, `test_runner.py`, `sandbox/` | 工具/脚本 | `needs_retry=True`，跳过 L1 |
| L1 | `task_completion.py` + `rubrics/*.yaml` | LLM 裁判（按表填表） | 任意 `severity=blocker` 未过 → `needs_retry=True` |
| L2 | 同 `task_completion.py` 或独立 `quality_hints.py` | LLM 可选 | 仅写入 `warnings`，默认不阻断 |

**与文档末尾「红线约束」的关系**: 产品级红线（零外部服务、危险操作确认等）在 **运行时/沙盒** 强制保证；其中无法在沙盒层完全覆盖的（如「修改是否越界」）写入 L1 Rubric 的 `HP-*` 条目。项目级规范来自 `.mitkii/rules.md`，启动时合并进 `rubrics/project.yaml`。

##### Rubric 条目 schema（YAML）

每条 Rubric 必须可二值判定，字段固定：

```yaml
# src/harness/scorer/rubrics/default.yaml 示例
rubrics:
  - id: TC-01
    category: task_completion
    severity: blocker
    question: "用户明确要求的行为是否都在 diff 中有对应实现？"
    pass_criteria: "每条用户诉求都能在变更中找到可指认的实现（含新增测试若用户要求）"
    fail_criteria: "遗漏任一明确诉求，或实现与诉求语义不符"
    evidence_required: ["user_message", "diff"]

  - id: SC-02
    category: scope
    severity: blocker
    question: "变更是否严格在用户描述范围内？"
    pass_criteria: "仅修改完成任务所必需的文件与逻辑"
    fail_criteria: "无关重构、顺手全文件格式化、未请求的依赖升级/重命名"
    evidence_required: ["user_message", "diff", "file_list"]

  - id: SEC-01
    category: security
    severity: blocker
    question: "是否引入不可接受的安全风险？"
    pass_criteria: "无硬编码密钥、无关闭鉴权、无对不可信输入 eval/exec"
    fail_criteria: "出现上述任一情况"
    evidence_required: ["diff"]

  - id: HP-03
    category: harness_policy
    severity: blocker
    question: "危险操作是否已走确认流程？"
    pass_criteria: "delete / 破坏性 shell / git push 均有 approval 记录或用户显式授权"
    fail_criteria: "存在危险操作但上下文无确认痕迹"
    evidence_required: ["tool_call_log", "diff"]

  - id: QL-01
    category: quality
    severity: warning
    question: "是否存在明显可读性问题？"
    pass_criteria: "命名与结构可理解，无大面积重复"
    fail_criteria: "严重影响维护（仅建议，默认不阻断）"
    evidence_required: ["diff"]
```

`mitkii init` 时若存在 `.mitkii/rules.md`，解析为额外 `blocker` 条目追加到 `rubrics/project.yaml`（`id` 前缀 `PRJ-`）。

##### LLM 裁判协议（强制结构化输出）

裁判 **不是** `_llm_self_evaluate("改好了吗")`，而是 `LLMJudge.evaluate(context, rubrics)`：把 `user_message`、`git diff`、`tool 调用摘要`、`rules.md` 摘要与 **完整 Rubric 表** 注入 system prompt，要求对 **每一条** 给出 pass/fail + 证据引用。

```json
{
  "verdict": "fail",
  "results": [
    {
      "id": "TC-01",
      "passed": false,
      "severity": "blocker",
      "reason": "用户要求处理空列表分支，diff 中未体现",
      "evidence": ["user: 空列表要返回 []", "diff: src/foo.py 无相关 hunk"]
    }
  ],
  "blockers": ["TC-01"],
  "warnings": []
}
```

解析失败或缺少任一 `blocker` 条目的 `evidence` → 视为 **Fail**（宁可误杀不可漏放）。

##### 实现骨架

```python
# src/harness/scorer/engine.py
class ScoringEngine:
    """
    Agent 每次修改代码后分层评分:
    L0 → (可选) L1 Rubric 裁判 → (可选) L2 建议

    Pass → 继续 or 返回结果
    Fail → feedback 含失败 Rubric id + 证据 → Agent 自动修正 (最多 max_retries 次)
    """

    async def evaluate(self, context: ScoringContext) -> ScoreResult:
        scores: dict[str, LayerScore] = {}

        # === L0: 程序化门控 (无 LLM) ===
        if context.language_supported:
            scores["lint"] = await self.code_quality.run_lint(context)
            if not scores["lint"].passed:
                return self._fail_fast(scores, layer="L0")

        if context.has_related_tests:
            scores["tests"] = await self.test_runner.run(context)
            if not scores["tests"].passed:
                return self._fail_fast(scores, layer="L0")

        # === L1: LLM 红线 Rubric 裁判 ===
        rubrics = self.rubric_loader.load(
            default="rubrics/default.yaml",
            project=context.project_rubrics_path,  # .mitkii/rules.md 合并结果
        )
        l1 = await self.task_completion.judge(context, rubrics=rubrics)
        scores["rubrics"] = l1
        if l1.blockers:
            return ScoreResult(
                passed=False,
                scores=scores,
                feedback=self._format_rubric_feedback(l1),
                needs_retry=True,
            )

        # === L2: 质量建议 (默认不阻断) ===
        warnings = [r for r in l1.results if r.severity == "warning" and not r.passed]
        return ScoreResult(
            passed=True,
            scores=scores,
            feedback=self._format_warnings(warnings) if warnings else None,
            needs_retry=False,
        )


# src/harness/scorer/task_completion.py
class LLMJudge:
    """按 Rubric 表逐条判定，禁止自由发挥总分。"""

    async def judge(self, context: ScoringContext, rubrics: list[Rubric]) -> RubricVerdict:
        prompt = self._build_judge_prompt(
            user_message=context.user_message,
            diff=context.diff,
            tool_log=context.recent_tool_calls,
            project_rules=context.rules_md_summary,
            rubrics=rubrics,
        )
        raw = await self.llm.chat(
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, prompt],
            response_format={"type": "json_object"},
        )
        verdict = RubricVerdict.parse(raw)  # 校验 schema；缺 evidence → Fail
        return verdict
```

`JUDGE_SYSTEM_PROMPT` 要点（写入 `task_completion.py` 或 `rubrics/judge_prompt.md`）:

1. 你是 **裁判**，不是编码助手；只输出 JSON，不修改代码。
2. 对 Rubric 表中 **每一条** 独立判定 `passed: true/false`。
3. `severity=blocker` 且 `passed=false` → 整体验证失败。
4. 每条判定必须包含 `evidence`，引用 `user:` / `diff:` / `tool:` 片段。
5. 不得用 1–10 分、不得用「总体尚可」代替逐条结论。
6. L0 已覆盖项（lint 错误、测试失败）不要重复判 Fail，除非 Rubric 明确要求复核语义。

##### 目录与文件

```
src/harness/scorer/
├── engine.py
├── code_quality.py      # L0
├── test_runner.py       # L0
├── task_completion.py   # L1 LLMJudge
├── rubric_loader.py     # 加载 default + project，合并 rules.md
├── rubrics/
│   ├── default.yaml     # 内置红线表
│   ├── judge_prompt.md  # 裁判 system prompt 模板
│   └── schema.json      # RubricVerdict JSON Schema (校验用)
└── feedback.py          # 将 blockers 格式化为 Agent 可执行的修正指引
```

Agent Loop 收到失败反馈时的 system 消息格式:

```
你的修改未通过 L1 红线评分。请仅针对下列条目修正，不要扩大范围:
- [TC-01] 用户要求处理空列表分支，diff 中未体现
  证据: user: ... | diff: ...
```

#### 2.4 Pipeline — 复杂任务编排

```python
class Pipeline:
    """
    用于复杂多步骤任务的 DAG 编排。
    简单任务不需要 Pipeline，直接在 Agent Loop 里完成。
    复杂任务 (如 "重构这个模块的所有文件") 启用 Pipeline。

    示例 Pipeline:
    IndexProject → AnalyzeStructure → PlanRefactor → [EditFile1 || EditFile2 || ...] → RunTests → Report
    """

    def define(self) -> PipelineDefinition:
        return (
            Pipeline("refactor")
            .stage("analyze", AnalyzeStage())
            .stage("plan", PlanStage(), depends_on=["analyze"])
            .stage("edit", EditStage(), depends_on=["plan"], parallel=True)
            .stage("verify", VerifyStage(), depends_on=["edit"])
            .on_failure("edit", retry=2, then="rollback")
        )
```

### 3. 上下文工程 — 最核心的差异化

```python
class ContextBuilder:
    """
    上下文 = System Prompt + 项目信息 + 对话历史 + 文件内容 + 工具结果

    核心原则:
    - 相关性 > 完整性 (不是塞越多越好)
    - 结构化 > 原始文本 (用 XML tags 划分区域)
    - 动态 > 静态 (根据当前任务动态加载)
    """

    async def build(self, user_message: str) -> list[Message]:
        messages = []

        # 1. System Prompt (固定 + 项目规则)
        system = self._build_system_prompt()
        if self.project_rules:
            system += f"\n\n<project_rules>\n{self.project_rules}\n</project_rules>"
        messages.append(SystemMessage(system))

        # 2. 项目上下文 (结构摘要、最近修改文件)
        project_context = await self._build_project_context()
        messages.append(SystemMessage(project_context))

        # 3. 对话历史 (可能被 Probe 压缩)
        messages.extend(self.conversation_history)

        # 4. 当前用户消息 + 自动附加的相关文件
        relevant_files = await self._find_relevant_files(user_message)
        enriched_message = self._enrich_with_files(user_message, relevant_files)
        messages.append(UserMessage(enriched_message))

        return messages

    async def _find_relevant_files(self, message: str) -> list[FileContext]:
        """基于用户消息智能判断需要加载哪些文件"""
        # 1. 显式提到的文件路径
        explicit = self._extract_file_paths(message)

        # 2. 语义搜索相关文件
        semantic = await self.indexer.search(message, top_k=5)

        # 3. 当前打开/最近编辑的文件 (来自 file_tracker)
        recent = self.file_tracker.get_recent(limit=3)

        # 去重 + 按相关性排序
        return self._dedupe_and_rank(explicit + semantic + recent)
```

### 4. 记忆系统 — 三层架构

```
┌───────────────────────────────────────────────────────┐
│  L1: Working Memory (当前会话)                         │
│  - 完整对话历史                                        │
│  - 当前任务的 plan 和进度                              │
│  - 本次修改过的文件列表                                │
│  生命周期: 会话结束后压缩为摘要存入 L3                  │
│  存储: 内存                                           │
├───────────────────────────────────────────────────────┤
│  L2: Project Memory (项目级)                           │
│  - 项目结构摘要 (目录树 + 关键文件说明)                │
│  - .mitkii/rules.md (用户定义的编码规范)               │
│  - 代码索引 (符号表 + 依赖关系)                        │
│  生命周期: 跟随项目，增量更新                           │
│  存储: .mitkii/index.db (SQLite)                      │
├───────────────────────────────────────────────────────┤
│  L3: Long-term Memory (跨项目)                         │
│  - 历史会话摘要 ("上次帮用户做了XX重构")                │
│  - 用户偏好 (编码风格、常用框架、沟通方式)             │
│  - 常见错误模式 (用户容易犯的错)                       │
│  生命周期: 永久，容量超限时淘汰最旧                     │
│  存储: ~/.mitkii/memory.db (SQLite)                   │
└───────────────────────────────────────────────────────┘
```

### 5. 工具系统设计

```python
class Tool(ABC):
    """工具基类 — 所有工具继承此类"""

    name: str                    # 工具名 (如 "read_file")
    description: str             # 给 LLM 看的描述
    parameters: dict             # JSON Schema 参数定义
    risk_level: RiskLevel        # safe / moderate / dangerous

    @abstractmethod
    async def execute(self, **params) -> ToolResult:
        """执行工具逻辑"""
        ...

    def to_schema(self) -> dict:
        """转为 LLM function calling schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolRegistry:
    """工具注册表 — Agent 只能调用已注册的工具"""

    def register(self, tool: Tool) -> None: ...
    def get_schemas(self) -> list[dict]: ...
    async def call(self, name: str, params: dict) -> ToolResult: ...
```

**工具清单与风险分级:**

| 工具 | 风险等级 | 是否需确认 | 说明 |
|------|---------|-----------|------|
| read_file | safe | 否 | 读文件 |
| grep_search | safe | 否 | 搜索代码 |
| glob_files | safe | 否 | 查找文件 |
| list_dir | safe | 否 | 列目录 |
| write_file | moderate | 新文件否/覆盖是 | 写文件 |
| edit_file | moderate | 否 (有 stash 兜底) | 编辑文件 |
| delete_file | dangerous | 是 | 删文件 |
| shell_exec | dangerous | 是 (可配白名单) | 执行命令 |
| git_commit | moderate | 是 | Git 提交 |
| git_push | dangerous | 是 | 推送远程 |

### 6. 权限系统

```python
class PermissionManager:
    """
    三级权限模型:
    1. auto_allow — 安全操作，直接执行
    2. ask_once — 首次确认，同类操作后续自动放行
    3. always_ask — 每次都要确认

    用户可通过 config 自定义规则:
    [permissions]
    shell_commands = ["npm test", "cargo build"]  # 白名单命令免确认
    auto_approve_edits = true                     # 编辑文件免确认
    never_auto = ["git push", "rm -rf"]           # 永远要确认
    """

    async def check(self, action: Action) -> PermissionResult:
        # 1. 检查白名单
        if self._in_whitelist(action):
            return PermissionResult.ALLOWED

        # 2. 检查风险等级
        match action.risk_level:
            case RiskLevel.SAFE:
                return PermissionResult.ALLOWED
            case RiskLevel.MODERATE:
                if self.config.auto_approve_moderate:
                    return PermissionResult.ALLOWED
                return PermissionResult.NEEDS_APPROVAL
            case RiskLevel.DANGEROUS:
                return PermissionResult.NEEDS_APPROVAL
```

---

## 用户体验设计

### CLI 交互流程

```
$ mitkii

  ╭─ MitKII v0.1.0 ─────────────────────────────────────────╮
  │  AI Code Agent · 输入需求，我来写代码                      │
  │  /help 查看命令 · /checkpoint 保存状态 · Ctrl+C 退出     │
  ╰──────────────────────────────────────────────────────────╯

  Project: harness-mitkii (Python, 23 files indexed)

> 帮我实现一个 retry 装饰器，支持指数退避和最大重试次数

  ● Thinking...

  我来实现这个 retry 装饰器。让我先看看项目中是否已有相关代码。

  ┌─ grep_search ──────────────────────────────────┐
  │ pattern: "retry|retries|backoff"               │
  │ 3 results in 2 files                           │
  └────────────────────────────────────────────────┘

  项目中 src/harness/retry.py 已有基础实现，我在此基础上增强。

  ┌─ edit_file ────────────────────────────────────┐
  │ path: src/harness/retry.py                     │
  │ + @overload                                    │
  │ + def retry(                                   │
  │ +     max_attempts: int = 3,                   │
  │ +     backoff: BackoffStrategy = exponential(), │
  │ +     ...                                      │
  │ (展示 diff)                                    │
  └────────────────────────────────────────────────┘

  ┌─ shell_exec ───────────────────────────────────┐
  │ ⚠ 需要确认: pytest tests/test_retry.py        │
  │ [y/n/always]:                                  │
  └────────────────────────────────────────────────┘

> y

  ✓ 4 tests passed

  完成! 已实现支持指数退避的 retry 装饰器，并通过了现有测试。
```

### 内置命令

| 命令 | 作用 |
|------|------|
| `/help` | 显示帮助 |
| `/checkpoint` | 手动保存检查点 |
| `/rollback` | 回滚到上一个检查点 |
| `/history` | 查看本次会话历史 |
| `/compact` | 手动压缩上下文 |
| `/plan` | 让 Agent 只输出计划不执行 |
| `/score` | 手动触发代码质量评分 |
| `/clear` | 清空当前会话 |
| `/config` | 查看/修改配置 |
| `/cost` | 查看本次会话 token 消耗和费用 |

---

## 开发阶段规划

### Phase 1: 骨架 (Week 1)

**目标**: 最小 Agent Loop 跑通——能对话、能调用一个工具

```
必须完成:
□ CLI 骨架 (Typer 入口 + Rich 渲染 + REPL 循环)
□ LiteLLM 集成 (能调通 Claude/GPT)
□ Agent Loop 基础版 (Think → Act → Observe 循环)
□ 1 个工具: read_file
□ 流式输出 (逐 token 打印)

验证标准:
- 输入 "读一下 README.md"，Agent 调用 read_file 并返回内容
- 输出逐字流式显示
```

### Phase 2: 工具矩阵 (Week 2)

**目标**: 完整工具集，Agent 能实际修改代码

```
必须完成:
□ 文件工具: read / write / edit / create / delete
□ 搜索工具: grep / glob / list_dir
□ Shell 工具: exec (带超时和输出捕获)
□ Git 工具: status / diff / commit / stash
□ 权限系统: risk level + 确认交互
□ Tool Registry 统一管理

验证标准:
- 输入 "在 src/utils/ 下创建一个 helpers.py，写一个 flatten 函数"
- Agent 能: 列目录确认路径 → 创建文件 → 写入代码
- 危险操作弹出确认
```

### Phase 3: Harness 探针 + Checkpoint (Week 3-4)

**目标**: 上下文不爆、状态可恢复

```
必须完成:
□ Token 计数 (tiktoken)
□ Context Probe 拦截器 (before_call / after_call)
□ 裁剪策略: 历史摘要 / 工具输出截断 / 文件驱逐 / 硬截断
□ Checkpoint 快照: 对话 + 文件变更 + git patch
□ 回滚机制: /rollback 命令
□ 用量统计: /cost 命令

验证标准:
- 连续对话 50 轮不崩 (上下文被正确裁剪)
- /rollback 能恢复代码和对话到之前状态
- /cost 显示准确的 token 消耗
```

### Phase 4: 代码索引 + 语义搜索 (Week 5)

**目标**: Agent 能理解整个项目结构

```
必须完成:
□ 项目扫描 (文件树 + .gitignore)
□ Tree-sitter 解析 (提取函数/类/import)
□ Embedding 生成 (增量更新)
□ SQLite-vec 向量搜索
□ 语义搜索工具: semantic_search
□ 项目上下文自动注入 (目录结构摘要)

验证标准:
- 新 clone 一个项目，mitkii init 后能建立索引
- 输入 "数据库连接在哪配置的"，能找到正确文件
- 索引增量更新 (文件改了不用全量重建)
```

### Phase 5: 评分 + 自我纠错 (Week 6)

**目标**: Agent 改完代码会自己验证；LLM 裁判按 Rubric 红线表判定，结果可复现、可审计

```
必须完成:
□ L0 门控: Lint 集成 (检测项目用的 linter 并运行)
□ L0 门控: Test runner (自动发现并运行相关测试)
□ L1 裁判: rubrics/default.yaml 内置红线表 (TC/SC/SEC/HP 等 blocker)
□ L1 裁判: LLMJudge 结构化 JSON 输出 + schema 校验 (缺 evidence 视为 Fail)
□ L1 裁判: .mitkii/rules.md → 合并为 project rubrics (PRJ-* 前缀)
□ rubric_loader + feedback 格式化 (失败时反馈 Rubric id + 证据)
□ 分层短路: L0 失败不调 LLM；L1 任一 blocker 失败 → needs_retry
□ 自动重试 (最多 3 次，反馈须指向具体 Rubric id)
□ /score 手动评分命令 (输出每层结果 + 逐条 Rubric 判定)

验证标准:
- lint/test 失败时不会调用 LLM 裁判 (省钱且避免结论冲突)
- 故意遗漏用户诉求时 TC-01 必须 Fail，且 feedback 含 TC-01 与证据
- 无关文件被修改时 SC-02 必须 Fail
- 最多重试 3 次，仍失败则向用户报告未通过的 Rubric id 列表
```

### Phase 6: 记忆 + 个性化 (Week 7)

**目标**: 越用越懂项目、越懂用户

```
必须完成:
□ Project Memory (代码规范记忆、项目架构摘要)
□ .mitkii/rules.md 支持 (用户自定义规则)
□ Long-term Memory (历史会话摘要)
□ 会话恢复 (mitkii resume)
□ 记忆检索 (新任务时自动回忆相关历史)

验证标准:
- 定义 rules.md "使用 4 空格缩进"，Agent 写代码遵守
- 第二天打开同项目，Agent 记得昨天做了什么
```

### Phase 7: Pipeline + 复杂任务 (Week 8)

**目标**: 能处理 "重构整个模块" 级别的大任务

```
必须完成:
□ Task Planner (大任务分解为子步骤)
□ Pipeline DAG 执行 (并行编辑多文件)
□ 进度展示 (当前步骤 / 总步骤)
□ 部分失败恢复 (某个文件编辑失败不影响其他)

验证标准:
- 输入 "把所有 print 替换为 logger.info"
- Agent: 搜索所有文件 → 生成 plan → 并行修改 → 验证 → 报告结果
```

### Phase 8: Server 模式 — GUI 接入层 (Week 9-10)

**目标**: Agent 引擎暴露为独立服务，任何 GUI 都能接入

```
必须完成:
□ 定义 Agent Protocol (JSON-RPC 2.0 消息格式)
□ mitkii serve --stdio 模式 (VS Code Extension 用)
□ mitkii serve --port 模式 (Web IDE / Electron 用)
□ Event 流标准化 (所有 Agent 输出都是结构化 Event)
□ 请求类型: sendMessage / editSection / getCompletion / acceptDiff / rejectDiff / rollback
□ 会话管理: 多会话并发、会话恢复

协议示例:
  → {"jsonrpc":"2.0","method":"sendMessage","params":{"text":"加个错误处理"},"id":1}
  ← {"jsonrpc":"2.0","result":{"type":"stream_start"},"id":1}
  ← {"jsonrpc":"2.0","method":"event","params":{"type":"thinking","content":"让我看看..."}}
  ← {"jsonrpc":"2.0","method":"event","params":{"type":"tool_call","tool":"edit_file","diff":{...}}}
  ← {"jsonrpc":"2.0","method":"event","params":{"type":"approval_request","action":"shell_exec"}}
  → {"jsonrpc":"2.0","method":"approvalResponse","params":{"approved":true},"id":2}
  ← {"jsonrpc":"2.0","method":"event","params":{"type":"done","summary":"已添加错误处理"}}

验证标准:
- 用一个简单的 Python 脚本能连上 stdio 模式并完成一轮对话
- Event 流包含足够信息让前端渲染 diff、进度、评分
- CLI 模式和 Server 模式共用同一个 Agent 引擎 (零重复代码)
```

### Phase 9: VS Code Extension (Week 11-13)

**目标**: 完整 IDE 内 AI 编码体验——边写代码边用 AI

```
必须完成:
□ Extension 骨架 (TypeScript, vsce 打包)
□ 侧边栏对话面板 (Webview, 类 Cursor 的 chat panel)
□ Inline Diff 展示 (Agent 修改代码 → 编辑器内绿红高亮 → 接受/拒绝)
□ 右键上下文菜单 (选中代码 → Explain / Refactor / Fix / Test)
□ Agent 进度指示 (Status Bar + 进度通知)
□ 文件变更预览 (打开 Diff Editor 展示变更)
□ 权限确认 (VS Code 通知弹窗替代终端 y/n)
□ 自动 Stash 提示 (修改前显示 "将创建安全快照")

可选增强:
□ Tab 补全 (InlineCompletionItemProvider, 单独的快速模型)
□ 错误自动检测 (监听 Diagnostics → 自动建议修复)
□ 多文件变更面板 (类 Source Control 视图，列出所有 AI 修改)
□ Checkpoint 历史面板 (可视化回滚点)

技术方案:
- Extension 通过 spawn 启动 `mitkii serve --stdio` 子进程
- 通信用 JSON-RPC over stdin/stdout (和 LSP 同架构)
- Webview 用 React 写对话面板 UI
- Diff 展示用 VS Code 原生 Decoration API

验证标准:
- 在侧边栏输入 "给这个函数加单测"，Agent 生成测试文件，编辑器弹出 diff
- 点击 Accept → 文件保存；点击 Reject → 恢复原样
- 选中代码右键 "Explain" → 侧边栏显示解释
```

### Phase 10 (可选): Electron 独立 IDE (远期)

**如果要做独立产品（类 Cursor），在 Phase 9 验证完后考虑：**

```
架构:
- Electron 主进程 + Monaco Editor (渲染进程)
- MitKII Agent Server (独立进程, JSON-RPC)
- 相比 VS Code Extension 的额外工作:
  □ 文件树面板
  □ 终端模拟器
  □ 语法高亮 / LSP 集成
  □ 快捷键系统
  □ 主题系统
  □ 自动更新

工期预估: 额外 2-3 个月
是否值得: 取决于是否要做产品化/创业，面试展示不需要走到这步
```

---

## 架构预留：GUI 兼容设计原则

为了让 Phase 8-9 顺利推进，Phase 1-7 编码时必须遵守:

**1. Agent 输出全部走 Event 流，禁止直接 print**

```python
# ❌ 错误: 直接打印，GUI 无法解析
print("正在搜索相关文件...")

# ✅ 正确: 结构化 Event，CLI 和 GUI 都能消费
yield AgentEvent(type=EventType.STATUS, message="正在搜索相关文件...")
```

**2. CLI 是 Event 的一个消费者，不是唯一出口**

```python
# src/cli/renderer.py 负责把 Event 渲染成终端输出
# src/server/handler.py 负责把 Event 序列化为 JSON-RPC 发给 GUI
# 两者消费同一个 Event 流，Agent 引擎不感知谁在用它
```

**3. 权限确认是异步回调，不是阻塞 input()**

```python
# ❌ 错误: 阻塞式确认，GUI 无法介入
answer = input("确认执行? [y/n]: ")

# ✅ 正确: yield 一个确认请求，等待外部回复
approval = yield AgentEvent(type=EventType.APPROVAL_REQUEST, action=action)
# CLI 模式: renderer 弹出 prompt_toolkit 确认
# GUI 模式: Extension 弹出 VS Code 通知
```

**4. 文件变更以 Diff 结构输出，不是直接写入**

```python
# Agent 产出的是 "变更意图"，由外层决定是直接应用还是等用户确认
yield AgentEvent(
    type=EventType.FILE_EDIT,
    path="src/main.py",
    diff=UnifiedDiff(hunks=[...]),  # 结构化 diff
    auto_apply=self.config.auto_approve_edits,  # CLI 可能自动应用
)
# GUI 模式下: Extension 收到 diff → 展示 inline decoration → 等 accept/reject
```

---

## 红线约束

1. **零外部服务依赖** — `pip install mitkii` 即用，不能要求用户装 Docker/PostgreSQL/Redis
2. **上下文绝不静默溢出** — Probe 必须在爆之前裁剪，宁可降级不能报错
3. **危险操作必须确认** — delete / shell / git push 必须用户同意
4. **编辑必须有回退** — 每次编辑前自动 git stash 或 checkpoint
5. **流式输出必须真实** — 不是攒完再吐，是 LLM 出一个 token 就显示一个
6. **Token 消耗透明** — 用户随时能看到花了多少 token/多少钱
7. **本地优先** — 代码不上传到任何服务器（除了发给 LLM 的部分），索引存本地

---

## 参考实现

学习对象 (开源):
- [aider](https://github.com/paul-gauthier/aider) — Python CLI code agent, 架构清晰
- [mentat](https://github.com/AbanteAI/mentat) — Python CLI, 上下文管理值得参考
- [continue](https://github.com/continuedev/continue) — IDE 插件, 工具设计参考
- [plandex](https://github.com/plandex-ai/plandex) — Go, 多步骤 plan 执行

不学什么:
- 不学 LangChain 的过度抽象
- 不学 AutoGPT 的无限循环
- 不学 GPT-Engineer 的一次性生成 (没有迭代能力)

---

## 给 AI 编码助手的指令

开始编码时:

1. **先写 `src/agent/loop.py`** — 这是灵魂，先确保 Think→Act→Observe 循环跑通
2. **再写 `src/tools/`** — 从 read_file 开始，逐个加工具
3. **然后 `src/harness/probe/`** — 探针是核心差异化，越早加越好
4. **接着 `src/harness/checkpoint/`** — 有了安全网才敢让 Agent 改代码
5. **然后 `src/context/`** — 上下文构建质量决定 Agent 智商
6. **再做 `src/indexer/`** — 代码索引让 Agent 理解项目
7. **最后 `src/harness/scorer/`** — 先 L0 门控，再 `rubrics/default.yaml` + `LLMJudge` 红线裁判，评分闭环才可依赖

每完成一层，确保能跑一个端到端 demo，不要攒到最后才联调。

---

*END OF VIBECODING PROMPT*
