# MitKII — AI Code Agent (类 Claude Code) 项目结构

## 产品定位

终端驱动的 AI 编码代理，用户在 CLI 中描述需求，Agent 自主规划、搜索代码、编辑文件、
执行命令、自我验证，完成多步骤编码任务。底层由 Harness 引擎提供上下文管理、
Checkpoint、评分、Pipeline 编排能力。

**手脑分离**：Planner / Executor（LLM）只负责推理与工具决策；Harness 负责 prompt 拼接、
上下文压缩、工具 I/O 管道、跨 subtask handoff 与评价门禁。

---

## 顶层目录

```
harness-mitkii/
├── src/
│   ├── cli/                    # CLI 入口与交互层
│   ├── agent/                  # Agent 核心逻辑
│   ├── harness/                # Harness 引擎 (探针/Checkpoint/评分/Pipeline)
│   ├── tools/                  # 工具集 (文件/搜索/Shell/Git...)
│   ├── context/                # 上下文工程
│   ├── memory/                 # 记忆系统
│   ├── planner/                # 任务规划与分解
│   ├── indexer/                # 代码库索引
│   ├── server/                 # Server 模式 (JSON-RPC, GUI 接入层)
│   └── config/                 # 配置管理
├── extensions/
│   └── vscode/                 # VS Code Extension (TypeScript)
├── tests/                      # 测试
├── benchmarks/                 # 性能与质量基准测试
├── prompts/                    # Prompt 模板库
├── docs/                       # 文档
├── pyproject.toml              # Python 项目配置
├── VIBECODING_PROMPT.md        # 架构总纲 Prompt
└── README.md
```

---

## 详细结构

### src/cli/ — CLI 交互层

```
src/cli/
├── __init__.py
├── app.py                      # 主入口 (Typer/Click)
├── repl.py                     # REPL 交互循环
├── renderer.py                 # 终端富文本渲染 (Rich)
├── spinner.py                  # 加载动画与进度
├── permissions.py              # 操作确认交互 (y/n/always)
├── theme.py                    # 终端主题配色
└── commands/
    ├── chat.py                 # 主对话命令 (mitkii chat)
    ├── init.py                 # 项目初始化 (mitkii init)
    ├── resume.py               # 恢复会话 (mitkii resume)
    ├── history.py              # 查看历史 (mitkii history)
    └── config.py               # 配置管理 (mitkii config)
```

### src/agent/ — Agent 核心

```
src/agent/
├── __init__.py
├── loop.py                     # Agent 主循环 (Think → Act → Observe)
├── router.py                   # 意图路由 (直接回答 vs 需要工具)
├── executor.py                 # 工具调用执行器
├── stream.py                   # 流式输出管理
├── error_recovery.py           # 错误恢复与自动纠错
├── self_correction.py          # 自我评估与修正 (改完代码跑测试)
└── types.py                    # Agent 相关类型定义
```

### src/harness/ — Harness 引擎 (核心基础设施)

```
src/harness/
├── __init__.py
├── engine.py                   # Harness 运行时主入口
│
├── pipeline/                   # Pipeline 编排
│   ├── __init__.py
│   ├── definition.py           # Pipeline DAG 定义
│   ├── executor.py             # Pipeline 执行器
│   ├── stage.py                # Stage 抽象 (顺序/并行/条件)
│   └── hooks.py                # 生命周期钩子 (before/after stage)
│
├── probe/                      # 上下文探针 (统一 LLM 前裁剪入口)
│   ├── interceptor.py          # before_call 通用 trim；Executor 先走 subtask digest
│   ├── token_budget.py         # Token 预算管理
│   ├── trimmer.py              # 动态裁剪策略
│   └── metrics.py              # 用量指标采集
│
├── checkpoint/                 # Checkpoint 系统
│   ├── __init__.py
│   ├── store.py                # Checkpoint 持久化
│   ├── snapshot.py             # 状态快照 (对话+文件变更+记忆)
│   ├── rollback.py             # 回滚机制
│   └── diff.py                 # 快照对比
│
├── scorer/                     # 评分引擎 (L0 程序化 + L1 Rubric 裁判)
│   ├── __init__.py
│   ├── engine.py               # 分层评分编排 (L0 短路 → L1 → L2)
│   ├── code_quality.py         # L0: lint/type/complexity
│   ├── test_runner.py          # L0: 测试执行与判定
│   ├── task_completion.py      # L1: LLMJudge 按 Rubric 表逐条判定
│   ├── rubric_loader.py        # 加载 default.yaml + project rules 合并
│   ├── feedback.py             # blocker → Agent 可执行修正指引
│   └── rubrics/
│       ├── default.yaml        # 内置红线 Rubric 表 (TC/SC/SEC/HP/QL)
│       ├── judge_prompt.md     # 裁判 system prompt 模板
│       └── schema.json         # RubricVerdict JSON Schema
│
├── subtask/                    # Subtask I/O 管道 (手层 — Executor 不拼 prompt)
│   ├── handoff.py              # prepare / commit / turn 策略
│   ├── prompt_builder.py       # L0/L2/L3 system + <file> 预载
│   ├── preload.py              # 物理读文件、slice、truncation 检测
│   ├── session_memory.py       # explore 缓存 + digest
│   ├── tool_pipeline.py        # 工具 before/after (serve-cache、截断)
│   └── context_pipeline.py     # fold / compact
│
├── gates/                      # Plan / Preflight / Exit 门禁
│   ├── plan_gate.py
│   ├── preflight_probe.py
│   ├── exit_gate.py
│   └── types.py
│
├── discovery/                  # Scout 与 manifest
│   ├── scout_agent.py
│   ├── scout_preflight.py      # Harness 确定性 grep
│   └── manifest.py
│
└── sandbox/                    # 沙盒环境
    ├── __init__.py
    ├── executor.py             # 命令沙盒执行
    ├── file_guard.py           # 文件操作守护 (防误删)
    └── resource_limit.py       # 资源限制 (时间/内存)
```

**Subtask 数据流**（Orchestrator 控制循环）：

```
Orchestrator
  collect_prior_summaries()     ← 前序 subtask 摘要
  prepare_executor_handoff()    ← prompt + runtime 策略
       ↓
SubTaskExecutor (脑)           LLM turn loop + tool round-trip
  before_executor_llm_call()    ← digest compact/fold → probe trim（统一入口）
  after_executor_tool_round() ← fold 策略（explore 后）
       ↓
  commit_subtask_success()      ← 摘要入库、diagnose → context_files 传播
  commit_subtask_failure()      ← digest 保留、attempt++
```

### src/executor/ — Subtask ReAct (瘦 Executor)

```
src/executor/
├── subtask_executor.py         # LLM turn loop；I/O 委托 harness/subtask/*
├── policy.py                   # 工具面解析 (由 handoff 调用)
├── retry_strategy.py           # attempt 模式 (由 handoff 调用)
└── context_compress.py         # digest 合并 (由 context_pipeline 调用)
```

### src/orchestrator/ — 控制循环

```
src/orchestrator/
├── orchestrator.py             # Scout → Planner → Gate → Preflight → Executor
├── isolation.py                # 兼容 re-export → harness/subtask/prompt_builder
└── escalation.py               # retry / replan 决策
```

### src/tools/ — 工具集

```
src/tools/
├── __init__.py
├── base.py                     # Tool 基类与注册机制
├── registry.py                 # 工具注册表
│
├── file/                       # 文件操作工具
│   ├── read.py                 # 读文件 (支持行号范围)
│   ├── write.py                # 写文件 (全量覆写)
│   ├── edit.py                 # 编辑文件 (精准字符串替换)
│   ├── create.py               # 创建文件
│   └── delete.py               # 删除文件
│
├── search/                     # 搜索工具
│   ├── grep.py                 # 正则搜索 (ripgrep wrapper)
│   ├── glob.py                 # 文件名模式匹配
│   ├── semantic.py             # 语义搜索 (基于索引)
│   └── symbol.py              # 符号搜索 (函数/类定义)
│
├── shell/                      # Shell 工具
│   ├── executor.py             # 命令执行
│   ├── background.py           # 后台进程管理
│   └── pty.py                  # PTY 模拟 (交互命令)
│
├── git/                        # Git 工具
│   ├── status.py               # git status/diff/log
│   ├── commit.py               # git add/commit
│   ├── branch.py               # 分支管理
│   └── stash.py                # 自动 stash (安全网)
│
└── web/                        # Web 工具 (可选)
    ├── fetch.py                # URL 抓取
    └── search.py               # Web 搜索
```

### src/context/ — 上下文工程

```
src/context/
├── __init__.py
├── builder.py                  # 上下文构建器 (组装 system + messages)
├── file_tracker.py             # 文件追踪 (哪些文件被读过/改过)
├── relevance.py                # 相关性判断 (哪些文件该加载)
├── compression.py              # 上下文压缩 (长对话摘要)
├── window.py                   # 滑动窗口管理
└── strategies/
    ├── recent_first.py         # 最近优先策略
    ├── dependency_aware.py     # 依赖感知策略
    └── task_focused.py         # 任务聚焦策略
```

### src/memory/ — 记忆系统

```
src/memory/
├── __init__.py
├── manager.py                  # 记忆管理器
├── layers/
│   ├── working.py              # L1: 工作记忆 (当前任务上下文)
│   ├── project.py              # L2: 项目记忆 (代码规范/架构/约定)
│   └── long_term.py            # L3: 长期记忆 (历史任务摘要)
├── store/
│   ├── sqlite_store.py         # SQLite 本地存储 (零依赖)
│   └── vector_store.py         # 向量存储 (本地 Embedding)
└── operations/
    ├── summarize.py            # 记忆压缩/摘要
    ├── retrieve.py             # 记忆检索
    └── forget.py               # 记忆淘汰策略
```

### src/planner/ — 任务规划

```
src/planner/
├── __init__.py
├── decomposer.py              # 任务分解 (大任务 → 子步骤)
├── dependency_graph.py         # 步骤依赖关系图
├── strategy.py                 # 执行策略选择
└── validator.py                # 计划可行性验证
```

### src/indexer/ — 代码库索引

```
src/indexer/
├── __init__.py
├── scanner.py                  # 项目文件扫描 (尊重 .gitignore)
├── parser.py                   # 代码解析 (AST → 符号表)
├── embedder.py                 # 代码 Embedding 生成
├── store.py                    # 索引存储 (SQLite + 向量)
├── incremental.py              # 增量更新 (文件变更时)
└── languages/                  # 多语言支持
    ├── python.py
    ├── typescript.py
    ├── java.py
    ├── go.py
    └── generic.py              # 通用文本回退
```

### src/config/ — 配置系统

```
src/config/
├── __init__.py
├── settings.py                 # 全局配置 (Pydantic Settings)
├── project_config.py           # 项目级配置 (.mitkii/config.toml)
├── model_config.py             # 模型配置 (provider/model/params)
└── permissions.py              # 权限规则配置
```

### src/server/ — Server 模式 (GUI 接入层, Phase 8)

```
src/server/
├── __init__.py
├── protocol.py                 # JSON-RPC 2.0 协议定义
├── stdio_transport.py          # stdin/stdout 传输 (VS Code 用)
├── websocket_transport.py      # WebSocket 传输 (Web IDE 用)
├── handler.py                  # 请求路由与处理
├── session.py                  # 多会话管理
└── events.py                   # Event → JSON 序列化
```

### extensions/vscode/ — VS Code Extension (Phase 9)

```
extensions/vscode/
├── src/
│   ├── extension.ts            # Extension 入口 (activate/deactivate)
│   ├── agent-client.ts         # 与 MitKII server 的 JSON-RPC 通信
│   ├── providers/
│   │   ├── inline-diff.ts      # Inline Diff 装饰器 (接受/拒绝)
│   │   ├── completion.ts       # InlineCompletionItemProvider (Tab补全)
│   │   ├── code-action.ts      # Quick Fix / Refactor 建议
│   │   └── code-lens.ts       # CodeLens 提示 (函数上方 "Ask AI")
│   ├── views/
│   │   ├── chat-panel.ts       # 侧边栏对话面板 (Webview)
│   │   ├── history-panel.ts    # Checkpoint 历史面板
│   │   └── changes-panel.ts    # AI 变更列表面板
│   ├── commands/
│   │   ├── explain.ts          # 选中代码 → 解释
│   │   ├── refactor.ts         # 选中代码 → 重构
│   │   ├── fix.ts              # 选中代码 → 修复
│   │   ├── test.ts             # 选中代码 → 生成测试
│   │   └── rollback.ts         # 回滚到 checkpoint
│   └── utils/
│       ├── diff-decorator.ts   # Diff → VS Code Decoration 转换
│       └── config.ts           # Extension 配置读取
├── webview/                    # 对话面板前端 (React)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── ChatMessage.tsx
│   │   ├── ToolCallBlock.tsx
│   │   └── DiffPreview.tsx
│   └── package.json
├── package.json                # Extension manifest
└── tsconfig.json
```

### 配置文件

```
项目根目录或用户 home:
~/.mitkii/
├── config.toml                 # 全局配置
├── credentials.toml            # API Keys (加密)
├── memory.db                   # 长期记忆 SQLite
└── sessions/                   # 历史会话存档

项目内 (可选):
.mitkii/
├── config.toml                 # 项目级配置覆盖
├── rules.md                    # 项目编码规范 (AI 遵守)
├── index.db                    # 代码索引缓存
└── checkpoints/                # Checkpoint 文件
```
