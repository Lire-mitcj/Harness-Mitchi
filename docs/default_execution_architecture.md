# Cursor 风格执行架构设计文档 (Cursor Execution Architecture)

本文档整理并阐述了 MitKII 当前所采用的 **Cursor 风格执行架构 (CursorLoop)**。该架构的核心设计思想是**通过确定性的规则和工具链，约束并保障 LLM 的单步决策与验证**，实现“先建立确定性运行时，再叠加语义层”的设计原则。

---

## 1. 架构总览与核心流程

CursorLoop 是一个确定性的、有状态的、受最大步数限制的循环体。其执行流程分为 **初始化与检索准备阶段** 以及 **迭代决策与验证阶段**：

```text
User Input (用户请求)
  │
  ├──► [Parallel] CursorInterLLM (意图理解与生成 Hint) 
  └──► [Parallel] CursorQueryBridge (生成结构化检索 Query)
        │
        ▼
  CursorGraphQueryBridge (AST符号图关联拓展与统一打分/Decay)
        │
        ▼
  CursorRetriever (确定性 Grep 与 AST 检索)
        │
        ▼
  CursorFusionEngine (检索结果融合与多策略排序)
        │
        ▼
  CursorContextPackBuilder (上下文代码窗口装配与语义标注)
        │
  ┌─────┴────────────────────────────────────────┐
  ▼                                              ▼
[Step Loop (1..Max Steps)]                      Success / Fail (直接返回/结束)
  │
  ├──► CursorDecisionLLM (LLM 决策：Edit/Answer/Clarify)
  │      │
  │      ├── (answer / ask_clarify) ──► 成功/澄清结束
  │      ▼
  │     (edit)
  │      ▼
  ├──► CursorExecutor & PatchApplier (原子性 Patch 应用)
  │      │
  │      └── [Failure] ──► 记录 Observation ──► 进入下一轮 Step Loop
  │      ▼ [Success]
  ├──► CursorValidator (运行自动化测试，如 pytest)
  │      │
  │      ├── [Success] ──► 标记成功并结束
  │      └── [Failure] ──► 记录错误 Observation ──► 进入下一轮 Step Loop
  ▼
Max Steps Exhausted ──► 标记失败并结束
```

---

## 2. 核心组件及其职责

### 2.1 初始化与多模态准备组件

*   **`StateAssembledLoop`** ([state_assembled_loop.py](file:///home/csh/harness-mitkii/src/agent/state_assembled_loop.py))
    *   **职责**：主控运行环境。采用状态装配（State Assembly）模式驱动任务的逐步迭代。负责调度并运行 codebase_retrieve 和 decision_edit 工具，并在 Step 循环中驱动决策、执行与验证的状态转移。
*   **`CursorInterLLM`** ([cursor_inter_llm.py](file:///home/csh/harness-mitkii/src/agent/cursor_inter_llm.py))
    *   **职责**：轻量级意图分类器。将用户请求快速分类为 `modify`、`explain` 或 `debug`，并输出置信度作为后续 Decision 决策的 hint，不影响控制流决策。
*   **`CursorQueryBridge`** ([cursor_query_bridge.py](file:///home/csh/harness-mitkii/src/agent/cursor_query_bridge.py))
    *   **职责**：检索意图重写器。利用 LLM 将自然语言请求转换为多维度结构化搜索参数（包括 `search_terms`、`symbols`、`file_hints` 等）。

### 2.2 符号图与融合检索组件

*   **`CursorGraphQueryBridge`** ([cursor_graph_bridge.py](file:///home/csh/harness-mitkii/src/agent/cursor_graph_bridge.py))
    *   **职责**：AST 符号相关度扩展引擎。
    *   **统一打分**：使用 `normalize(lexical, semantic, graph, alias, feedback)` 统一计算每个符号的评分。
    *   **边类型感知衰减**：利用 `_decay(edge_type, distance)` 对 reference, import, co_occurrence, naming_similarity, semantic_alias 边按深度和类型进行自适应分值衰减。
    *   **动态候选过滤上限**：`_semantic_candidates` 依据仓库规模，动态计算 Embedding 校准过滤上限 (`max(cap, min(len(symbols) // 5, 1500))`)。
*   **`CursorRetriever`** ([cursor_retriever.py](file:///home/csh/harness-mitkii/src/agent/cursor_retriever.py))
    *   **职责**：基于 Grep、路径精确匹配与 AST 提取，进行高效率、无模型的确定性代码片段捞取。
*   **`CursorFusionEngine`** ([cursor_fusion.py](file:///home/csh/harness-mitkii/src/agent/cursor_fusion.py))
    *   **职责**：融合多路召回结果，对其进行去重、按权重交叉排序并限制上限，选定候选代码范围。
*   **`CursorContextPackBuilder`** ([cursor_context_pack_builder.py](file:///home/csh/harness-mitkii/src/agent/cursor_context_pack_builder.py))
    *   **职责**：代码上下文装配器。将检索到的符号与文件片段展开为决策 LLM 容易阅读的代码窗口（Code Windows），并在支持时注入语义标注。

### 2.3 迭代控制与验证组件

*   **`CursorDecisionLLM`** ([cursor_decision.py](file:///home/csh/harness-mitkii/src/agent/cursor_decision.py))
    *   **职责**：单步局部的决策引擎。每次决策只允许输出三种 action 之一：`edit`（编辑指定文件）、`answer`（直接回答）、`ask_clarify`（澄清请求）。
    *   **约束**：每次 `edit` 只能对**单个候选文件**生成 Search/Replace 风格的 Patch。
*   **`CursorExecutor`** ([cursor_executor.py](file:///home/csh/harness-mitkii/src/agent/cursor_executor.py)) & **`CursorPatchApplier`** ([cursor_patch_applier.py](file:///home/csh/harness-mitkii/src/agent/cursor_patch_applier.py))
    *   **职责**：Patch 执行与应用层。
    *   **约束**：支持空格容错的精准 Search/Replace。采用原子写入机制，多 block 替换时若任意 block 失败则文件完全不变，并返回具体错误作为 Observation。
*   **`CursorValidator`** ([cursor_validator.py](file:///home/csh/harness-mitkii/src/agent/cursor_validator.py))
    *   **职责**：运行时行为验证器。对应用完 Patch 的代码运行项目预配置的命令（通常是 `pytest`），并返回执行的 stdout/stderr 截断日志。
*   **`CursorStateManager`** ([manager.py](file:///home/csh/harness-mitkii/src/harness/cursor/manager.py))
    *   **职责**：状态管理器与格式化输出。
    *   **约束**：维护的 `CursorState` 体积必须限制在指定的字节上限（默认 2KB）内，仅保留当前步骤的 `last_patch` 与 `last_observation`（前次执行/验证结果），无长短期对话历史，确保 LLM 每次调用只做单步上下文计算。

---

## 3. 核心设计原则

1.  **确定性运行时优先 (Deterministic Runtime)**
    所有的控制流程、检索过滤、Patch应用、测试验证全部由静态代码和确定性规则严格约束，不允许 LLM 自行决定执行哪些 shell 命令或修改哪些无关文件。
2.  **单点局部决策 (Single Local Decision)**
    决策层在每一步仅输入当前快照状态与限制大小的上下文，不做多文件 Patch。如果 validation 失败，错误将被转化为 Observation 输入下一轮状态循环重新决策。
3.  **零多轮记忆 carry (Stateless Loop State)**
    State 绝不保留完整的历史 Assistant 对话消息，每次仅保留最新的 patch 和 observation 错误。极大避免了 LLM 产生幻觉以及上下文膨胀造成的推理漂移。
