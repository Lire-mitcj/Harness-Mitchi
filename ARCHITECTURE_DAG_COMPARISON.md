# Planner/Harness Architecture Comparison

本文对比当前严格阶段式架构与建议的类 DAG 架构，重点关注 Planner 执行计划、Harness 门控、上下文组装与评分系统边界。Executor 当前能力基本足够，主要不需要大改。

这份方案的核心原则是：Harness 是中间件，不应该侵入业务决策。它只应该回答“能不能改、改哪、改完对不对”，不应该替 Agent 决定“该怎么改、该搜什么、该用哪个 view”。

## 总结

当前系统名义上有 DAG，但 Planner prompt、PlanGate、ExitGate 把任务压成了严格的 CoT pipeline：

```text
diagnose -> design -> edit -> verify
```

建议改成目标 DAG + artifact blackboard，并把职责边界收敛成：

```text
CLI
 ↓
Planner                 负责计划、业务策略、依赖假设
 ↓
Harness
    ├── Scope Gate       只管允许改哪里
    ├── Safety Gate      只管危险操作和越权风险
    ├── Context Assembly 只组装证据，不决定业务方案
    └── Validator/Scorer 只验证结果是否满足用户目标和工程红线
 ↓
Executor                负责搜索、判断、编辑、修正
 ↓
Tools
```

核心变化：不要让 `diagnose/design` 的 JSON 成为流程通行证，也不要让 `dependencies_resolved=false` 这种业务状态直接禁止 edit。Harness 只在越权、危险、无法验证、明显不符合用户目标时拦截。

## 职责边界

| 组件 | 应该负责 | 不应该负责 |
|---|---|---|
| Planner | 任务拆解、DAG 依赖、是否需要发现节点、是否需要设计节点、验收目标 | 被 Harness 强制走固定 CoT 阶段 |
| Executor | 搜索代码、判断依赖、选择实现方案、选择 view/helper/API、执行编辑、根据失败继续查证 | 被限制成只能消费上一节点 handoff |
| Scope Gate | 写范围、路径边界、是否改到了允许文件/符号 | 判断应该用哪个 view、哪个 SQL 策略 |
| Safety Gate | 删除/覆写/批量改动/shell 等危险动作审批，保护工作区 | 判断业务依赖是否已解析 |
| Context Assembly | 根据 scope、artifact、recent evidence 组装上下文 | 裁决业务事实真伪或替 Agent 选方案 |
| Validator | 改完后用测试、lint、diff、rubric 验证结果 | 在执行前替 Agent 做业务路线选择 |
| Scorer | L0 工程检查、L1 用户目标符合度、L2 非阻断质量建议 | 把“未发现依赖”当成禁止执行的前置审批 |

## 对比表

| 维度 | 当前架构 | 建议改后架构 |
|---|---|---|
| 总体形态 | 名义上有 DAG，但实际像严格 CoT pipeline | 真正的目标 DAG + 共享事实黑板 |
| Planner 职责 | 生成固定阶段链：`diagnose -> design -> edit -> verify` | 生成目标/能力节点 DAG：节点表达要完成什么，不强制先后阶段 |
| Planner 判断依据 | `HARNESS_TASK_ANALYSIS_JSON.edit_ready` 和 complexity 强绑定流程 | Planner 自己决定流程形状；Harness 只给风险/上下文提示 |
| 复杂任务计划 | high 必须 diagnose/design/edit/verify；medium 必须 diagnose/edit/verify | high 可以拆多个目标节点；edit 只要求 scope/safety 清楚，业务证据由 Executor 查证、Validator 验证 |
| Edit 前置条件 | 必须依赖 diagnose/design 输出，尤其 `PATCH_INTENT_JSON` | 只要求写范围安全；业务依赖可由 edit 节点继续查证 |
| Diagnose 地位 | 近似必经审批步骤 | 可选发现节点；只有事实缺失时才需要 |
| Design 地位 | high complexity 必须输出完整 `PATCH_INTENT_JSON` | 可选策略节点；复杂改动可保留，但不作为通用硬协议 |
| 节点依赖 | 多数节点按阶段线性依赖上一节点 | 节点按数据依赖、写冲突、验证依赖组成 DAG |
| 多目标编辑 | 倾向顺序拆成多个 edit step | 可按目标拆，也可并行发现；写同一文件/符号时再串行 |
| Planner 输出 | 每步带 `kind/tools/context_files/requires_handoff` | 每步除 kind/tools 外，声明 `requires_artifacts`、`produces_artifacts`、`write_scope` |
| 工具策略 | kind 强约束工具；edit 已可 `context_search`，但语义上仍被要求基于 handoff | Executor 工具面基本不变；edit 可查证、可补 artifact，但写操作仍受 scope/gate 控制 |
| Handoff | 上一步 final summary / JSON 传给下一步 | 结构化 artifact store：事实、证据、置信度、版本、冲突 |
| 上一步结果使用方式 | 下一步被鼓励/要求消费上一步输出 | 下一步先 query artifact，可复用、可验证、可补查、可纠错 |
| 视图字段问题 | `target_view`/字段名在 diagnose/design/edit 间可能不一致，导致重复搜索或阻塞 | 用 canonical schema + alias + evidence 合并；字段冲突只局部补查 |
| Harness 组装 prompt | `collect_prior_summaries()` + `prepare_executor_handoff()` 把前序摘要塞入上下文 | 组装 scope、artifact slice、相关 evidence、最近探索摘要；不替 Agent 选方案 |
| Context preload | 主要来自 `context_files` 和前序 handoff | 来自 scope、artifact、evidence 引用和最近探索记录；不把 artifact 当硬门票 |
| PlanGate | 检查是否按 complexity 出现固定阶段、是否先 diagnose/design | 只检查 DAG 结构、节点字段、工具权限、写范围声明，不做业务策略审批 |
| ExitGate for diagnose | 强校验 handoff evidence：file:line/symbol/snippet 缺失就 block | diagnose 产出 partial evidence 也可通过；缺失项作为 warning/next hints |
| ExitGate for design | 强校验 `PATCH_INTENT_JSON` 必填字段和 `edit_ready=true` | design 只输出策略建议；不把 `edit_ready` 当硬门票 |
| Edit Gate | 主要看工具错误、changed_files、context_files whitelist | 只检查是否越过 write_scope、安全策略、是否产生有效变更 |
| 失败恢复 | edit 失败后 replan，经常回到 diagnose/design/edit | edit 失败先补 artifact 或扩大局部 evidence，再决定是否 replan |
| 可用性 | 保守但容易卡死；信息不完整就不过 harness | 保守点移到写前证据，发现过程更自由 |
| 安全性 | 靠固定流程保证安全 | 靠 scope、权限、安全审批、diff 验证、测试和评分保证安全 |
| Executor 改动 | 当前 ReAct executor 已够用 | 基本不动；最多让 prompt 明确“先查 artifact，缺失再 context_search” |
| 主要改动文件 | `prompts/planner_prompt.md`、`src/harness/gates/plan_gate.py`、`src/harness/gates/exit_gate.py` | 同上，外加新增 artifact store / schema / resolver / prompt assembly adapter |

## 当前架构的主要限制点

### Planner Prompt

当前 `prompts/planner_prompt.md` 把复杂任务固定为阶段链：

```text
diagnose -> design -> edit -> verify
```

这导致 Planner 的核心职责从“规划达成目标的节点”变成“满足 Harness 期待的审批流程”。对 SQL view、helper/function refactor、API/contract modification 等任务尤其明显。

### PlanGate

当前 PlanGate 会基于 complexity 和 `edit_ready` 阻止某些计划形状：

- `edit_ready=false` 时不能直接 edit。
- high complexity 必须包含 diagnose、design、edit、verify。
- medium complexity 必须包含 diagnose、edit、verify。
- 某些任务必须以 diagnose 开头。

这些规则提升了表面安全性，但也让 Harness 开始替 Agent 做业务决策。典型问题是：

```text
dependencies_resolved=false
↓
禁止 edit
```

这不是安全判断，而是业务判断。依赖是否解析、应该使用哪个 view、是否需要继续搜索，应该由 Planner + Executor 决定。Harness 最多提示风险、收集证据、限制写范围。

### ExitGate

当前 ExitGate 对 diagnose/design 的输出要求很硬：

- diagnose 必须满足 file:line、symbol、snippet/decision 等 handoff evidence。
- design 必须输出完整 `PATCH_INTENT_JSON`。
- `PATCH_INTENT_JSON.edit_ready` 必须为 true。
- `target_view` 等字段在 SQL view rewrite 场景下必须出现。

这会把“发现过程中的不完整信息”当成失败，而不是把它作为后续节点可继续补全的 partial evidence。

### Scorer / Quality Gate

当前评分系统本身的分层方向是对的：

- L0：lint/test 等程序化检查。
- L1：LLM rubric judge 判断任务完成度。
- L2：非阻断质量建议。

需要调整的是使用边界：评分系统应该在 edit 后验证结果，不应该被前置成业务路线审批。比如“是否用了某个 view”可以作为用户目标的一部分在 L1/L0 中验证，但不应该在 edit 前由 Harness 强制决定。

## 建议架构

### 1. Planner 生成目标 DAG

Planner 不再按固定阶段输出流程，而是输出目标节点和数据依赖。

示例：

```json
{
  "root_task": "Switch report SQL to view",
  "nodes": [
    {
      "id": "st-1",
      "kind": "diagnose",
      "description": "Locate report SQL and candidate database view",
      "produces_artifacts": ["code_target", "database_view"],
      "depends_on": []
    },
    {
      "id": "st-2",
      "kind": "edit",
      "description": "Rewrite report SQL using resolved view",
      "requires_artifacts": ["code_target", "database_view"],
      "write_scope": ["app/report.py"],
      "depends_on": ["st-1"]
    },
    {
      "id": "st-3",
      "kind": "verify",
      "description": "Run related tests",
      "depends_on": ["st-2"]
    }
  ]
}
```

如果已有 repo_map、context pack 或 artifact 能满足 edit 前置条件，Planner 可以直接生成 edit 节点，不必强制 diagnose/design。

### 2. Handoff 改为 Artifact Store

上一步不再把自然语言摘要或单个 JSON 直接交给下一步，而是写入共享 artifact store。

示例：

```json
{
  "artifact_type": "database_view",
  "canonical_id": "db.view.view_ticket_report_detail",
  "name": "view_ticket_report_detail",
  "aliases": ["ticket_report_detail_view"],
  "columns": [
    {
      "canonical_name": "ticket_id",
      "observed_names": ["ticket_id", "id"],
      "type": "unknown",
      "nullable": null,
      "evidence_refs": ["schema.sql:42-88"],
      "confidence": 0.88
    }
  ],
  "evidence": [
    {
      "file": "schema.sql",
      "line_start": 42,
      "line_end": 88
    }
  ],
  "confidence": 0.91,
  "producer": "st-1",
  "version": 1
}
```

下一步可以复用 artifact，也可以补查；补查结果要 merge 回 artifact store。

### 3. 视图字段冲突用 Resolver 处理

字段名不一致时，不应该让每个节点重新搜索一遍。应增加 artifact resolver：

- 同名字段直接命中。
- alias 命中时合并。
- evidence 指向同一 SQL expression 时合并。
- 类型或来源冲突时标记 conflict。
- edit 前只补查 conflict 字段，而不是重查整个 view。

目标是把“字段不一致”转成可管理的 fact reconciliation，而不是流程失败或重复探索。

### 4. Harness Gate 改为职责门控

PlanGate 不再检查是否按固定阶段排列，也不再用 `dependencies_resolved=false` 阻止 edit。它只检查：

- DAG 是否无环。
- 节点依赖是否存在。
- edit 节点是否声明 `write_scope` 或可由 Planner context 推导出候选范围。
- 多个 edit 节点是否存在写冲突。
- 节点工具是否越权，例如 diagnose/design 不能写文件。
- 高风险任务是否需要用户确认或更强 validator，而不是固定 design step。

Scope Gate 检查：

- 写入范围被限制在 `write_scope`。
- 实际 changed_files 是否落在允许范围内。
- 编辑是否越过用户明确禁止的路径、生成物、锁文件、配置边界。
- 对不在 scope 内的改动 block；对 scope 模糊的改动 warn 或要求 Planner 收窄。

Safety Gate 检查：

- 删除文件、全量覆写、大规模替换、shell 命令等危险动作。
- 是否触碰 secret、凭据、权限、生产配置。
- 是否需要用户确认。
- 是否违反 sandbox/path guard。

Validator / Scorer 检查：

- L0：lint、测试、格式、静态检查。
- L1：用户目标是否完成，是否符合 acceptance criteria。
- L2：质量建议、范围外重构提示、可维护性提醒。

ExitGate 只做节点收尾：

- diagnose/design 节点产生 partial evidence 也可通过。
- edit 节点必须有有效变更；无变更但有 recoverable tool error 时给 Executor 继续机会。
- verify 节点按测试/命令结果判断。
- 不再因为业务字段不完整而强制 re_plan。

## 对 Executor 的影响

Executor 基本不需要改，因为当前 edit already supports `context_search`，足够支撑“边编辑边查证”的执行模型。

需要调整的主要是 prompt 和 handoff：

- Executor prompt 中加入“先查 artifact，缺失再 context_search”。
- 工具面仍按 kind 控制，尤其写工具仍只在 edit 节点开放。
- Executor 的输出多一个 artifact update 区域，供 Harness merge。

也就是说，Executor 仍然是 ReAct loop；变的是它拿到的上下文不再是上一步摘要堆叠，而是当前节点所需的 artifact slice 和 evidence。

## 分步修改方案

### Phase 1: 先停止 Harness 侵入业务决策

目标：不引入大新系统，先把最阻塞的业务审批从 block 改成 warn/hint。

修改点：

1. `prompts/planner_prompt.md`
   - 删除 high/medium 必须固定 `diagnose -> design -> edit -> verify` 的强制措辞。
   - 把 `HARNESS_TASK_ANALYSIS_JSON` 改成参考信息，不再 authoritative。
   - 明确 Planner 可以直接安排 edit，只要有合理 scope，edit 节点可以继续 `context_search`。
   - 保留“复杂任务推荐 diagnose/design”的建议，但不能写成必须。

2. `src/harness/gates/plan_gate.py`
   - 移除或降级 `edit_ready=false` 禁止 edit 的 block。
   - 移除 high/medium complexity 的固定阶段顺序 block。
   - 保留 DAG 基础校验、未知工具校验、禁止写工具出现在 diagnose/design。
   - 新增或强化 scope 校验：edit 必须有 `context_files`、`write_scope`、repo_map target 之一。

3. `src/harness/task_analysis.py`
   - `dependencies_resolved`、`targets_resolved`、`acceptance_resolved` 继续计算，但改名或定位为 `planning_hints`。
   - `edit_ready` 不再作为 gate 硬条件，只作为 Planner prompt 的风险提示。

验收标准：

- `dependencies_resolved=false` 不再直接导致 PlanGate block edit。
- Planner 可以输出单个 edit + verify，Harness 只在写范围不清或工具越权时拦截。

### Phase 2: 拆清 Gate 职责

目标：把 PlanGate/ExitGate 里的混合逻辑拆成 Scope Gate、Safety Gate、Validator。

建议职责：

| 模块 | 保留/新增职责 | 从该模块移走的职责 |
|---|---|---|
| PlanGate | DAG 无环、节点字段完整、工具权限、写范围声明 | 固定阶段顺序、业务依赖是否解析、必须用哪个 view |
| Scope Gate | 写路径、符号范围、changed_files 是否越界 | 判断业务方案是否正确 |
| Safety Gate | 删除、覆写、shell、secret、生产配置、sandbox/path guard | 判断应该搜索什么或怎么改 |
| ExitGate | 节点是否产出结果、错误是否可恢复、是否需要 retry/replan | diagnose/design JSON 字段审批 |
| Validator/Scorer | edit 后验证 diff、测试、用户目标、工程质量 | edit 前阻止业务探索 |

具体改动：

1. 从 `exit_gate.py` 中降级 diagnose handoff 缺字段：
   - `file:line`、`symbol`、`snippet/decision` 缺失时返回 warning。
   - 只有节点声称完成但结果自相矛盾、工具错误不可恢复时 block。

2. 从 `exit_gate.py` 中降级 `PATCH_INTENT_JSON`：
   - 缺 `target_view`、`edit_ready` 不再 block。
   - `PATCH_INTENT_JSON` 格式非法可以 warn，除非后续节点明确 `requires_handoff=["PATCH_INTENT_JSON"]` 且没有 fallback。

3. 把 changed_files/context_files whitelist 逻辑升级成 Scope Gate：
   - 改出 scope：block。
   - scope 缺失：plan warning 或要求 Planner 收窄。
   - scope 内但业务证据弱：不 block，交给 Executor/Validator。

验收标准：

- Gate 的 block 原因主要是越权、危险、结构非法、验证失败。
- Gate 的 warn 原因可以包括依赖不完整、证据不足、handoff partial。

### Phase 3: 调整 Context Assembly，不替 Agent 选方案

目标：Harness 负责把信息摆好，不负责裁决哪个信息是业务答案。

修改点：

1. `prepare_executor_handoff()` / prompt builder
   - 继续提供 prior summaries，但标注为 evidence/hints。
   - 不写“must use resolved dependency/view”这类硬指令。
   - 对 edit 节点明确：可以复用 prior evidence，也可以自行 `context_search` 验证或补查。

2. context preload
   - 根据 scope 和 Planner 给出的候选文件预载。
   - 对 artifact/evidence 只做引用和摘要，不把它们变成业务真理。

3. prompt wording
   - 用“available evidence”替代“approved dependency”。
   - 用“validate before editing”替代“must use this view”。

验收标准：

- Executor 看到的是证据包，不是审批结论。
- 上一步 JSON 字段不一致时，Executor 可以自行查证并继续，而不是被 Harness 阻塞。

### Phase 4: 引入轻量 Artifact Store

目标：减少重复搜索，但不让 artifact 变成新的业务审批系统。

先支持三类 artifact：

| Artifact | 用途 | 是否可阻塞 edit |
|---|---|---|
| `code_target` | 记录候选文件、符号、行号、snippet | 不能单独阻塞；只辅助 scope |
| `database_view` | 记录 view 名、字段、证据来源、别名 | 不能阻塞；字段冲突作为 warning/hint |
| `patch_intent` | 记录某次设计建议 | 不能作为必须通行证，只作为策略建议和验证参考 |

resolver 规则：

- 同名字段直接命中。
- alias 命中时合并。
- evidence 指向同一 SQL expression 时合并。
- 冲突时标记 conflict，提示 Executor 局部补查。
- artifact 置信度只影响 prompt 排序和 warning，不直接控制能否 edit。

验收标准：

- artifact 用于减少重复查找。
- artifact 冲突不会直接让 Harness 禁止 edit。
- edit 仍受 Scope/Safety/Validator 约束。

### Phase 5: 重构 Validator / Scorer 分层

目标：评分系统只验证结果，不做前置业务决策。

建议分层：

| 层级 | 名称 | Block 条件 | Non-block 条件 |
|---|---|---|---|
| L0 | Programmatic Validator | 测试失败、lint/type 失败、命令失败、生成无效文件 | 无相关测试、测试未配置 |
| L1 | Task Completion Judge | 明显没完成用户目标、diff 与目标无关、破坏关键行为 | 证据不足但 diff 合理 |
| L2 | Quality Advisor | 默认不 block | 重构建议、命名建议、性能提醒 |
| Scope | Scope Validator | 改出允许范围、触碰禁止路径 | scope 模糊但未越界 |
| Safety | Safety Validator | 未确认的删除/覆写/危险 shell/secret 风险 | 低风险安全提醒 |

具体改动：

1. `src/harness/quality_gate.py`
   - 保持 edit 后执行。
   - `auto_rewrite` 只由 L0 fail、L1 明确 blocker、Scope/Safety fail 触发。
   - L2 warning 不触发 rewrite。

2. `src/harness/scorer/engine.py`
   - 保留 L0 -> L1 -> L2。
   - L1 rubric 只评价“用户目标是否完成”和“是否有行为回归”。
   - 不把“没有使用 Harness 解析出的 dependency”作为 blocker。

3. `src/harness/scorer/rubrics/default.yaml`
   - Scope 类 rubric 保留“是否超出用户范围”。
   - 删除或避免“必须使用某个 Harness dependency/view”的措辞。
   - SQL/view 类要求应来自用户目标或测试结果，而不是 Harness 预判。

验收标准：

- Scorer 不再因为 Planner/Diagnose 没给完整 dependency 而失败。
- Scorer 只根据 diff、测试、用户目标、明确 acceptance criteria 做判断。

### Phase 6: 再考虑真正 DAG 执行

目标：在职责边界清楚后，再引入并行/条件 DAG，避免过早复杂化。

建议顺序：

1. 先允许 Planner 输出非线性 `depends_on`。
2. Orchestrator 使用 ready pending nodes，而不是固定 first pending。
3. 有写冲突的 edit 节点串行，无写冲突的 diagnose/verify 可并行。
4. 失败恢复优先局部 retry，必要时 replan 子图。

验收标准：

- DAG 是执行优化，不是业务审批工具。
- Harness 的 gate 仍只管 scope/safety/validation。

## 最小可行版本

如果想先小步试验，不必一次做完整 artifact 系统，可以先做三个低风险改动：

1. Planner prompt 删除 high/medium 必须固定阶段链的强制规则，改成“推荐但不强制”。
2. PlanGate 不再因 `edit_ready=false`、`dependencies_resolved=false`、缺少 diagnose/design 直接 block，改成 warn。
3. ExitGate 对 diagnose/design 的缺字段改成 partial evidence warning，不直接 re_plan。
4. Scope/Safety 继续强 block：越权写、危险操作、写范围不明时不放行。
5. Quality Gate 只在 edit 后运行；L0/L1 blocker 才触发 retry，L2 只提示。

这样能明显提升可用性，同时不必立即重写 Executor，也不会牺牲真正的安全边界。
