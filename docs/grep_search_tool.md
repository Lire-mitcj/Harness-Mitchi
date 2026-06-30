# Grep 物理检索工具设计与优化文档 (Grep Search Tool Design & Optimization)

本文档整理并阐述了在 AI Agent 开发与评估架构中，**`grep_search` 物理检索工具**的设计原则、底层实现、针对大模型（LLM）的正则优化策略，以及其与静态地图（`RepoMap`）的协同工作流。

---

## 1. 工具设计原则 (Design Principles)

在大规模代码仓库检索中，完全依赖大模型进行“盲读”或“全词匹配”不仅速度慢，而且会迅速消耗上下文 Token。`grep_search` 工具遵循以下核心设计思想：

1.  **极速的物理执行 (High-performance Execution)**
    直接基于文本特征进行扫描，绕过大模型推理，作为第一道轻量级的物理过滤器。
2.  **噪音抑制与安全截断 (Noise Negation & Safety Capping)**
    大模型输入空泛关键字（如 `a` 或 `user`）可能会导致数以千计的匹配结果。工具端必须具备全局截断和排除特定噪音文件的能力，防止上下文通道被瞬间撑爆。
3.  **大模型友好 (LLM-Friendly Parameter Normalization)**
    由于调用工具的是 LLM，它常常会生成半结构化的模糊正则或纯文本短语，工具端必须在前置处理中对参数进行智能规整。

---

## 2. 底层技术实现 (Implementation)

在 Python 开发的 Agent 环境中，`grep_search` 底层封装了 **`ripgrep` (`rg`)** 二进制程序。其检索性能比原生的 Python 文件扫描高出数十到上百倍。

### 2.1 核心命令参数说明
工具在执行时使用 `asyncio.create_subprocess_exec` 异步调用 `rg`，并附带以下关键配置标志：

*   `--json`：开启 JSON 输出模式。`rg` 会生成包含匹配详细信息（文件路径、行号、匹配子串等）的结构化 JSON Lines，极度方便 Python 代码精准解析。
*   `--line-number` (`-n`) & `--column`：随检索结果一同提取出对应的行号与列号。
*   `--max-count <N>`：限制单个文件内最大的匹配行数（默认配置为 `100`），避免在单一巨型文本（如日志或大配置文件）中深陷。
*   `--ignore-case` (`-i`)：根据 LLM 的搜索输入智能开启大小写忽略。
*   `--glob <pattern>`：支持限制在特定文件扩展名（如 `*.py`）或特定目录内查找。

---

## 3. 针对 LLM 调用的正则匹配优化 (Regex Optimization)

由于大模型输入的检索词不总是完美的正则表达式，`grep_search` 在工具端实施了三项核心优化：

### 3.1 符号定义提升 (Symbol Definition Promotion)
如果大模型检索的是一个驼峰或下划线命名（如 `archive_passenger`），它绝大多数意图是想寻找**它的定义位置**，而非海量的调用位置（调用噪音）。
*   **优化策略**：工具端自动为其生成一个声明级的正则匹配前缀：
    ```regex
    \b(def|class|async\s+def|function)\s+archive_passenger\b
    ```
    工具会首先用这一高精度正则进行 `rg` 查找，若未匹配到定义，再退回到普通单词边界 `\barchive_passenger\b` 的泛搜索，确保首轮搜索的高召回率与低噪音。

### 3.2 空格多词交叉匹配 (AND 逻辑降级)
大模型常常像在搜索引擎中一样输入空格分隔的多个词（如 `"login expired token"`）。物理 grep 在匹配单行时无法直接命中这种交叉状态。
*   **优化策略**：当检测到 Pattern 包含空格且无显式正则元字符时，工具会自动将其改写为多路 AND 交叉正则：
    ```regex
    (login.*expired.*token|token.*expired.*login|...)
    ```
    或者在 Python 执行层使用管道机制：先用 `rg login` 筛选出包含该词的文件，再在内存中利用 Python Regex 过滤出同时包含 `expired` 和 `token` 的行。

### 3.3 全局干扰与噪音Negation
为防范不必要的全局扫描造成的性能衰减，工具在生成命令时会默认追加 `--glob "!*.lock"`、`--glob "!*.map"`、`--glob "!*.svg"`，并过滤掉 `.git/`、`.venv/`、`node_modules/`、`dist/` 等非源码目录。

---

## 4. 与 `RepoMap` 的协同工作流（“地图+雷达”架构）

在 MitKII 架构中，静态索引生成的 `RepoMap` 相当于**地图**，`grep_search` 相当于**雷达**。两者的协同工作机制如下：

```text
┌──────────────────────────────────────────────┐
│                  RepoMap                     │
│  (宏观地图：PageRank 节点、函数签名 skeleton)    │
└──────────────────────┬───────────────────────┘
                       │ 1. Planner 定位高频 Hub 文件与大体签名范围
                       ▼
┌──────────────────────────────────────────────┐
│                grep_search                   │
│  (微观雷达：限定 glob 路径进行正则/词组精准搜索)   │
└──────────────────────┬───────────────────────┘
                       │ 2. 定位具体的关联逻辑位置及精确行号
                       ▼
┌──────────────────────────────────────────────┐
│                 view_file                    │
│  (精确读取：按行段载入，进入修改决策循环)      │
└──────────────────────────────────────────────┘
```

### 4.1 协同检索流程
1.  **宏观定位**：Coordinating LLM 在第一步通过注入的 `RepoMap` Skeleton 骨架（只包含接口签名，不含实现）对项目建立全局的心智模型（Mental Model）。例如，看到 `list.py` 中有 `L48-81 archive_passenger` 符号。
2.  **微观探测**：当需要调查该接口的具体实现或其操作的数据库表时，Coordinating LLM 结合 `RepoMap` 中的文件名与函数名，向 `grep_search` 发射限定 glob 的定向雷达（如 `path="list.py", pattern="passenger_info"`）。
3.  **精准修改**：`grep_search` 返回少量高价值匹配行（包括行号和摘要），LLM 据此范围发起精准的 `view_file` 读取源码，并利用 `decision_edit` 生成精确的 Replace Patch。

这种设计将全局大文件的上下文开销（数十万 token）降到了局部读取的极低水准（几百 token），从根本上避免了上下文膨胀与推理漂移问题。
