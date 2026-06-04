# MitKII — AI Code Agent

终端驱动的 AI 编码代理。用自然语言描述需求，Agent 自主搜索代码、编辑文件、执行命令、自我验证。

类似 Claude Code / Aider，但底层由自研 Harness 引擎提供上下文探针、Checkpoint、评分、Pipeline 编排能力。

## 安装

```bash
pip install mitkii
```

## 快速开始

```bash
# 进入项目目录
cd your-project

# 初始化 (建立代码索引)
mitkii init

# 开始对话
mitkii chat
```

## 配置

```bash
# 设置 API Key
mitkii config set api_key sk-xxx

# 或使用环境变量
export ANTHROPIC_API_KEY=sk-ant-xxx
```

## 核心特性

- **多工具调用** — 文件读写、代码搜索、Shell 执行、Git 操作
- **上下文探针** — 智能管理 token 预算，长对话不爆上下文
- **Checkpoint** — 每次修改前自动快照，一键回滚
- **自我评分** — L0 lint/test 门控 + L1 LLM 红线 Rubric 裁判，不通过按条目自动修复
- **代码索引** — Tree-sitter 解析 + 语义搜索，理解整个项目
- **记忆系统** — 记住项目规范、用户偏好，越用越懂你
- **权限控制** — 危险操作需确认，白名单免打扰

## 开发

```bash
# 克隆项目
git clone https://github.com/yourname/mitkii.git
cd mitkii

# 安装开发依赖
uv sync --dev

# 运行测试
uv run pytest

# 本地运行
uv run mitkii chat
```

## 项目结构

详见 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)

## 架构说明

详见 [VIBECODING_PROMPT.md](./VIBECODING_PROMPT.md)

## 技术栈

- Python 3.12+ / asyncio
- LiteLLM (多模型统一调用)
- Typer + Rich (CLI)
- Tree-sitter (代码解析)
- SQLite + sqlite-vec (本地存储+向量搜索)
- ripgrep (代码搜索)
