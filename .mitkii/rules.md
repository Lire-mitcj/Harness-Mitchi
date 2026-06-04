# MitKII Project Rules

## Code Style
- Python 3.12+ with type hints on all public functions
- Use `from __future__ import annotations` in every module
- Line length limit: 100 characters
- Use `ruff` for linting and formatting

## Architecture
- All agent output must go through the Event system — never `print()` directly
- Tools must inherit from `Tool` base class and register via `ToolRegistry`
- Async by default — use `async def` for any I/O operation
- Pydantic models for configuration, dataclasses for internal data

## Agent behavior (framework reads)
- L0/L1/L2 scoring is built into the runtime — agents must **not** read `src/harness/**`, `src/agent/**`, `src/cli/**`, or `prompts/**` when writing user files
- User tasks that mention gate/L0/L1/score still only edit the target user file; do not re-implement the harness in demo scripts
- Framework source reads are allowed only when the user explicitly names the path to modify (e.g. `src/agent/loop.py`)
- During quality-gate rewrite: use `write_file`/`edit_file` only; no `shell_exec` or framework reads

## Testing
- All new features need at least one test
- Use `pytest` + `pytest-asyncio`
- Mock LLM calls in tests — never hit real APIs

## Git
- Atomic commits with clear messages
- No secrets in commits
