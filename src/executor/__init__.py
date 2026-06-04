"""Subtask-scoped ReAct executor."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.executor.subtask_executor import SubTaskExecutor, SubTaskResult

__all__ = ["SubTaskExecutor", "SubTaskResult"]


def __getattr__(name: str):
    if name in __all__:
        from src.executor import subtask_executor as mod

        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
