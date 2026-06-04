from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, Callable, TypeVar

from src.harness.sandbox.file_guard import FileGuard
from src.harness.sandbox.resource_limit import ResourceLimiter

log = logging.getLogger(__name__)

T = TypeVar("T")


class SandboxExecutor:
    """Runs functions with resource limits and working-directory isolation."""

    def __init__(
        self,
        file_guard: FileGuard | None = None,
        limiter: ResourceLimiter | None = None,
        default_timeout: float = 30.0,
    ) -> None:
        self._guard = file_guard or FileGuard()
        self._limiter = limiter or ResourceLimiter()
        self._default_timeout = default_timeout

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        isolate_cwd: bool = False,
        **kwargs: Any,
    ) -> Any:
        effective_timeout = timeout or self._default_timeout
        self._limiter.check_timeout(effective_timeout)

        original_cwd = os.getcwd()
        tmp_dir: str | None = None

        try:
            if isolate_cwd:
                tmp_dir = tempfile.mkdtemp(prefix="mitkii_sandbox_")
                os.chdir(tmp_dir)

            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=effective_timeout,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=effective_timeout,
                )

            self._limiter.check_output_size(result)
            return result

        except asyncio.TimeoutError:
            log.warning("Sandbox execution timed out after %.1fs", effective_timeout)
            raise TimeoutError(
                f"Execution exceeded {effective_timeout}s timeout"
            ) from None
        finally:
            os.chdir(original_cwd)

    async def execute_shell(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Run a shell command with resource limits.

        Returns ``(returncode, stdout, stderr)``.
        """
        effective_timeout = timeout or self._default_timeout
        self._limiter.check_timeout(effective_timeout)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"Shell command exceeded {effective_timeout}s timeout"
            ) from None

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        max_out = self._limiter.max_output_bytes
        stdout = stdout[:max_out]
        stderr = stderr[:max_out]

        return proc.returncode or 0, stdout, stderr
