from __future__ import annotations

import sys
from typing import Any

_MB = 1024 * 1024


class ResourceLimiter:
    """Enforces execution-time, output-size, and file-size limits."""

    def __init__(
        self,
        max_execution_seconds: float = 120.0,
        max_output_bytes: int = 5 * _MB,
        max_file_bytes: int = 10 * _MB,
    ) -> None:
        self.max_execution_seconds = max_execution_seconds
        self.max_output_bytes = max_output_bytes
        self.max_file_bytes = max_file_bytes

    def check_timeout(self, requested: float) -> None:
        """Raise if *requested* timeout exceeds the configured maximum."""
        if requested > self.max_execution_seconds:
            raise ValueError(
                f"Requested timeout {requested}s exceeds limit "
                f"of {self.max_execution_seconds}s"
            )

    def check_output_size(self, output: Any) -> None:
        """Raise if *output* is unreasonably large."""
        if output is None:
            return
        if isinstance(output, (str, bytes)):
            size = len(output) if isinstance(output, bytes) else len(output.encode("utf-8", errors="replace"))
        else:
            size = sys.getsizeof(output)

        if size > self.max_output_bytes:
            raise ValueError(
                f"Output size {size:,} bytes exceeds limit "
                f"of {self.max_output_bytes:,} bytes"
            )

    def check_file_size(self, size_bytes: int) -> None:
        """Raise if a file write would exceed the configured limit."""
        if size_bytes > self.max_file_bytes:
            raise ValueError(
                f"File size {size_bytes:,} bytes exceeds limit "
                f"of {self.max_file_bytes:,} bytes"
            )
