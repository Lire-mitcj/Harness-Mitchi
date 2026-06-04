from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from src.server.handler import ServerHandler
from src.server.protocol import (
    PARSE_ERROR,
    INVALID_REQUEST,
    format_error,
    format_event,
    parse_request,
)

log = logging.getLogger(__name__)

_ENCODING = "utf-8"


class StdioTransport:
    """LSP-style stdin/stdout JSON-RPC transport.

    Messages are framed with ``Content-Length`` headers, matching the
    Language Server Protocol wire format. This makes MitKII compatible
    with any editor/IDE that speaks JSON-RPC over stdio.
    """

    def __init__(self, handler: ServerHandler) -> None:
        self._handler = handler
        self._running = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        """Begin reading requests from stdin and writing responses to stdout."""
        self._running = True
        self._reader = asyncio.StreamReader()
        transport = await _connect_stdin(self._reader)

        loop = asyncio.get_running_loop()
        self._writer = asyncio.StreamWriter(
            transport=asyncio.get_event_loop()._csock if hasattr(asyncio.get_event_loop(), '_csock') else None,  # type: ignore[attr-defined]
            protocol=asyncio.StreamReaderProtocol(asyncio.StreamReader()),
            reader=None,
            loop=loop,
        ) if False else None  # placeholder — use raw stdout below

        log.info("StdioTransport started")

        try:
            while self._running:
                data = await self._read_message()
                if data is None:
                    break
                await self._handle_message(data)
        except asyncio.CancelledError:
            log.info("StdioTransport cancelled")
        except Exception:
            log.exception("StdioTransport error")
        finally:
            self._running = False

    async def stop(self) -> None:
        self._running = False

    async def send_event(self, event_dict: dict[str, Any]) -> None:
        """Push an agent event as a JSON-RPC notification to stdout."""
        payload = format_event(event_dict)
        self._write_message(payload)

    async def _read_message(self) -> str | None:
        """Read a Content-Length framed message from stdin."""
        assert self._reader is not None
        content_length = 0

        while True:
            line = await self._reader.readline()
            if not line:
                return None

            header = line.decode(_ENCODING).strip()
            if not header:
                break

            if header.lower().startswith("content-length:"):
                try:
                    content_length = int(header.split(":", 1)[1].strip())
                except ValueError:
                    log.warning("Invalid Content-Length header: %s", header)
                    return None

        if content_length <= 0:
            return None

        body = await self._reader.readexactly(content_length)
        return body.decode(_ENCODING)

    async def _handle_message(self, data: str) -> None:
        try:
            request = parse_request(data)
        except ValueError as exc:
            error_msg = format_error(PARSE_ERROR, str(exc), None)
            self._write_message(error_msg)
            return

        response = await self._handler.handle(request)

        if not request.is_notification:
            import json
            self._write_message(json.dumps(response.to_dict(), ensure_ascii=False))

    @staticmethod
    def _write_message(content: str) -> None:
        """Write a Content-Length framed message to stdout."""
        encoded = content.encode(_ENCODING)
        header = f"Content-Length: {len(encoded)}\r\n\r\n"
        sys.stdout.buffer.write(header.encode(_ENCODING))
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()


async def _connect_stdin(reader: asyncio.StreamReader) -> Any:
    """Connect stdin to an asyncio StreamReader."""
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        sys.stdin,
    )
    return transport
