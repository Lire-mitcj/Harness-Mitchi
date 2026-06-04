from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.server.protocol import (
    APPROVAL_RESPONSE,
    CANCEL,
    CHECKPOINT,
    COMPACT,
    GET_HISTORY,
    GET_STATUS,
    ROLLBACK,
    SCORE,
    SEND_MESSAGE,
    JsonRpcRequest,
    JsonRpcResponse,
    error_response,
    format_event,
)

if TYPE_CHECKING:
    from src.agent.loop import AgentLoop


class ServerHandler:
    """Routes JSON-RPC requests to agent actions and emits events."""

    def __init__(self, agent_loop: AgentLoop) -> None:
        self.agent = agent_loop
        self._pending_approval: Any = None

    async def handle(self, request: JsonRpcRequest) -> JsonRpcResponse:
        handler_map = {
            SEND_MESSAGE: self._handle_send_message,
            APPROVAL_RESPONSE: self._handle_approval,
            ROLLBACK: self._handle_rollback,
            GET_STATUS: self._handle_status,
            GET_HISTORY: self._handle_history,
            CHECKPOINT: self._handle_checkpoint,
            COMPACT: self._handle_compact,
            SCORE: self._handle_score,
            CANCEL: self._handle_cancel,
        }

        handler = handler_map.get(request.method)
        if handler is None:
            return error_response(request.id, -32601, f"Method not found: {request.method}")

        try:
            result = await handler(request.params)
            return JsonRpcResponse(id=request.id, result=result)
        except Exception as e:
            return error_response(request.id, -32000, str(e))

    async def _handle_send_message(self, params: dict) -> dict:
        text = params.get("text", "")
        return {"type": "stream_start", "message": text}

    async def _handle_approval(self, params: dict) -> dict:
        approved = params.get("approved", False)
        return {"approved": approved}

    async def _handle_rollback(self, params: dict) -> dict:
        checkpoint_id = params.get("checkpoint_id")
        return {"status": "rolled_back", "checkpoint_id": checkpoint_id}

    async def _handle_status(self, params: dict) -> dict:
        repo_map = "disabled"
        agent = self.agent
        if hasattr(agent, "_loop") and hasattr(agent._loop, "context"):
            service = getattr(agent._loop.context, "repo_map_service", None)
            if service is not None:
                repo_map = service.build_state.value
        return {"status": "ready", "repo_map": repo_map}

    async def _handle_history(self, params: dict) -> dict:
        return {"sessions": []}

    async def _handle_checkpoint(self, params: dict) -> dict:
        return {"status": "saved"}

    async def _handle_compact(self, params: dict) -> dict:
        return {"status": "compacted"}

    async def _handle_score(self, params: dict) -> dict:
        return {"status": "scored"}

    async def _handle_cancel(self, params: dict) -> dict:
        return {"status": "cancelled"}
