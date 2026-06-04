from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent.events import AgentEvent

# JSON-RPC method names
SEND_MESSAGE = "sendMessage"
APPROVAL_RESPONSE = "approvalResponse"
ROLLBACK = "rollback"
GET_STATUS = "getStatus"
GET_HISTORY = "getHistory"
CHECKPOINT = "checkpoint"
COMPACT = "compact"
SCORE = "score"
CANCEL = "cancel"

# Event notification method
EVENT = "event"


@dataclass
class JsonRpcRequest:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None
    jsonrpc: str = "2.0"


@dataclass
class JsonRpcError:
    code: int
    message: str
    data: Any = None


@dataclass
class JsonRpcResponse:
    id: str | int | None = None
    result: Any = None
    error: JsonRpcError | None = None
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = asdict(self.error)
        else:
            d["result"] = self.result
        return json.dumps(d)


def parse_request(data: str) -> JsonRpcRequest:
    obj = json.loads(data)
    return JsonRpcRequest(
        method=obj.get("method", ""),
        params=obj.get("params", {}),
        id=obj.get("id"),
        jsonrpc=obj.get("jsonrpc", "2.0"),
    )


def format_event(event: AgentEvent) -> str:
    return json.dumps({
        "jsonrpc": "2.0",
        "method": EVENT,
        "params": {
            "id": event.id,
            "type": event.type.value,
            "content": event.content,
            "data": event.data,
            "timestamp": event.timestamp,
        },
    })


def error_response(request_id: Any, code: int, message: str) -> JsonRpcResponse:
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(code=code, message=message),
    )
