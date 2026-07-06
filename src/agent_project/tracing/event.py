from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    content: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    duration_ms: int = 0
    parent_event_id: int | None = None
    node: str = ""
    raw_error: str = ""
    tool_call_id: str = ""
    token_usage: dict[str, Any] = field(default_factory=dict)
    prompt_chars: int = 0
