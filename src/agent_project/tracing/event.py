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
