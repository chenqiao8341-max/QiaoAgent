from __future__ import annotations

import os
import sys


def progress_enabled() -> bool:
    return os.getenv("AGENT_SHOW_TOOL_PROGRESS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def emit_progress(message: str) -> None:
    """Print a concise tool progress event without polluting tool return values."""
    if not progress_enabled():
        return
    print(f"[agent] {message}", file=sys.stderr, flush=True)
