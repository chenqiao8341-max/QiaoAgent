from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from agent_project.agent import invoke_agent  # noqa: E402


def main() -> None:
    message = " ".join(sys.argv[1:]).strip()
    if not message:
        message = "What tools can you use?"

    print(invoke_agent(message))


if __name__ == "__main__":
    main()
