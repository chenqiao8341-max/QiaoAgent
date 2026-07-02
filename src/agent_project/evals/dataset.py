from __future__ import annotations

import json
from pathlib import Path

from agent_project.evals.schemas import EvalExample


def load_dataset(path: str | Path) -> list[EvalExample]:
    dataset_path = Path(path).expanduser().resolve()
    examples: list[EvalExample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            examples.append(EvalExample.model_validate_json(stripped))
        except Exception as exc:
            raise ValueError(f"Invalid eval example at {dataset_path}:{line_number}: {exc}") from exc
    return examples


def write_jsonl(path: str | Path, examples: list[EvalExample]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
