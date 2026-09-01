from __future__ import annotations

from collections import Counter
from pathlib import Path

from agent_project.agent_rl.filesystem.schemas import FileTask


def load_file_tasks(path: str | Path) -> list[FileTask]:
    dataset_path = Path(path).expanduser().resolve()
    tasks: list[FileTask] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            tasks.append(FileTask.model_validate_json(stripped))
        except Exception as exc:
            raise ValueError(
                f"Invalid file RL task at {dataset_path}:{line_number}: {exc}"
            ) from exc
    id_counts = Counter(task.id for task in tasks)
    duplicate_ids = {task_id for task_id, count in id_counts.items() if count > 1}
    if duplicate_ids:
        raise ValueError(f"Duplicate task IDs: {', '.join(sorted(duplicate_ids))}")
    return tasks
