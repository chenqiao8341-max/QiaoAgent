#!/usr/bin/env python3
"""Validate V3 split isolation, observability invariants, and every oracle episode."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_project.agent_rl.filesystem.dataset import load_file_tasks
from agent_project.agent_rl.filesystem.rollout import (
    ReferenceFilePolicy,
    iter_file_dataset_concurrently,
    summarize_file_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    split_tasks = {
        split: load_file_tasks(args.dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    split_ids = {split: {task.id for task in tasks} for split, tasks in split_tasks.items()}
    overlaps = {
        "train_validation": sorted(split_ids["train"] & split_ids["validation"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "validation_test": sorted(split_ids["validation"] & split_ids["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Task ID leakage across splits: {overlaps}")

    report: dict[str, object] = {"version": "file-crud-rl-v3", "overlaps": overlaps}
    split_reports: dict[str, object] = {}
    for split, tasks in split_tasks.items():
        observability = Counter(task.metadata.get("path_observability") for task in tasks)
        discover_tasks = [
            task for task in tasks if task.metadata.get("path_observability") == "discover"
        ]
        for task in discover_tasks:
            actions = task.reference_actions
            if not actions or actions[0].type != "list_dir" or actions[0].path != ".":
                raise RuntimeError(f"Discovery oracle does not start at root: {task.id}")
            if "list_dir" not in task.required_actions:
                raise RuntimeError(f"Discovery action is not required: {task.id}")
            if task.max_steps < len(actions) + 2:
                raise RuntimeError(f"No recovery margin in max_steps: {task.id}")
            read_paths = [action.path for action in actions if action.type == "read_file"]
            if any(path in task.instruction for path in read_paths):
                raise RuntimeError(f"Discovery task exposes its target path: {task.id}")

        trajectories = list(
            iter_file_dataset_concurrently(
                tasks,
                ReferenceFilePolicy,
                concurrency=max(args.concurrency, 1),
            )
        )
        failures = [trajectory.task_id for trajectory in trajectories if not trajectory.success]
        if failures:
            raise RuntimeError(f"Reference failures in {split}: {failures[:20]}")
        split_reports[split] = {
            "count": len(tasks),
            "observability": dict(sorted(observability.items())),
            "reference": summarize_file_trajectories(trajectories),
        }

    report["splits"] = split_reports
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
