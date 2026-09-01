#!/usr/bin/env python3
"""Build focused Safety SFT and direct-action DPO corpora.

The corpus deliberately teaches the first divergent Agent action: an unsafe path request must
select ``finish`` instead of ``read_file``. Ordinary CRUD states are replayed in both stages to
reduce capability regression. Train, validation, and test tasks remain strictly separated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_project.agent_rl.filesystem.dataset import load_file_tasks
from agent_project.agent_rl.filesystem.dpo import (
    _build_task_pairs,
    _round_robin_pair_kinds,
    canonical_action_to_tool_message,
)
from agent_project.agent_rl.filesystem.rollout import FILE_POLICY_PROMPT, FILE_POLICY_TOOLS
from agent_project.agent_rl.filesystem.schemas import FileTask

SFT_TRAIN_NAME = "file_crud_safety_sft_train"
SFT_EVAL_NAME = "file_crud_safety_sft_eval"
DPO_TRAIN_NAME = "file_crud_safety_dpo_train"
DPO_EVAL_NAME = "file_crud_safety_dpo_eval"
REFUSAL_ANSWERS = (
    "无法访问工作区外路径",
    "无法访问工作区外路径；请提供工作区内的相对路径。",
)


def _initial_conversation(task: FileTask) -> list[dict[str, str]]:
    return [
        {
            "from": "human",
            "value": (
                f"Task: {task.instruction}\n"
                f"You may use at most {task.max_steps} tool calls.\n"
                "Environment status: Episode reset. The workspace contents are initially unknown."
            ),
        }
    ]


def _base_metadata(task: FileTask) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "source_split": task.split,
        "operation": task.operation,
        "template_family": task.metadata.get("template_family"),
        "safety_case": bool(task.metadata.get("safety_case", False)),
        "system": FILE_POLICY_PROMPT,
        "tools": json.dumps(FILE_POLICY_TOOLS, ensure_ascii=False),
    }


def _safety_sft_records(
    tasks: Sequence[FileTask], *, safety_language: str | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        if not task.metadata.get("safety_case", False):
            continue
        if safety_language and task.metadata.get("language") != safety_language:
            continue
        for variant, answer in enumerate(REFUSAL_ANSWERS):
            chosen = canonical_action_to_tool_message({"type": "finish", "answer": answer})
            records.append(
                {
                    "sample_id": f"{task.id}:safety-refusal:{variant}",
                    **_base_metadata(task),
                    "curriculum_group": "safety_refusal",
                    "conversations": [*_initial_conversation(task), chosen],
                }
            )
    return records


def _ordinary_state_candidates(
    tasks: Sequence[FileTask], *, quota: int, seed: int
) -> list[dict[str, Any]]:
    """Sample oracle states evenly across ordinary CRUD operations."""
    by_operation: dict[str, list[FileTask]] = defaultdict(list)
    for task in tasks:
        if not task.metadata.get("safety_case", False):
            by_operation[task.operation].append(task)
    operations = sorted(by_operation)
    if not operations or quota <= 0:
        return []
    rng = random.Random(seed)
    base, remainder = divmod(quota, len(operations))
    selected: list[dict[str, Any]] = []
    for op_index, operation in enumerate(operations):
        target = base + (1 if op_index < remainder else 0)
        operation_tasks = list(by_operation[operation])
        rng.shuffle(operation_tasks)
        candidates: list[dict[str, Any]] = []
        for task in operation_tasks:
            pairs = _build_task_pairs(task)
            first_pair_by_state: dict[int, dict[str, Any]] = {}
            for pair in pairs:
                first_pair_by_state.setdefault(pair["reference_step"], pair)
            for state, pair in sorted(first_pair_by_state.items()):
                candidates.append(
                    {
                        "sample_id": f"{task.id}:crud-state:{state}",
                        **_base_metadata(task),
                        "curriculum_group": "crud_replay",
                        "conversations": [*pair["conversations"], pair["chosen"]],
                    }
                )
            if len(candidates) >= target * 2:
                break
        rng.shuffle(candidates)
        selected.extend(candidates[:target])
    rng.shuffle(selected)
    if len(selected) != quota:
        raise RuntimeError(f"Only generated {len(selected)} of {quota} requested SFT replay states")
    return selected


def build_sft_records(
    tasks: Sequence[FileTask],
    *,
    safety_quota: int,
    replay_quota: int,
    seed: int,
    safety_language: str | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    safety = _safety_sft_records(tasks, safety_language=safety_language)
    rng.shuffle(safety)
    if len(safety) < safety_quota:
        raise RuntimeError(f"Only {len(safety)} Safety SFT records for quota {safety_quota}")
    records = safety[:safety_quota]
    records.extend(_ordinary_state_candidates(tasks, quota=replay_quota, seed=seed + 1))
    rng.shuffle(records)
    return records


def _safety_direct_dpo_pairs(
    tasks: Sequence[FileTask], *, safety_language: str | None = None
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for task in tasks:
        if not task.metadata.get("safety_case", False):
            continue
        if safety_language and task.metadata.get("language") != safety_language:
            continue
        direct = next(
            pair for pair in _build_task_pairs(task) if pair["pair_kind"] == "safety_unsafe_read"
        )
        for variant, answer in enumerate(REFUSAL_ANSWERS):
            chosen = canonical_action_to_tool_message({"type": "finish", "answer": answer})
            identity = f"{direct['pair_id']}:direct-refusal:{variant}"
            pairs.append(
                {
                    **direct,
                    "pair_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                    "chosen": chosen,
                    "pair_kind": "safety_finish_vs_unsafe_read",
                    "curriculum_group": "safety_direct_action",
                }
            )
    return pairs


def _ordinary_dpo_replay(
    tasks: Sequence[FileTask], *, quota: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_operation: dict[str, list[FileTask]] = defaultdict(list)
    for task in tasks:
        if not task.metadata.get("safety_case", False):
            by_operation[task.operation].append(task)
    operations = sorted(by_operation)
    base, remainder = divmod(quota, len(operations))
    selected: list[dict[str, Any]] = []
    for op_index, operation in enumerate(operations):
        target = base + (1 if op_index < remainder else 0)
        operation_tasks = list(by_operation[operation])
        rng.shuffle(operation_tasks)
        candidates: list[dict[str, Any]] = []
        for task in operation_tasks:
            candidates.extend(_build_task_pairs(task))
            if len(candidates) >= target * 3:
                break
        sampled = _round_robin_pair_kinds(candidates, quota=target, rng=rng)
        for pair in sampled:
            pair["curriculum_group"] = "crud_replay"
        selected.extend(sampled)
    rng.shuffle(selected)
    if len(selected) != quota:
        raise RuntimeError(f"Only generated {len(selected)} of {quota} requested DPO replay pairs")
    return selected


def build_dpo_pairs(
    tasks: Sequence[FileTask],
    *,
    safety_quota: int,
    replay_quota: int,
    seed: int,
    safety_language: str | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    safety = _safety_direct_dpo_pairs(tasks, safety_language=safety_language)
    rng.shuffle(safety)
    if len(safety) < safety_quota:
        raise RuntimeError(f"Only {len(safety)} direct Safety pairs for quota {safety_quota}")
    pairs = safety[:safety_quota]
    pairs.extend(_ordinary_dpo_replay(tasks, quota=replay_quota, seed=seed + 1))
    rng.shuffle(pairs)
    return pairs


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")
    return payload


def _split_summary(records: Sequence[dict[str, Any]], payload: str) -> dict[str, Any]:
    return {
        "count": len(records),
        "unique_tasks": len({record["task_id"] for record in records}),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "operations": dict(sorted(Counter(record["operation"] for record in records).items())),
        "curriculum_groups": dict(
            sorted(Counter(record["curriculum_group"] for record in records).items())
        ),
    }


def write_sft_corpus(
    output_dir: Path, train: Sequence[dict[str, Any]], eval_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_name = f"{SFT_TRAIN_NAME}.jsonl"
    eval_name = f"{SFT_EVAL_NAME}.jsonl"
    train_payload = _write_jsonl(output_dir / train_name, train)
    eval_payload = _write_jsonl(output_dir / eval_name, eval_records)
    columns = {"messages": "conversations", "system": "system", "tools": "tools"}
    info = {
        SFT_TRAIN_NAME: {
            "file_name": train_name,
            "formatting": "sharegpt",
            "columns": columns,
        },
        SFT_EVAL_NAME: {
            "file_name": eval_name,
            "formatting": "sharegpt",
            "columns": columns,
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "version": "file-crud-safety-sft-v1",
        "train": _split_summary(train, train_payload),
        "eval": _split_summary(eval_records, eval_payload),
        "task_overlap": sorted(
            {record["task_id"] for record in train}
            & {record["task_id"] for record in eval_records}
        ),
    }


def write_dpo_corpus(
    output_dir: Path, train: Sequence[dict[str, Any]], eval_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_name = f"{DPO_TRAIN_NAME}.jsonl"
    eval_name = f"{DPO_EVAL_NAME}.jsonl"
    train_payload = _write_jsonl(output_dir / train_name, train)
    eval_payload = _write_jsonl(output_dir / eval_name, eval_records)
    columns = {
        "messages": "conversations",
        "system": "system",
        "tools": "tools",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    info = {
        DPO_TRAIN_NAME: {
            "file_name": train_name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": columns,
        },
        DPO_EVAL_NAME: {
            "file_name": eval_name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": columns,
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "version": "file-crud-safety-dpo-v2",
        "train": _split_summary(train, train_payload),
        "eval": _split_summary(eval_records, eval_payload),
        "pair_kinds": {
            "train": dict(sorted(Counter(record["pair_kind"] for record in train).items())),
            "eval": dict(sorted(Counter(record["pair_kind"] for record in eval_records).items())),
        },
        "task_overlap": sorted(
            {record["task_id"] for record in train}
            & {record["task_id"] for record in eval_records}
        ),
    }
    return manifest


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--sft-output-dir", required=True, type=Path)
    parser.add_argument("--dpo-output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=35_402)
    parser.add_argument("--safety-language", choices=["zh", "en"])
    parser.add_argument("--sft-train-safety", type=int, default=800)
    parser.add_argument("--sft-train-replay", type=int, default=800)
    parser.add_argument("--sft-eval-safety", type=int, default=100)
    parser.add_argument("--sft-eval-replay", type=int, default=100)
    parser.add_argument("--dpo-train-safety", type=int, default=800)
    parser.add_argument("--dpo-train-replay", type=int, default=800)
    parser.add_argument("--dpo-eval-safety", type=int, default=100)
    parser.add_argument("--dpo-eval-replay", type=int, default=100)
    args = parser.parse_args()

    train_tasks = load_file_tasks(args.train_dataset)
    validation_tasks = load_file_tasks(args.validation_dataset)
    sft_train = build_sft_records(
        train_tasks,
        safety_quota=args.sft_train_safety,
        replay_quota=args.sft_train_replay,
        seed=args.seed,
        safety_language=args.safety_language,
    )
    sft_eval = build_sft_records(
        validation_tasks,
        safety_quota=args.sft_eval_safety,
        replay_quota=args.sft_eval_replay,
        seed=args.seed + 10,
        safety_language=args.safety_language,
    )
    dpo_train = build_dpo_pairs(
        train_tasks,
        safety_quota=args.dpo_train_safety,
        replay_quota=args.dpo_train_replay,
        seed=args.seed + 20,
        safety_language=args.safety_language,
    )
    dpo_eval = build_dpo_pairs(
        validation_tasks,
        safety_quota=args.dpo_eval_safety,
        replay_quota=args.dpo_eval_replay,
        seed=args.seed + 30,
        safety_language=args.safety_language,
    )
    sft_manifest = write_sft_corpus(args.sft_output_dir, sft_train, sft_eval)
    dpo_manifest = write_dpo_corpus(args.dpo_output_dir, dpo_train, dpo_eval)
    _write_manifest(args.sft_output_dir, sft_manifest)
    _write_manifest(args.dpo_output_dir, dpo_manifest)
    print(json.dumps({"sft": sft_manifest, "dpo": dpo_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
