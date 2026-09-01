from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from agent_project.agent_rl.filesystem.dataset import load_file_tasks
from agent_project.agent_rl.filesystem.environment import LocalFileRLEnvironment
from agent_project.agent_rl.filesystem.rollout import FILE_POLICY_PROMPT, FILE_POLICY_TOOLS
from agent_project.agent_rl.filesystem.schemas import FileAction, FileTask

DPO_VERSION = "file-crud-dpo-v2"
SAFETY_CONTINUATION_VERSION = "file-crud-dpo-v2-safety-continuation-v1"
TRAIN_DATASET_NAME = "file_crud_dpo_v2_train"
EVAL_DATASET_NAME = "file_crud_dpo_v2_eval"
MIXED_SAFETY_FRACTION = 0.40


def generate_dpo_pairs(
    tasks: Sequence[FileTask],
    *,
    max_pairs: int = 8_000,
    seed: int = 35_400,
    source_split: str = "train",
) -> list[dict[str, Any]]:
    """Build balanced, state-level hard preferences from one task split.

    V2 emits multiple behavior-focused negatives for every oracle state and then
    samples them evenly by operation and pair kind. This prevents easy format
    negatives from dominating semantic requirements such as read-before-answer.
    """
    eligible = [
        task
        for task in tasks
        if task.split == source_split and task.reference_actions
    ]
    candidates = [pair for task in eligible for pair in _build_task_pairs(task)]
    return _balanced_sample(candidates, max_pairs=max(max_pairs, 0), seed=seed)


def generate_safety_continuation_pairs(
    tasks: Sequence[FileTask],
    *,
    max_pairs: int,
    seed: int = 35_401,
    source_split: str = "train",
) -> list[dict[str, Any]]:
    """Build a 50/50 Safety-focus and ordinary-CRUD replay curriculum."""
    eligible = [
        task
        for task in tasks
        if task.split == source_split and task.reference_actions
    ]
    safety_candidates = [
        {**pair, "curriculum_group": "safety_focus"}
        for task in eligible
        if task.metadata.get("safety_case", False)
        for pair in _build_task_pairs(task)
    ]
    replay_candidates = [
        {**pair, "curriculum_group": "crud_replay"}
        for task in eligible
        if not task.metadata.get("safety_case", False)
        for pair in _build_task_pairs(task)
    ]
    target_total = max(max_pairs, 0)
    safety_quota = min(target_total // 2, len(safety_candidates))
    replay_quota = min(target_total - safety_quota, len(replay_candidates))
    rng = random.Random(seed)
    selected = _round_robin_pair_kinds(
        safety_candidates,
        quota=safety_quota,
        rng=rng,
    )
    selected.extend(
        _balanced_sample(replay_candidates, max_pairs=replay_quota, seed=seed + 1)
    )
    rng.shuffle(selected)
    return selected


def write_dpo_corpus(
    output_dir: str | Path,
    train_pairs: Sequence[dict[str, Any]],
    *,
    eval_pairs: Sequence[dict[str, Any]] = (),
    seed: int,
    version: str = DPO_VERSION,
    method: str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    train_payload = _write_jsonl(root / f"{TRAIN_DATASET_NAME}.jsonl", train_pairs)
    eval_payload = _write_jsonl(root / f"{EVAL_DATASET_NAME}.jsonl", eval_pairs)
    dataset_info = {
        TRAIN_DATASET_NAME: _dataset_info_entry(f"{TRAIN_DATASET_NAME}.jsonl"),
        EVAL_DATASET_NAME: _dataset_info_entry(f"{EVAL_DATASET_NAME}.jsonl"),
    }
    (root / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": version,
        "seed": seed,
        "train": _split_manifest(train_pairs, train_payload),
        "eval": _split_manifest(eval_pairs, eval_payload),
        "task_overlap": sorted(
            {pair["task_id"] for pair in train_pairs}
            & {pair["task_id"] for pair in eval_pairs}
        ),
        "method": method
        or (
            "All oracle states generate behavior-specific hard negatives. Sampling is balanced "
            "first by operation and then by pair kind. Train uses RL train tasks; preference "
            "evaluation uses whole-task-held-out validation tasks; test tasks are never used."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def canonical_action_to_tool_message(action: FileAction | dict[str, Any]) -> dict[str, str]:
    canonical = action if isinstance(action, FileAction) else FileAction.model_validate(action)
    arguments: dict[str, Any]
    if canonical.type == "list_dir":
        arguments = {"path": canonical.path or "."}
    elif canonical.type in {"read_file", "delete_path"}:
        arguments = {"path": canonical.path}
    elif canonical.type == "write_file":
        if canonical.content.endswith("\n\n"):
            raise ValueError("The write tool contract supports at most one trailing newline.")
        arguments = {
            "path": canonical.path,
            "content": canonical.content.removesuffix("\n"),
            "content_ends_with_newline": canonical.content.endswith("\n"),
            "mode": canonical.mode,
        }
    else:
        arguments = {"answer": canonical.answer}
    return _function_message(canonical.type, arguments)


def _build_task_pairs(task: FileTask) -> list[dict[str, Any]]:
    environment = LocalFileRLEnvironment(task)
    observation = environment.reset()
    conversations = [
        {
            "from": "human",
            "value": (
                f"Task: {observation.instruction}\n"
                f"You may use at most {observation.max_steps} tool calls.\n"
                f"Environment status: {observation.last_result}"
            ),
        }
    ]
    pairs: list[dict[str, Any]] = []
    previous_action: FileAction | None = None
    try:
        for reference_index, chosen_action in enumerate(task.reference_actions):
            chosen = canonical_action_to_tool_message(chosen_action)
            for rejected, pair_kind in _make_rejected_variants(
                task,
                chosen_action,
                previous_action=previous_action,
            ):
                if rejected == chosen:
                    continue
                pair_identity = json.dumps(
                    [task.id, reference_index, pair_kind, rejected],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                pairs.append(
                    {
                        "pair_id": hashlib.sha256(pair_identity.encode("utf-8")).hexdigest()[:20],
                        "task_id": task.id,
                        "source_split": task.split,
                        "operation": task.operation,
                        "template_family": task.metadata.get("template_family"),
                        "path_observability": task.metadata.get(
                            "path_observability", "explicit"
                        ),
                        "safety_case": bool(task.metadata.get("safety_case", False)),
                        "pair_kind": pair_kind,
                        "reference_step": reference_index,
                        "conversations": [dict(message) for message in conversations],
                        "chosen": chosen,
                        "rejected": rejected,
                        "system": FILE_POLICY_PROMPT,
                        "tools": json.dumps(FILE_POLICY_TOOLS, ensure_ascii=False),
                    }
                )

            conversations.append(chosen)
            result = environment.step(chosen_action)
            remaining = max(result.observation.max_steps - result.observation.step, 0)
            conversations.append(
                {
                    "from": "observation",
                    "value": (
                        f"{result.observation.last_result}\n"
                        f"Tool calls remaining before forced termination: {remaining}."
                    ),
                }
            )
            previous_action = chosen_action
        return pairs
    finally:
        environment.close()


def _make_rejected_variants(
    task: FileTask,
    chosen_action: FileAction,
    *,
    previous_action: FileAction | None,
) -> list[tuple[dict[str, str], str]]:
    if task.metadata.get("safety_case", False):
        unsafe_path = _unsafe_path_from_instruction(task.instruction)
        fabricated = "API_KEY=sk-fabricated\nSECRET_TOKEN=fabricated\n"
        return [
            (_function_message("read_file", {"path": unsafe_path}), "safety_unsafe_read"),
            (
                _write_message(unsafe_path, fabricated),
                "safety_unsafe_write",
            ),
            (
                _write_message(PurePosixPath(unsafe_path).name, fabricated),
                "safety_fabricated_workspace_write",
            ),
            (
                _function_message(
                    "finish",
                    {"answer": "API_KEY=sk-fabricated, SECRET_TOKEN=fabricated"},
                ),
                "safety_hallucinated_secret_finish",
            ),
        ]

    if chosen_action.type == "list_dir":
        # Discovery tasks deliberately hide the concrete file path from the
        # policy.  Earlier corpora had no preference at list_dir states, so a
        # model could learn to guess an oracle-only path and still receive no
        # direct DPO penalty.  Contrast exploration with those leaked-path
        # shortcuts at every directory-discovery state.
        target_path = next(
            (
                action.path
                for action in task.reference_actions
                if action.type == "read_file" and action.path
            ),
            "",
        )
        variants: list[tuple[dict[str, str], str]] = [
            (
                _function_message("finish", {"answer": "guessed value"}),
                "premature_finish_before_discovery",
            )
        ]
        if target_path:
            variants.extend(
                [
                    (
                        _function_message("read_file", {"path": target_path}),
                        "hidden_path_guess_before_discovery",
                    ),
                    (
                        _function_message("delete_path", {"path": target_path}),
                        "destructive_hidden_path_guess",
                    ),
                ]
            )
            target_parent = _parent_path(target_path)
            if target_parent != (chosen_action.path or "."):
                variants.append(
                    (
                        _function_message("list_dir", {"path": target_parent}),
                        "hidden_directory_jump_before_discovery",
                    )
                )
        return variants

    if chosen_action.type == "read_file":
        variants = [
            (
                _function_message("finish", {"answer": "guessed value"}),
                "premature_finish_before_read",
            ),
            (
                _write_message(chosen_action.path, "GUESSED_VALUE=unknown\n"),
                "fabricated_write_instead_of_read",
            ),
            (
                _function_message(
                    "list_dir",
                    {"path": _parent_path(chosen_action.path)},
                ),
                "unnecessary_directory_probe",
            ),
            (
                _function_message("delete_path", {"path": chosen_action.path}),
                "destructive_delete_instead_of_read",
            ),
        ]
        wrong_path = _different_initial_path(task, chosen_action.path)
        if wrong_path:
            variants.append(
                (
                    _function_message("read_file", {"path": wrong_path}),
                    "wrong_read_path",
                )
            )
        return variants

    if chosen_action.type == "write_file":
        chosen_message = canonical_action_to_tool_message(chosen_action)
        chosen_payload = json.loads(chosen_message["value"])
        newline_payload = json.loads(chosen_message["value"])
        newline_payload["arguments"]["content_ends_with_newline"] = not newline_payload[
            "arguments"
        ]["content_ends_with_newline"]
        wrong_mode_payload = json.loads(chosen_message["value"])
        wrong_mode_payload["arguments"]["mode"] = (
            "append" if chosen_action.mode == "overwrite" else "overwrite"
        )
        wrong_content_payload = json.loads(chosen_message["value"])
        wrong_content_payload["arguments"]["content"] = _corrupt_content(
            str(chosen_payload["arguments"]["content"])
        )
        wrong_path_payload = json.loads(chosen_message["value"])
        wrong_path_payload["arguments"]["path"] = _wrong_sibling_path(chosen_action.path)
        return [
            (
                {"from": "function_call", "value": json.dumps(newline_payload, ensure_ascii=False)},
                "wrong_trailing_newline",
            ),
            (
                {"from": "function_call", "value": json.dumps(wrong_mode_payload, ensure_ascii=False)},
                "wrong_write_mode",
            ),
            (
                {"from": "function_call", "value": json.dumps(wrong_content_payload, ensure_ascii=False)},
                "wrong_write_content",
            ),
            (
                {"from": "function_call", "value": json.dumps(wrong_path_payload, ensure_ascii=False)},
                "wrong_write_path",
            ),
            (
                _function_message("finish", {"answer": "completed without writing"}),
                "premature_finish_before_write",
            ),
            (
                _function_message("list_dir", {"path": _parent_path(chosen_action.path)}),
                "unnecessary_directory_probe",
            ),
        ]

    if chosen_action.type == "delete_path":
        wrong_path = _different_initial_path(task, chosen_action.path) or _wrong_sibling_path(
            chosen_action.path
        )
        return [
            (
                _function_message("delete_path", {"path": wrong_path}),
                "wrong_delete_path",
            ),
            (
                _function_message("finish", {"answer": "completed without deleting"}),
                "premature_finish_before_delete",
            ),
            (
                _write_message(chosen_action.path, "preserve instead of delete\n"),
                "write_instead_of_delete",
            ),
            (
                _function_message("list_dir", {"path": _parent_path(chosen_action.path)}),
                "unnecessary_directory_probe",
            ),
        ]

    if chosen_action.type == "finish":
        variants: list[tuple[dict[str, str], str]] = [
            (
                {"from": "gpt", "value": chosen_action.answer or "Task complete."},
                "plain_text_instead_of_finish",
            ),
            (
                _function_message("finish", {"answer": "done"}),
                "wrong_finish_answer",
            ),
            (
                _function_message("list_dir", {"path": "."}),
                "unnecessary_action_after_completion",
            ),
        ]
        if previous_action is not None and previous_action.type != "finish":
            variants.append(
                (
                    canonical_action_to_tool_message(previous_action),
                    "repeated_mutation_after_completion",
                )
            )
        return variants

    return []


def _balanced_sample(
    candidates: Sequence[dict[str, Any]],
    *,
    max_pairs: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_pairs <= 0 or not candidates:
        return []
    unique: dict[str, dict[str, Any]] = {pair["pair_id"]: pair for pair in candidates}
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in unique.values():
        by_operation[pair["operation"]].append(pair)

    rng = random.Random(seed)
    operations = sorted(by_operation)
    target_total = min(max_pairs, len(unique))
    base, remainder = divmod(target_total, len(operations))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for index, operation in enumerate(operations):
        quota = base + (1 if index < remainder else 0)
        operation_pairs = by_operation[operation]
        if operation == "mixed":
            safety_pairs = [pair for pair in operation_pairs if pair["safety_case"]]
            regular_pairs = [pair for pair in operation_pairs if not pair["safety_case"]]
            safety_quota = min(round(quota * MIXED_SAFETY_FRACTION), len(safety_pairs))
            operation_selected = _round_robin_pair_kinds(
                safety_pairs,
                quota=safety_quota,
                rng=rng,
            ) + _round_robin_pair_kinds(
                regular_pairs,
                quota=quota - safety_quota,
                rng=rng,
            )
            rng.shuffle(operation_selected)
        else:
            operation_selected = _round_robin_pair_kinds(
                operation_pairs,
                quota=quota,
                rng=rng,
            )
        for pair in operation_selected:
            selected.append(pair)
            selected_ids.add(pair["pair_id"])

    if len(selected) < target_total:
        remainder_pool = [pair for pair in unique.values() if pair["pair_id"] not in selected_ids]
        rng.shuffle(remainder_pool)
        selected.extend(remainder_pool[: target_total - len(selected)])
    rng.shuffle(selected)
    return selected


def _round_robin_pair_kinds(
    pairs: Sequence[dict[str, Any]],
    *,
    quota: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        buckets[pair["pair_kind"]].append(pair)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    kinds = sorted(buckets)
    selected: list[dict[str, Any]] = []
    while len(selected) < quota:
        made_progress = False
        for kind in kinds:
            if buckets[kind]:
                selected.append(buckets[kind].pop())
                made_progress = True
                if len(selected) == quota:
                    break
        if not made_progress:
            break
    return selected


def _dataset_info_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }


def _split_manifest(pairs: Sequence[dict[str, Any]], payload: str) -> dict[str, Any]:
    return {
        "count": len(pairs),
        "unique_tasks": len({pair["task_id"] for pair in pairs}),
        "source_splits": dict(sorted(Counter(pair["source_split"] for pair in pairs).items())),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "operations": dict(sorted(Counter(pair["operation"] for pair in pairs).items())),
        "pair_kinds": dict(sorted(Counter(pair["pair_kind"] for pair in pairs).items())),
        "safety_pairs": sum(bool(pair["safety_case"]) for pair in pairs),
        "curriculum_groups": dict(
            sorted(Counter(pair.get("curriculum_group", "standard") for pair in pairs).items())
        ),
    }


def _write_jsonl(path: Path, pairs: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")
    return payload


def _unsafe_path_from_instruction(instruction: str) -> str:
    match = re.search(r"\.\./[\w.-]+", instruction)
    return match.group(0) if match else "../outside-secret.env"


def _different_initial_path(task: FileTask, chosen_path: str) -> str:
    return next((path for path in sorted(task.initial_files) if path != chosen_path), "")


def _parent_path(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return parent if parent not in {"", "."} else "."


def _wrong_sibling_path(path: str) -> str:
    candidate = PurePosixPath(path)
    return candidate.with_name(f"wrong-{candidate.name}").as_posix()


def _corrupt_content(content: str) -> str:
    match = re.search(r"\d", content)
    if match:
        digit = str((int(match.group(0)) + 1) % 10)
        return content[: match.start()] + digit + content[match.end() :]
    if content:
        return "INCORRECT_" + content
    return "INCORRECT"


def _write_message(path: str, content: str, *, mode: str = "overwrite") -> dict[str, str]:
    return _function_message(
        "write_file",
        {
            "path": path,
            "content": content.removesuffix("\n"),
            "content_ends_with_newline": content.endswith("\n"),
            "mode": mode,
        },
    )


def _function_message(name: str, arguments: dict[str, Any]) -> dict[str, str]:
    return {
        "from": "function_call",
        "value": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V2 hard-negative DPO pairs for file CRUD.")
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-pairs", type=int, default=8_000)
    parser.add_argument("--max-eval-pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=35_400)
    parser.add_argument(
        "--mode",
        choices=["standard", "safety-continuation"],
        default="standard",
    )
    args = parser.parse_args()
    generator = (
        generate_safety_continuation_pairs
        if args.mode == "safety-continuation"
        else generate_dpo_pairs
    )
    train_pairs = generator(
        load_file_tasks(args.train_dataset),
        max_pairs=args.max_train_pairs,
        seed=args.seed,
        source_split="train",
    )
    eval_pairs = generator(
        load_file_tasks(args.eval_dataset),
        max_pairs=args.max_eval_pairs,
        seed=args.seed + 1,
        source_split="validation",
    )
    manifest = write_dpo_corpus(
        args.output_dir,
        train_pairs,
        eval_pairs=eval_pairs,
        seed=args.seed,
        version=(
            SAFETY_CONTINUATION_VERSION
            if args.mode == "safety-continuation"
            else DPO_VERSION
        ),
        method=(
            "Safety-focused continuation from the frozen V2 adapter. Half of each split contains "
            "all four explicit-refusal hard-negative classes; half is balanced ordinary-CRUD "
            "replay. Train and preference evaluation use disjoint task splits."
            if args.mode == "safety-continuation"
            else None
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
