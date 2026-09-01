#!/usr/bin/env python3
"""Build a production-template-aligned repair curriculum for file CRUD v3.

The curriculum continues from the legacy CRUD adapter instead of replacing it.
It focuses on two first-action errors observed in production-template rollouts:

* hidden file paths must be discovered with ``list_dir`` rather than guessed;
* paths outside the workspace must be rejected with ``finish``.

Ordinary explicit-path CRUD states are replayed to limit capability regression.
The Qwen3.5 tool preamble is rendered by the model's own tokenizer and embedded
in the system field.  The LLaMA-Factory ``tools`` column is intentionally not
used, because its formatter places the policy before the tool definitions while
the production tokenizer places the tool definitions first.
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

from transformers import AutoTokenizer

from agent_project.agent_rl.filesystem.dataset import load_file_tasks
from agent_project.agent_rl.filesystem.dpo import (
    _build_task_pairs,
    _round_robin_pair_kinds,
    canonical_action_to_tool_message,
)
from agent_project.agent_rl.filesystem.rollout import FILE_POLICY_PROMPT, FILE_POLICY_TOOLS
from agent_project.agent_rl.filesystem.schemas import FileTask

SFT_TRAIN_NAME = "file_crud_v3_repair_sft_train"
SFT_EVAL_NAME = "file_crud_v3_repair_sft_eval"
DPO_TRAIN_NAME = "file_crud_v3_repair_dpo_train"
DPO_EVAL_NAME = "file_crud_v3_repair_dpo_eval"


def render_official_system(model_path: str | Path) -> str:
    """Render exactly the system-body ordering used by the production tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    marker = "__QIAO_FILE_RL_TEMPLATE_PROBE__"
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": FILE_POLICY_PROMPT},
            {"role": "user", "content": marker},
        ],
        tools=FILE_POLICY_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    system_prefix = "<|im_start|>system\n"
    system_suffix = "<|im_end|>\n<|im_start|>user\n"
    if not rendered.startswith(system_prefix) or system_suffix not in rendered:
        raise RuntimeError("Unexpected Qwen3.5 tokenizer system-message rendering")
    system_body = rendered[len(system_prefix) :].split(system_suffix, 1)[0]
    if marker in system_body or "# Tools" not in system_body or FILE_POLICY_PROMPT.strip() not in system_body:
        raise RuntimeError("Official system rendering is incomplete or contains the probe")
    if system_body.index("# Tools") > system_body.index(FILE_POLICY_PROMPT.strip()):
        raise RuntimeError("Production template did not render tools before the policy")
    return system_body


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


def _metadata(task: FileTask, official_system: str) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "source_split": task.split,
        "operation": task.operation,
        "template_family": task.metadata.get("template_family"),
        "path_observability": task.metadata.get("path_observability", "explicit"),
        "safety_case": bool(task.metadata.get("safety_case", False)),
        "system": official_system,
    }


def _normalize_pair(pair: dict[str, Any], official_system: str) -> dict[str, Any]:
    normalized = dict(pair)
    normalized["system"] = official_system
    normalized.pop("tools", None)
    return normalized


def _refusal_answers(task: FileTask) -> tuple[str, str]:
    if task.metadata.get("language") == "en":
        return (
            "I cannot access paths outside the workspace.",
            "I cannot access paths outside the workspace; provide a relative in-workspace path.",
        )
    return (
        "无法访问工作区外路径。",
        "无法访问工作区外路径；请提供工作区内的相对路径。",
    )


def _safety_sft(tasks: Sequence[FileTask], official_system: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        if not task.metadata.get("safety_case", False):
            continue
        for variant, answer in enumerate(_refusal_answers(task)):
            records.append(
                {
                    "sample_id": f"{task.id}:safety-refusal:{variant}",
                    **_metadata(task, official_system),
                    "curriculum_group": "safety_first_action",
                    "conversations": [
                        *_initial_conversation(task),
                        canonical_action_to_tool_message({"type": "finish", "answer": answer}),
                    ],
                }
            )
    return records


def _discovery_sft(tasks: Sequence[FileTask], official_system: str) -> list[dict[str, Any]]:
    """Use every discovery task once, always at its initial list_dir state."""
    records: list[dict[str, Any]] = []
    for task in tasks:
        if task.operation != "read" or task.metadata.get("path_observability") != "discover":
            continue
        first_action = task.reference_actions[0]
        if first_action.type != "list_dir" or first_action.path != ".":
            raise RuntimeError(f"Discovery task {task.id} does not start with list_dir('.')")
        records.append(
            {
                "sample_id": f"{task.id}:discovery-first-action",
                **_metadata(task, official_system),
                "curriculum_group": "discovery_first_action",
                "conversations": [
                    *_initial_conversation(task),
                    canonical_action_to_tool_message(first_action),
                ],
            }
        )
    return records


def _ordinary_sft_replay(
    tasks: Sequence[FileTask], *, quota: int, seed: int, official_system: str
) -> list[dict[str, Any]]:
    operations = ("create", "update", "delete", "mixed")
    rng = random.Random(seed)
    by_operation: dict[str, list[FileTask]] = defaultdict(list)
    for task in tasks:
        if (
            task.operation in operations
            and not task.metadata.get("safety_case", False)
            and task.metadata.get("path_observability", "explicit") == "explicit"
        ):
            by_operation[task.operation].append(task)
    base, remainder = divmod(quota, len(operations))
    selected: list[dict[str, Any]] = []
    for op_index, operation in enumerate(operations):
        target = base + (op_index < remainder)
        operation_tasks = list(by_operation[operation])
        rng.shuffle(operation_tasks)
        candidates: list[dict[str, Any]] = []
        for task in operation_tasks:
            first_by_state: dict[int, dict[str, Any]] = {}
            for pair in _build_task_pairs(task):
                first_by_state.setdefault(pair["reference_step"], pair)
            for state, pair in first_by_state.items():
                candidates.append(
                    {
                        "sample_id": f"{task.id}:crud-replay:{state}",
                        **_metadata(task, official_system),
                        "curriculum_group": "crud_replay",
                        "conversations": [*pair["conversations"], pair["chosen"]],
                    }
                )
        rng.shuffle(candidates)
        selected.extend(candidates[:target])
    if len(selected) != quota:
        raise RuntimeError(f"Generated {len(selected)} of {quota} requested SFT replay records")
    rng.shuffle(selected)
    return selected


def build_sft(
    tasks: Sequence[FileTask], *, replay_quota: int, seed: int, official_system: str
) -> list[dict[str, Any]]:
    records = _safety_sft(tasks, official_system)
    records.extend(_discovery_sft(tasks, official_system))
    records.extend(
        _ordinary_sft_replay(
            tasks, quota=replay_quota, seed=seed, official_system=official_system
        )
    )
    random.Random(seed + 1).shuffle(records)
    return records


def _safety_dpo(tasks: Sequence[FileTask], official_system: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for task in tasks:
        if not task.metadata.get("safety_case", False):
            continue
        unsafe = next(
            pair for pair in _build_task_pairs(task) if pair["pair_kind"] == "safety_unsafe_read"
        )
        for variant, answer in enumerate(_refusal_answers(task)):
            chosen = canonical_action_to_tool_message({"type": "finish", "answer": answer})
            identity = f"{unsafe['pair_id']}:production-refusal:{variant}"
            pairs.append(
                _normalize_pair(
                    {
                        **unsafe,
                        "pair_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                        "chosen": chosen,
                        "pair_kind": "safety_finish_vs_unsafe_read",
                        "curriculum_group": "safety_first_action",
                    },
                    official_system,
                )
            )
    return pairs


def _discovery_dpo(
    tasks: Sequence[FileTask], *, quota: int, seed: int, official_system: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    mandatory: list[dict[str, Any]] = []
    for task in tasks:
        if task.operation != "read" or task.metadata.get("path_observability") != "discover":
            continue
        task_pairs = [
            pair
            for pair in _build_task_pairs(task)
            if pair["pair_kind"]
            in {
                "hidden_path_guess_before_discovery",
                "hidden_directory_jump_before_discovery",
                "premature_finish_before_discovery",
                "destructive_hidden_path_guess",
            }
        ]
        first_hidden_guess = next(
            pair
            for pair in task_pairs
            if pair["reference_step"] == 0
            and pair["pair_kind"] == "hidden_path_guess_before_discovery"
        )
        mandatory.append(first_hidden_guess)
        candidates.extend(pair for pair in task_pairs if pair["pair_id"] != first_hidden_guess["pair_id"])
    if quota < len(mandatory):
        raise RuntimeError(
            f"Discovery quota {quota} cannot cover all {len(mandatory)} first-action tasks"
        )
    rng = random.Random(seed)
    selected = list(mandatory)
    selected.extend(
        _round_robin_pair_kinds(candidates, quota=quota - len(selected), rng=rng)
    )
    if len(selected) != quota:
        raise RuntimeError(f"Generated {len(selected)} of {quota} discovery DPO pairs")
    for pair in selected:
        pair["curriculum_group"] = "discovery_focus"
    rng.shuffle(selected)
    return [_normalize_pair(pair, official_system) for pair in selected]


def _ordinary_dpo_replay(
    tasks: Sequence[FileTask], *, quota: int, seed: int, official_system: str
) -> list[dict[str, Any]]:
    operations = ("create", "update", "delete", "mixed")
    rng = random.Random(seed)
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if (
            task.operation in operations
            and not task.metadata.get("safety_case", False)
            and task.metadata.get("path_observability", "explicit") == "explicit"
        ):
            by_operation[task.operation].extend(_build_task_pairs(task))
    base, remainder = divmod(quota, len(operations))
    selected: list[dict[str, Any]] = []
    for op_index, operation in enumerate(operations):
        target = base + (op_index < remainder)
        sampled = _round_robin_pair_kinds(by_operation[operation], quota=target, rng=rng)
        for pair in sampled:
            pair["curriculum_group"] = "crud_replay"
        selected.extend(sampled)
    if len(selected) != quota:
        raise RuntimeError(f"Generated {len(selected)} of {quota} CRUD DPO replay pairs")
    rng.shuffle(selected)
    return [_normalize_pair(pair, official_system) for pair in selected]


def build_dpo(
    tasks: Sequence[FileTask],
    *,
    discovery_quota: int,
    replay_quota: int,
    seed: int,
    official_system: str,
) -> list[dict[str, Any]]:
    pairs = _safety_dpo(tasks, official_system)
    pairs.extend(
        _discovery_dpo(
            tasks,
            quota=discovery_quota,
            seed=seed,
            official_system=official_system,
        )
    )
    pairs.extend(
        _ordinary_dpo_replay(
            tasks,
            quota=replay_quota,
            seed=seed + 1,
            official_system=official_system,
        )
    )
    random.Random(seed + 2).shuffle(pairs)
    return pairs


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")
    return payload


def _summary(records: Sequence[dict[str, Any]], payload: str) -> dict[str, Any]:
    return {
        "count": len(records),
        "unique_tasks": len({record["task_id"] for record in records}),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "operations": dict(sorted(Counter(record["operation"] for record in records).items())),
        "curriculum_groups": dict(
            sorted(Counter(record["curriculum_group"] for record in records).items())
        ),
        "pair_kinds": dict(
            sorted(Counter(record.get("pair_kind", "sft") for record in records).items())
        ),
    }


def _write_corpus(
    output_dir: Path,
    *,
    train_name: str,
    eval_name: str,
    train: Sequence[dict[str, Any]],
    eval_records: Sequence[dict[str, Any]],
    ranking: bool,
    official_system: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = f"{train_name}.jsonl"
    eval_file = f"{eval_name}.jsonl"
    train_payload = _write_jsonl(output_dir / train_file, train)
    eval_payload = _write_jsonl(output_dir / eval_file, eval_records)
    columns = {"messages": "conversations", "system": "system"}
    if ranking:
        columns.update({"chosen": "chosen", "rejected": "rejected"})
    dataset_info = {
        train_name: {
            "file_name": train_file,
            "formatting": "sharegpt",
            **({"ranking": True} if ranking else {}),
            "columns": columns,
        },
        eval_name: {
            "file_name": eval_file,
            "formatting": "sharegpt",
            **({"ranking": True} if ranking else {}),
            "columns": columns,
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "version": "file-crud-v3-production-template-repair-v1",
        "method": (
            "Continue from legacy DPO100. Teach discovery and Safety first actions while "
            "replaying explicit-path CRUD. The official tokenizer tool preamble is embedded "
            "in system; LLaMA-Factory tool rendering is disabled."
        ),
        "official_system_sha256": hashlib.sha256(official_system.encode()).hexdigest(),
        "tools_embedded_in_system": True,
        "train": _summary(train, train_payload),
        "eval": _summary(eval_records, eval_payload),
        "task_overlap": sorted(
            {record["task_id"] for record in train}
            & {record["task_id"] for record in eval_records}
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sft-output-dir", required=True, type=Path)
    parser.add_argument("--dpo-output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=35_410)
    parser.add_argument("--sft-train-replay", type=int, default=800)
    parser.add_argument("--sft-eval-replay", type=int, default=200)
    parser.add_argument("--dpo-train-discovery", type=int, default=2400)
    parser.add_argument("--dpo-train-replay", type=int, default=1600)
    parser.add_argument("--dpo-eval-discovery", type=int, default=300)
    parser.add_argument("--dpo-eval-replay", type=int, default=400)
    args = parser.parse_args()

    official_system = render_official_system(args.model_path)
    train_tasks = load_file_tasks(args.train_dataset)
    validation_tasks = load_file_tasks(args.validation_dataset)
    sft_train = build_sft(
        train_tasks,
        replay_quota=args.sft_train_replay,
        seed=args.seed,
        official_system=official_system,
    )
    sft_eval = build_sft(
        validation_tasks,
        replay_quota=args.sft_eval_replay,
        seed=args.seed + 10,
        official_system=official_system,
    )
    dpo_train = build_dpo(
        train_tasks,
        discovery_quota=args.dpo_train_discovery,
        replay_quota=args.dpo_train_replay,
        seed=args.seed + 20,
        official_system=official_system,
    )
    dpo_eval = build_dpo(
        validation_tasks,
        discovery_quota=args.dpo_eval_discovery,
        replay_quota=args.dpo_eval_replay,
        seed=args.seed + 30,
        official_system=official_system,
    )
    sft_manifest = _write_corpus(
        args.sft_output_dir,
        train_name=SFT_TRAIN_NAME,
        eval_name=SFT_EVAL_NAME,
        train=sft_train,
        eval_records=sft_eval,
        ranking=False,
        official_system=official_system,
    )
    dpo_manifest = _write_corpus(
        args.dpo_output_dir,
        train_name=DPO_TRAIN_NAME,
        eval_name=DPO_EVAL_NAME,
        train=dpo_train,
        eval_records=dpo_eval,
        ranking=True,
        official_system=official_system,
    )
    print(json.dumps({"sft": sft_manifest, "dpo": dpo_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
