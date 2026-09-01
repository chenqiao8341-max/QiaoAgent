from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

FileOperation = Literal["create", "read", "update", "delete", "mixed"]
FileActionType = Literal["list_dir", "read_file", "write_file", "delete_path", "finish"]


def validate_relative_path(path: str, *, allow_root: bool = False) -> str:
    """Validate and normalize a portable, workspace-relative task path."""
    clean = path.strip()
    if allow_root and clean in {"", "."}:
        return "."
    if not clean or "\x00" in clean or "\\" in clean:
        raise ValueError(f"Unsafe relative path: {path!r}")
    candidate = PurePosixPath(clean)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Unsafe relative path: {path!r}")
    return candidate.as_posix()


class FileAction(BaseModel):
    """One model-generated action in the local-file environment."""

    model_config = ConfigDict(extra="forbid")

    type: FileActionType
    path: str = ""
    content: str = ""
    mode: Literal["overwrite", "append"] = "overwrite"
    answer: str = ""


class FileTask(BaseModel):
    """A reproducible local-file task and its hidden evaluator labels."""

    model_config = ConfigDict(extra="forbid")

    id: str
    split: Literal["train", "validation", "test"] = "train"
    operation: FileOperation
    instruction: str
    initial_files: dict[str, str] = Field(default_factory=dict)
    # None means that the path must not exist at the end of the episode.
    expected_files: dict[str, str | None] = Field(default_factory=dict)
    expected_answer_contains: list[str] = Field(default_factory=list)
    required_actions: list[FileActionType] = Field(default_factory=list)
    reference_actions: list[FileAction] = Field(default_factory=list)
    max_steps: int = Field(default=8, ge=1, le=50)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        self.initial_files = {
            validate_relative_path(path): content for path, content in self.initial_files.items()
        }
        self.expected_files = {
            validate_relative_path(path): content for path, content in self.expected_files.items()
        }
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if self.reference_actions and self.reference_actions[-1].type != "finish":
            raise ValueError("reference_actions must end with finish when provided")
        return self


class FileObservation(BaseModel):
    """The policy-visible state. Hidden labels are deliberately excluded."""

    task_id: str
    instruction: str
    step: int
    max_steps: int
    last_result: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    available_actions: list[FileActionType] = Field(
        default_factory=lambda: [
            "list_dir",
            "read_file",
            "write_file",
            "delete_path",
            "finish",
        ]
    )


class FileStepResult(BaseModel):
    observation: FileObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    action: dict[str, Any]
    raw_output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileTransition(BaseModel):
    step: int
    observation: FileObservation
    action: dict[str, Any]
    raw_model_output: str = ""
    next_observation: FileObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class FileTrajectory(BaseModel):
    task_id: str
    split: str
    operation: FileOperation
    policy: str
    transitions: list[FileTransition]
    total_reward: float
    success: bool
    operation_success: bool = False
    response_success: bool = False
    strict_success: bool = False
    final_answer: str = ""
    final_files: dict[str, str] = Field(default_factory=dict)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
