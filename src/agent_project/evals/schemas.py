from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


EvalCategory = Literal[
    "route",
    "planning",
    "local_file_suggestion",
    "rag_qa",
    "codex_routing",
]


class EvalExample(BaseModel):
    id: str
    category: EvalCategory
    input: str
    expected_route: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_grounding: list[str] = Field(default_factory=list)
    actual_route: str = ""
    actual_tools: list[str] = Field(default_factory=list)
    actual_answer: str = ""
    success: bool | None = None
    grounded: bool | None = None
    latency_seconds: float = 0.0
    tokens_or_prompt_chars: int = 0
    human_intervention_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    example_id: str
    category: EvalCategory
    route_correct: bool | None = None
    tool_call_correct: bool | None = None
    task_success: bool | None = None
    grounded: bool | None = None
    latency_seconds: float = 0.0
    tokens_or_prompt_chars: int = 0
    human_intervention_count: int = 0
    notes: str = ""
