#!/usr/bin/env python3
"""Run LLaMA-Factory with a production-aligned Qwen3.5 no-think template.

LLaMA-Factory's built-in Qwen3.5 reasoning template adds an empty thinking
prefix only to the current response during pairwise preprocessing.  The model's
official tokenizer adds it to every assistant message in a multi-step tool
history.  This local entry point registers a template that reproduces the
official behavior without patching the external LLaMA-Factory checkout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llamafactory.data.formatter import StringFormatter, ToolFormatter
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.template import TEMPLATES, register_template

TEMPLATE_NAME = "qwen3_5_official_nothink"
EMPTY_THINK = "<think>\n\n</think>\n\n"


@dataclass
class OfficialQwen35FunctionFormatter(StringFormatter):
    """Match the tokenizer's Jinja ``value | string`` argument rendering.

    LLaMA-Factory's stock Qwen3.5 formatter serializes every non-string value
    with JSON, producing ``true``/``false``.  The official Jinja template uses
    Python string rendering for scalar values, producing ``True``/``False`` in
    production chat histories.  Containers still use JSON via ``tojson``.
    """

    def apply(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        content = kwargs.pop("content")
        calls = json.loads(content)
        if not isinstance(calls, list):
            calls = [calls]
        rendered_calls: list[str] = []
        for call in calls:
            rendered = f"<tool_call>\n<function={call['name']}>"
            for name, value in call["arguments"].items():
                if isinstance(value, (dict, list)):
                    value_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                else:
                    value_text = str(value)
                rendered += f"\n<parameter={name}>\n{value_text}\n</parameter>"
            rendered += "\n</function>\n</tool_call>"
            rendered_calls.append(rendered)
        return super().apply(content="\n".join(rendered_calls))


def register_qwen35_official_nothink() -> None:
    if TEMPLATE_NAME in TEMPLATES:
        return
    register_template(
        name=TEMPLATE_NAME,
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
        ),
        format_assistant=StringFormatter(slots=[f"{EMPTY_THINK}{{{{content}}}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_function=OfficialQwen35FunctionFormatter(
            slots=[f"{EMPTY_THINK}{{{{content}}}}<|im_end|>\n"],
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response>"
                "<|im_end|>\n<|im_start|>assistant\n"
            ]
        ),
        format_tools=ToolFormatter(tool_format="qwen3_5"),
        stop_words=["<|im_end|>"],
        replace_eos=True,
        mm_plugin=get_mm_plugin(
            name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"
        ),
    )


def main() -> None:
    register_qwen35_official_nothink()
    from llamafactory.cli import main as llamafactory_main

    llamafactory_main()


if __name__ == "__main__":
    main()
