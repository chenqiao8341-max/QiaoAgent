#!/usr/bin/env python3
"""Verify the isolated QiaoAgent GRPO runtime without downloading a model."""

from __future__ import annotations

import json

import accelerate
import datasets
import peft
import torch
import transformers
import vllm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config


MODEL_PATH = "/home/qingao/models/Qwen3.5-4B"


def main() -> None:
    versions: dict[str, object] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {versions}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish a task.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }
    ]
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "finish now"},
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "# Tools" not in rendered or "<|im_start|>assistant" not in rendered:
        raise RuntimeError("Qwen3.5 tool template smoke test failed")

    config = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_embd=32,
        n_layer=1,
        n_head=4,
    )
    model = AutoModelForCausalLM.from_config(config)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        ),
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids=input_ids, labels=input_ids)
    output.loss.backward()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0 or not torch.isfinite(output.loss):
        raise RuntimeError("PEFT LoRA forward/backward smoke test failed")

    versions.update(
        {
            "qwen_template_chars": len(rendered),
            "lora_trainable_parameters": trainable,
            "smoke_loss": float(output.loss.detach()),
        }
    )
    print(json.dumps(versions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
