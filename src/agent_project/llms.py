from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agent_project.config import Settings


def _add_no_proxy_host(host: str) -> None:
    for name in ("NO_PROXY", "no_proxy"):
        values = [
            item.strip() for item in os.getenv(name, "").split(",") if item.strip()
        ]
        if host not in values:
            values.append(host)
        os.environ[name] = ",".join(values)


def _prepare_local_vllm_environment(base_url: str) -> None:
    if "127.0.0.1" in base_url:
        _add_no_proxy_host("127.0.0.1")
    if "localhost" in base_url:
        _add_no_proxy_host("localhost")


def build_chat_model(settings: Settings):
    """Create a chat model from environment-driven provider settings."""
    provider = settings.model_provider

    if provider == "openai":
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.model_temperature,
        )

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            temperature=settings.model_temperature,
        )

    if provider == "anthropic":
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=settings.model_temperature,
        )

    if provider in {"local-vllm", "vllm", "local"}:
        _prepare_local_vllm_environment(settings.local_vllm_base_url)
        return ChatOpenAI(
            model=settings.local_vllm_model,
            temperature=settings.model_temperature,
            api_key=settings.local_vllm_api_key,
            base_url=settings.local_vllm_base_url,
        )

    if provider in {"openai-compatible", "dashscope", "bailian", "aliyun"}:
        base_url = settings.openai_compatible_base_url
        if provider in {"dashscope", "bailian", "aliyun"} and not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not base_url:
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL is required for openai-compatible provider."
            )

        return ChatOpenAI(
            model=settings.openai_compatible_model,
            temperature=settings.model_temperature,
            api_key=settings.openai_compatible_api_key,
            base_url=base_url,
        )

    supported = (
        "openai, google, anthropic, openai-compatible, local-vllm, "
        "vllm, local, dashscope, bailian, aliyun"
    )
    raise ValueError(
        f"Unsupported MODEL_PROVIDER={provider!r}. Supported providers: {supported}"
    )
