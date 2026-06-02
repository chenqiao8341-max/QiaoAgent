from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agent_project.config import Settings


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

    if provider in {"openai-compatible", "dashscope", "bailian", "aliyun"}:
        base_url = settings.openai_compatible_base_url
        if provider in {"dashscope", "bailian", "aliyun"} and not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not base_url:
            raise ValueError("OPENAI_COMPATIBLE_BASE_URL is required for openai-compatible provider.")

        return ChatOpenAI(
            model=settings.openai_compatible_model,
            temperature=settings.model_temperature,
            api_key=settings.openai_compatible_api_key,
            base_url=base_url,
        )

    supported = "openai, google, anthropic, openai-compatible, dashscope, bailian, aliyun"
    raise ValueError(f"Unsupported MODEL_PROVIDER={provider!r}. Supported providers: {supported}")
