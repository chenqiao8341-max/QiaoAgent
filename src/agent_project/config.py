from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model_provider: str = "openai"
    model_temperature: float = 0.2
    openai_model: str = "gpt-4o-mini"
    google_model: str = "gemini-1.5-flash"
    anthropic_model: str = "claude-3-5-haiku-latest"
    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    openai_compatible_model: str = "qwen-plus"
    agent_workspace_root: str | None = None
    agent_enable_human_approval: bool = True
    agent_enable_shell_commands: bool = True
    agent_enable_web_search: bool = True
    agent_enable_browser_tools: bool = True


def load_settings() -> Settings:
    load_dotenv()

    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "openai").strip().lower(),
        model_temperature=float(os.getenv("MODEL_TEMPERATURE", "0.2")),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        google_model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
        openai_compatible_model=os.getenv("OPENAI_COMPATIBLE_MODEL", "qwen-plus"),
        agent_workspace_root=os.getenv("AGENT_WORKSPACE_ROOT"),
        agent_enable_human_approval=os.getenv("AGENT_ENABLE_HUMAN_APPROVAL", "true")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        agent_enable_shell_commands=os.getenv("AGENT_ENABLE_SHELL_COMMANDS", "true")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        agent_enable_web_search=os.getenv("AGENT_ENABLE_WEB_SEARCH", "true")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        agent_enable_browser_tools=os.getenv("AGENT_ENABLE_BROWSER_TOOLS", "true")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
    )
