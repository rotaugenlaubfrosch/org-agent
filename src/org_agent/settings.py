from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from org_agent.models import AppConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str | None = Field(default=None, alias="ORG_AGENT_LLM_PROVIDER")
    llm_model: str | None = Field(default=None, alias="ORG_AGENT_LLM_MODEL")
    api_key: str | None = Field(default=None, alias="ORG_AGENT_API_KEY")
    search_provider: str | None = Field(default=None, alias="ORG_AGENT_SEARCH_PROVIDER")
    search_api_key: str | None = Field(default=None, alias="ORG_AGENT_SEARCH_API_KEY")
    ollama_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ORG_AGENT_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
    )
    request_timeout: float = Field(default=20.0, alias="ORG_AGENT_REQUEST_TIMEOUT")
    crawl_max_pages: int = Field(default=6, alias="ORG_AGENT_CRAWL_MAX_PAGES")
    crawl_max_depth: int = Field(default=2, alias="ORG_AGENT_CRAWL_MAX_DEPTH")
    playwright_headless: bool = Field(default=True, alias="ORG_AGENT_PLAYWRIGHT_HEADLESS")
    playwright_slow_mo: int = Field(default=0, alias="ORG_AGENT_PLAYWRIGHT_SLOW_MO")

    def __init__(self, **data: object) -> None:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        super().__init__(**data)

    @field_validator(
        "request_timeout",
        "crawl_max_pages",
        "crawl_max_depth",
        "playwright_headless",
        "playwright_slow_mo",
        mode="before",
    )
    @classmethod
    def blank_optional_runtime_to_default(cls, value: object, info: ValidationInfo) -> object:
        if value != "":
            return value
        defaults = {
            "request_timeout": 20.0,
            "crawl_max_pages": 6,
            "crawl_max_depth": 2,
            "playwright_headless": True,
            "playwright_slow_mo": 0,
        }
        return defaults[info.field_name]


def validate_settings(settings: Settings) -> None:
    missing: list[str] = []
    if _blank(settings.llm_provider):
        missing.append("ORG_AGENT_LLM_PROVIDER")
    if _blank(settings.llm_model):
        missing.append("ORG_AGENT_LLM_MODEL")
    provider = (settings.llm_provider or "").lower().strip()
    if provider in {"openai", "anthropic"} and _blank(settings.api_key):
        missing.append("ORG_AGENT_API_KEY")
    if provider == "ollama" and _blank(settings.ollama_base_url):
        missing.append("ORG_AGENT_OLLAMA_BASE_URL")

    search_provider = (settings.search_provider or "").lower().strip()
    if search_provider in {"tavily", "brave"} and _blank(settings.search_api_key):
        missing.append("ORG_AGENT_SEARCH_API_KEY")

    if missing:
        formatted = ", ".join(dict.fromkeys(missing))
        raise ValueError(f"Missing required environment configuration: {formatted}")

    supported_llm_providers = {"openai", "anthropic", "ollama"}
    if provider not in supported_llm_providers:
        raise ValueError(
            "Unsupported ORG_AGENT_LLM_PROVIDER. Supported providers: openai, anthropic, ollama."
        )

    supported_search_providers = {"", "none", "disabled", "tavily", "brave"}
    if search_provider not in supported_search_providers:
        raise ValueError(
            "Unsupported ORG_AGENT_SEARCH_PROVIDER. Supported providers: tavily, brave, none."
        )


def _blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def load_app_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    return AppConfig.model_validate(data)
