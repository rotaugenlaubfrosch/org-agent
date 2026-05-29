from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from org_agent.models import AppConfig, RegistryEndpointConfig


DEFAULT_DESCRIPTION_SYSTEM_PROMPT = (
    "Formuliere eine neutrale, sachliche Kurzbeschreibung auf Deutsch für eine "
    "Unternehmensdatenbank. Die Beschreibung muss zwingend in der Form beginnen: "
    "'[Account Name] beschäftigt sich mit …'. Verzichte auf werbliche Sprache und "
    "Superlative. Nenne sachlich, womit sich das Unternehmen oder die Organisation "
    "befasst. Die Ausgabe muss zwingend in deutscher Sprache sein. Gebe ausschliesslich "
    "die Beschreibung aus, ohne Kommentare oder Zusätze ."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str | None = Field(default=None, alias="ORG_AGENT_LLM_PROVIDER")
    llm_model: str | None = Field(default=None, alias="ORG_AGENT_LLM_MODEL")
    api_key: str | None = Field(default=None, alias="ORG_AGENT_API_KEY")
    search_provider: str | None = Field(default=None, alias="ORG_AGENT_SEARCH_PROVIDER")
    search_api_key: str | None = Field(default=None, alias="ORG_AGENT_SEARCH_API_KEY")
    zefix_username: str | None = Field(default=None, alias="ORG_AGENT_ZEFIX_USERNAME")
    zefix_password: str | None = Field(default=None, alias="ORG_AGENT_ZEFIX_PASSWORD")
    ollama_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ORG_AGENT_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
    )
    request_timeout: float = Field(default=20.0, alias="ORG_AGENT_REQUEST_TIMEOUT")
    crawl_max_pages: int = Field(default=6, alias="ORG_AGENT_CRAWL_MAX_PAGES")
    crawl_max_depth: int = Field(default=2, alias="ORG_AGENT_CRAWL_MAX_DEPTH")
    crawl_log_enabled: bool = Field(default=True, alias="ORG_AGENT_CRAWL_LOG_ENABLED")
    crawl_log_dir: str | None = Field(default=None, alias="ORG_AGENT_CRAWL_LOG_DIR")
    playwright_headless: bool = Field(default=True, alias="ORG_AGENT_PLAYWRIGHT_HEADLESS")
    playwright_slow_mo: int = Field(default=0, alias="ORG_AGENT_PLAYWRIGHT_SLOW_MO")
    description_system_prompt: str = Field(
        default=DEFAULT_DESCRIPTION_SYSTEM_PROMPT,
        alias="ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT",
    )

    def __init__(self, **data: object) -> None:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        super().__init__(**data)

    @field_validator(
        "request_timeout",
        "crawl_max_pages",
        "crawl_max_depth",
        "crawl_log_enabled",
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
            "crawl_log_enabled": True,
            "playwright_headless": True,
            "playwright_slow_mo": 0,
        }
        return defaults[info.field_name]


def validate_settings(settings: Settings, selected_registries: list[str] | None = None) -> None:
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

    selected = {(name or "").lower().strip() for name in (selected_registries or [])}
    if "zefix" in selected:
        if _blank(settings.zefix_username):
            missing.append("ORG_AGENT_ZEFIX_USERNAME")
        if _blank(settings.zefix_password):
            missing.append("ORG_AGENT_ZEFIX_PASSWORD")

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


def load_app_config(path: str | Path | None, selected_registries: list[str] | None = None) -> AppConfig:
    if path is None:
        config = AppConfig()
    else:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
        config = AppConfig.model_validate(data)

    return _apply_selected_registries(config, selected_registries)


def _apply_selected_registries(
    config: AppConfig,
    selected_registries: list[str] | None,
) -> AppConfig:
    if not selected_registries:
        return config

    selected = {(name or "").lower().strip() for name in selected_registries}
    selected.discard("")
    unsupported = selected - {"zefix"}
    if unsupported:
        bad = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported --registry value(s): {bad}. Supported values: zefix.")

    existing_names = {registry.name.lower().strip() for registry in config.registries}
    for registry in config.registries:
        if registry.name.lower().strip() in selected or (registry.provider or "").lower().strip() in selected:
            registry.enabled = True

    if "zefix" in selected and "zefix" not in existing_names:
        config.registries.append(
            RegistryEndpointConfig(
                name="zefix",
                provider="zefix",
                base_url="https://www.zefix.admin.ch/ZefixPublicREST",
                enabled=True,
            )
        )

    return AppConfig.model_validate(config.model_dump())
