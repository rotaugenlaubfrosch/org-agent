from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from org_agent.registry import validate_country_code


DEFAULT_DESCRIPTION_SYSTEM_PROMPT = (
    "Formuliere eine neutrale, sachliche Kurzbeschreibung auf Deutsch für eine "
    "Unternehmensdatenbank. Die Beschreibung muss zwingend in der Form beginnen: "
    "'[Account Name] beschäftigt sich mit …'. Verzichte auf werbliche Sprache und "
    "Superlative. Nenne sachlich, womit sich das Unternehmen oder die Organisation "
    "befasst. Die Ausgabe muss zwingend in deutscher Sprache sein. Gebe ausschliesslich "
    "die Beschreibung aus, ohne Kommentare oder Zusätze ."
)

DEFAULT_INDUSTRIES_CSV = str(Path(__file__).resolve().parent / "data" / "industries.csv")
DEFAULT_SECTORS_CSV = str(Path(__file__).resolve().parent / "data" / "sectors.csv")
DEFAULT_COMPANY_TYPES_CSV = str(Path(__file__).resolve().parent / "data" / "company_types.csv")
DEFAULT_LEGAL_STRUCTURES_CSV = str(Path(__file__).resolve().parent / "data" / "legal_structures.csv")
DEFAULT_CRAWL_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs")


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
    crawl_log_enabled: bool = Field(default=True, alias="ORG_AGENT_CRAWL_LOG_ENABLED")
    crawl_log_dir: str | None = Field(default=DEFAULT_CRAWL_LOG_DIR, alias="ORG_AGENT_CRAWL_LOG_DIR")
    playwright_headless: bool = Field(default=True, alias="ORG_AGENT_PLAYWRIGHT_HEADLESS")
    playwright_slow_mo: int = Field(default=0, alias="ORG_AGENT_PLAYWRIGHT_SLOW_MO")
    description_system_prompt: str = Field(
        default=DEFAULT_DESCRIPTION_SYSTEM_PROMPT,
        alias="ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT",
    )
    industries_csv: str = Field(default=DEFAULT_INDUSTRIES_CSV, alias="ORG_AGENT_INDUSTRIES_CSV")
    max_industries: int = Field(default=1, alias="ORG_AGENT_MAX_INDUSTRIES")
    industry_shortlist_size: int = Field(default=25, alias="ORG_AGENT_INDUSTRY_SHORTLIST_SIZE")
    sectors_csv: str = Field(default=DEFAULT_SECTORS_CSV, alias="ORG_AGENT_SECTORS_CSV")
    company_types_csv: str = Field(
        default=DEFAULT_COMPANY_TYPES_CSV,
        alias="ORG_AGENT_COMPANY_TYPES_CSV",
    )
    legal_structures_csv: str = Field(
        default=DEFAULT_LEGAL_STRUCTURES_CSV,
        alias="ORG_AGENT_LEGAL_STRUCTURES_CSV",
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
        "max_industries",
        "industry_shortlist_size",
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
            "max_industries": 1,
            "industry_shortlist_size": 25,
        }
        return defaults[info.field_name]

    @field_validator("crawl_log_dir", mode="before")
    @classmethod
    def blank_crawl_log_dir_to_default(cls, value: object) -> object:
        if value == "":
            return DEFAULT_CRAWL_LOG_DIR
        return value


def validate_settings(settings: Settings, country: str | None = None) -> None:
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

    if missing:
        formatted = ", ".join(dict.fromkeys(missing))
        raise ValueError(f"Missing required environment configuration: {formatted}")

    supported_llm_providers = {"openai", "anthropic", "ollama"}
    if provider not in supported_llm_providers:
        raise ValueError(
            "Unsupported ORG_AGENT_LLM_PROVIDER. Supported providers: openai, anthropic, ollama."
        )

    validate_country_code(country)


def _blank(value: str | None) -> bool:
    return value is None or value.strip() == ""
