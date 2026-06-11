from pathlib import Path

import pytest

from org_agent.settings import (
    DEFAULT_COMPANY_TYPES_CSV,
    DEFAULT_CRAWL_LOG_DIR,
    DEFAULT_DESCRIPTION_SYSTEM_PROMPT,
    DEFAULT_INDUSTRIES_CSV,
    DEFAULT_LEGAL_STRUCTURES_CSV,
    DEFAULT_SECTORS_CSV,
    Settings,
    validate_settings,
)


def test_settings_accepts_project_scoped_ollama_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://ollama.local:11434")
    monkeypatch.delenv("ORG_AGENT_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("ORG_AGENT_CRAWL_LOG_ENABLED", raising=False)
    monkeypatch.delenv("ORG_AGENT_CRAWL_LOG_DIR", raising=False)
    monkeypatch.delenv("ORG_AGENT_PLAYWRIGHT_HEADLESS", raising=False)
    monkeypatch.delenv("ORG_AGENT_PLAYWRIGHT_SLOW_MO", raising=False)
    monkeypatch.delenv("ORG_AGENT_INDUSTRIES_CSV", raising=False)
    monkeypatch.delenv("ORG_AGENT_MAX_INDUSTRIES", raising=False)
    monkeypatch.delenv("ORG_AGENT_INDUSTRY_SHORTLIST_SIZE", raising=False)
    monkeypatch.delenv("ORG_AGENT_SECTORS_CSV", raising=False)
    monkeypatch.delenv("ORG_AGENT_COMPANY_TYPES_CSV", raising=False)
    monkeypatch.delenv("ORG_AGENT_LEGAL_STRUCTURES_CSV", raising=False)

    settings = Settings()

    assert settings.ollama_base_url == "http://ollama.local:11434"
    assert settings.request_timeout == 20.0
    assert settings.crawl_max_pages == 6
    assert settings.crawl_max_depth == 2
    assert settings.crawl_log_enabled is True
    assert settings.crawl_log_dir == DEFAULT_CRAWL_LOG_DIR
    assert settings.playwright_headless is True
    assert settings.playwright_slow_mo == 0
    assert settings.description_system_prompt == DEFAULT_DESCRIPTION_SYSTEM_PROMPT
    assert settings.industries_csv == DEFAULT_INDUSTRIES_CSV
    assert settings.max_industries == 1
    assert settings.industry_shortlist_size == 25
    assert settings.sectors_csv == DEFAULT_SECTORS_CSV
    assert settings.company_types_csv == DEFAULT_COMPANY_TYPES_CSV
    assert settings.legal_structures_csv == DEFAULT_LEGAL_STRUCTURES_CSV
    validate_settings(settings)


def test_settings_accepts_description_system_prompt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT", "Describe [Account Name].")

    settings = Settings()

    assert settings.description_system_prompt == "Describe [Account Name]."


def test_settings_accepts_industry_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_INDUSTRIES_CSV", "industries.csv")
    monkeypatch.setenv("ORG_AGENT_MAX_INDUSTRIES", "3")
    monkeypatch.setenv("ORG_AGENT_INDUSTRY_SHORTLIST_SIZE", "10")

    settings = Settings()

    assert settings.industries_csv == "industries.csv"
    assert settings.max_industries == 3
    assert settings.industry_shortlist_size == 10


def test_settings_accepts_sector_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_SECTORS_CSV", "sectors.csv")

    settings = Settings()

    assert settings.sectors_csv == "sectors.csv"


def test_settings_accepts_company_type_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_COMPANY_TYPES_CSV", "company_types.csv")

    settings = Settings()

    assert settings.company_types_csv == "company_types.csv"


def test_settings_accepts_legal_structure_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_LEGAL_STRUCTURES_CSV", "legal_structures.csv")

    settings = Settings()

    assert settings.legal_structures_csv == "legal_structures.csv"


def test_settings_accepts_headed_playwright(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_PLAYWRIGHT_HEADLESS", "false")
    monkeypatch.setenv("ORG_AGENT_PLAYWRIGHT_SLOW_MO", "300")

    settings = Settings()

    assert settings.playwright_headless is False
    assert settings.playwright_slow_mo == 300


def test_settings_accepts_crawl_log_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_CRAWL_LOG_DIR", "logs")

    settings = Settings()

    assert settings.crawl_log_dir == "logs"


def test_settings_uses_default_crawl_log_dir_for_blank_value(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_CRAWL_LOG_DIR", "")

    settings = Settings()

    assert settings.crawl_log_dir == DEFAULT_CRAWL_LOG_DIR


def test_settings_accepts_disabled_crawl_logging(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_CRAWL_LOG_ENABLED", "false")

    settings = Settings()

    assert settings.crawl_log_enabled is False


def test_settings_rejects_blank_required_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "")

    settings = Settings()

    with pytest.raises(ValueError, match="ORG_AGENT_LLM_PROVIDER"):
        validate_settings(settings)


def test_settings_requires_ollama_model_and_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("ORG_AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("ORG_AGENT_OLLAMA_BASE_URL", raising=False)

    settings = Settings()

    with pytest.raises(ValueError, match="ORG_AGENT_LLM_MODEL"):
        validate_settings(settings)


def test_validate_settings_accepts_country_without_registry_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("ORG_AGENT_REGISTRY_CH_USERNAME", raising=False)
    monkeypatch.delenv("ORG_AGENT_REGISTRY_CH_PASSWORD", raising=False)

    settings = Settings()

    validate_settings(settings, country="ch")


def test_validate_settings_rejects_unsupported_country(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")

    settings = Settings()

    with pytest.raises(ValueError, match="Unsupported --country value: xx"):
        validate_settings(settings, country="xx")
