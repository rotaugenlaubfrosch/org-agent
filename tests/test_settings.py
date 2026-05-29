from pathlib import Path

import pytest

from org_agent.settings import (
    DEFAULT_DESCRIPTION_SYSTEM_PROMPT,
    DEFAULT_INDUSTRIES_CSV,
    Settings,
    load_app_config,
    validate_settings,
)


def test_load_app_config(tmp_path: Path) -> None:
    config_path = tmp_path / "org-agent.yaml"
    config_path.write_text(
        """
registries:
  - name: demo
    base_url: https://example.com/search
    query_param: q
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert len(config.registries) == 1
    assert config.registries[0].name == "demo"


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

    settings = Settings()

    assert settings.ollama_base_url == "http://ollama.local:11434"
    assert settings.request_timeout == 20.0
    assert settings.crawl_max_pages == 6
    assert settings.crawl_max_depth == 2
    assert settings.crawl_log_enabled is True
    assert settings.crawl_log_dir is None
    assert settings.playwright_headless is True
    assert settings.playwright_slow_mo == 0
    assert settings.description_system_prompt == DEFAULT_DESCRIPTION_SYSTEM_PROMPT
    assert settings.industries_csv == DEFAULT_INDUSTRIES_CSV
    assert settings.max_industries == 1
    assert settings.industry_shortlist_size == 25
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


def test_validate_settings_requires_zefix_credentials_when_selected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("ORG_AGENT_ZEFIX_USERNAME", raising=False)
    monkeypatch.delenv("ORG_AGENT_ZEFIX_PASSWORD", raising=False)

    settings = Settings()

    with pytest.raises(ValueError, match="ORG_AGENT_ZEFIX_USERNAME"):
        validate_settings(settings, selected_registries=["zefix"])


def test_load_app_config_adds_selected_zefix_registry() -> None:
    config = load_app_config(None, selected_registries=["zefix"])

    assert len(config.registries) == 1
    assert config.registries[0].name == "zefix"
    assert config.registries[0].provider == "zefix"
