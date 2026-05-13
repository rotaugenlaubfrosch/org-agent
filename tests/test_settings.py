from pathlib import Path

import pytest

from org_agent.settings import load_app_config, validate_settings
from org_agent.settings import Settings
from org_agent.website import _score_link


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


def test_settings_accepts_project_scoped_ollama_url(monkeypatch) -> None:
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://ollama.local:11434")
    monkeypatch.delenv("ORG_AGENT_REQUEST_TIMEOUT", raising=False)

    settings = Settings()

    assert settings.ollama_base_url == "http://ollama.local:11434"
    assert settings.request_timeout == 20.0
    assert settings.crawl_max_pages == 6
    assert settings.crawl_max_depth == 2
    validate_settings(settings)


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


def test_link_scoring_prefers_contact_pages() -> None:
    contact_score = _score_link("https://example.com/kontakt", "Kontakt", "navigation")
    account_score = _score_link("https://example.com/account", "Account", "navigation")

    assert contact_score > 0
    assert account_score == 0
