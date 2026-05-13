import pytest

from org_agent.api import lookup_organization


def test_name_only_lookup_requires_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("ORG_AGENT_REQUEST_TIMEOUT", "20")
    monkeypatch.setenv("ORG_AGENT_SEARCH_PROVIDER", "none")

    with pytest.raises(ValueError, match="Name-only lookup requires"):
        lookup_organization("Example Ltd")
