import pytest

from org_agent import api


def test_lookup_requires_website(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("ORG_AGENT_REQUEST_TIMEOUT", "20")

    with pytest.raises(ValueError, match="A website is required"):
        api.lookup_organization("Example Ltd")


def test_lookup_normalizes_bare_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORG_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ORG_AGENT_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("ORG_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("ORG_AGENT_REQUEST_TIMEOUT", "20")

    captured = {}

    async def fake_run_lookup(lookup_input, settings, app_config, progress=None):
        captured["website"] = str(lookup_input.website)
        return api.OrganizationProfile(queried_name=lookup_input.name, website=str(lookup_input.website))

    monkeypatch.setattr(api, "run_lookup", fake_run_lookup)

    api.lookup_organization("Example Ltd", website="example.com")

    assert captured["website"] == "https://example.com/"
