from __future__ import annotations

import asyncio

from org_agent.graph import run_lookup
from org_agent.models import LookupInput, OrganizationProfile
from org_agent.progress import ProgressCallback, report
from org_agent.settings import Settings, load_app_config, validate_settings


async def lookup_organization_async(
    name: str,
    website: str | None = None,
    config: str | None = None,
    progress: ProgressCallback | None = None,
) -> OrganizationProfile:
    settings = Settings()
    validate_settings(settings)
    app_config = load_app_config(config)
    has_registry = any(registry.enabled for registry in app_config.registries)
    has_search = (settings.search_provider or "").lower().strip() not in {"", "none", "disabled"}
    report(progress, "config", f"LLM provider: {settings.llm_provider} ({settings.llm_model})")
    report(progress, "config", f"Search provider: {settings.search_provider}")
    report(progress, "config", f"Enabled registries: {sum(1 for r in app_config.registries if r.enabled)}")
    if website is None and not has_search and not has_registry:
        raise ValueError(
            "Name-only lookup requires ORG_AGENT_SEARCH_PROVIDER or enabled registry config. "
            "Provide --website to inspect a known official website."
        )
    lookup_input = LookupInput(name=name, website=website)
    return await run_lookup(lookup_input, settings, app_config, progress=progress)


def lookup_organization(
    name: str,
    website: str | None = None,
    config: str | None = None,
    progress: ProgressCallback | None = None,
) -> OrganizationProfile:
    return asyncio.run(
        lookup_organization_async(name=name, website=website, config=config, progress=progress)
    )
