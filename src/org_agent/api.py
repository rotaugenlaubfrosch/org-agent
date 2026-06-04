from __future__ import annotations

import asyncio

from org_agent.graph import run_lookup
from org_agent.models import LookupInput, OrganizationProfile
from org_agent.progress import ProgressCallback, report
from org_agent.settings import Settings, load_app_config, validate_settings
from org_agent.website import normalize_url


async def lookup_organization_async(
    name: str,
    website: str | None = None,
    config: str | None = None,
    registries: list[str] | None = None,
    progress: ProgressCallback | None = None,
) -> OrganizationProfile:
    if not website or not website.strip():
        raise ValueError("A website is required. Provide --website or website=...")
    normalized_website = normalize_url(website.strip())
    settings = Settings()
    validate_settings(settings, selected_registries=registries)
    app_config = load_app_config(config, selected_registries=registries)
    report(progress, "config", f"LLM provider: {settings.llm_provider} ({settings.llm_model})")
    report(progress, "config", f"Enabled registries: {sum(1 for r in app_config.registries if r.enabled)}")
    lookup_input = LookupInput(name=name, website=normalized_website)
    return await run_lookup(lookup_input, settings, app_config, progress=progress)


def lookup_organization(
    name: str,
    website: str | None = None,
    config: str | None = None,
    registries: list[str] | None = None,
    progress: ProgressCallback | None = None,
) -> OrganizationProfile:
    return asyncio.run(
        lookup_organization_async(
            name=name,
            website=website,
            config=config,
            registries=registries,
            progress=progress,
        )
    )
