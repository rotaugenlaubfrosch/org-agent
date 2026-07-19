from __future__ import annotations

import asyncio

from org_agent.graph import run_lookup
from org_agent.models import LookupInput, LookupResult, OrganizationProfile  # noqa: F401
from org_agent.progress import ProgressCallback, report
from org_agent.registry import normalize_country_code
from org_agent.settings import Settings, validate_settings
from org_agent.website import normalize_url


async def lookup_organization_async(
    name: str,
    website: str | None = None,
    country: str | None = None,
    browser_use: bool = False,
    progress: ProgressCallback | None = None,
) -> LookupResult:
    if not website or not website.strip():
        raise ValueError("A website is required. Provide --website or website=...")
    normalized_website = normalize_url(website.strip())
    settings = Settings()
    validate_settings(settings, country=country)
    normalized_country = normalize_country_code(country)
    report(progress, "config", f"LLM provider: {settings.llm_provider} ({settings.llm_model})")
    report(progress, "config", f"Country registry: {normalized_country or 'none'}")
    lookup_input = LookupInput(name=name, website=normalized_website)
    if browser_use:
        from org_agent.browser_use_lookup import run_browser_use_lookup

        report(progress, "config", "Browser automation: browser-use local mode")
        return await run_browser_use_lookup(
            lookup_input,
            settings,
            country=normalized_country,
            progress=progress,
        )
    return await run_lookup(lookup_input, settings, country=normalized_country, progress=progress)


def lookup_organization(
    name: str,
    website: str | None = None,
    country: str | None = None,
    browser_use: bool = False,
    progress: ProgressCallback | None = None,
) -> LookupResult:
    return asyncio.run(
        lookup_organization_async(
            name=name,
            website=website,
            country=country,
            browser_use=browser_use,
            progress=progress,
        )
    )
