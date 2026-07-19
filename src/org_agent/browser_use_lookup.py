from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from org_agent.graph import (
    _build_registry_profile,
    _derive_profile_country_from_address,
    _fragment_profile_address,
    _load_industries,
    _normalize_profile_address_country,
    _normalize_profile_country,
    _queried_country_value,
    _registry_result_message,
    _validate_profile_email,
)
from org_agent.llm import build_chat_model
from org_agent.models import LookupInput, LookupResult, OrganizationProfile, WebsitePage
from org_agent.progress import ProgressCallback, report
from org_agent.registry import query_country_registry
from org_agent.settings import Settings

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class BrowserUseContactExtraction(BaseModel):
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    address_country: str | None = None
    source_text: str | None = None


class BrowserUseFactsExtraction(BaseModel):
    description: str | None = None
    employees: int | None = None
    legal_structure: str | None = None
    source_text: str | None = None


class BrowserUseClassificationExtraction(BaseModel):
    sector: str | None = None
    company_type: str | None = None
    industry: str | None = None
    source_text: str | None = None


async def run_browser_use_lookup(
    lookup_input: LookupInput,
    settings: Settings,
    country: str | None = None,
    progress: ProgressCallback | None = None,
) -> LookupResult:
    _validate_browser_use_settings(settings)
    _disable_browser_use_telemetry()

    report(progress, "browser_use", "Using local browser-use. No Browser Use API or Cloud SDK is used.")
    report(
        progress,
        "browser_use",
        f"Local model: ollama ({settings.llm_model})",
    )
    browser_headless = _browser_use_headless(settings)
    report(
        progress,
        "browser_use",
        f"Local browser: headless={str(browser_headless).lower()}, max_steps={settings.browser_use_max_steps}",
    )

    profile = OrganizationProfile(
        queried_name=lookup_input.name,
        queried_website=str(lookup_input.website) if lookup_input.website else None,
        queried_country=_queried_country_value(country),
    )
    source_pages: list[WebsitePage] = []

    browser_use = _load_browser_use()
    llm = _build_browser_use_llm(browser_use, settings)
    browser = browser_use.Browser(
        headless=browser_headless,
        highlight_elements=True,
        dom_highlight_elements=True,
    )
    try:
        contact = await _run_stage(
            browser_use,
            llm,
            browser,
            _contact_task(lookup_input, country),
            BrowserUseContactExtraction,
            settings.browser_use_max_steps,
            progress,
            "Stage 1/3: extracting address, phone, email, country",
        )
        _merge_contact(profile, contact)
        _append_source_text(source_pages, profile.queried_website, "contact", contact.source_text)
        _report_stage_fields(progress, "Stage 1/3 complete", contact, exclude={"source_text"})

        facts = await _run_stage(
            browser_use,
            llm,
            browser,
            _facts_task(lookup_input, country),
            BrowserUseFactsExtraction,
            settings.browser_use_max_steps,
            progress,
            "Stage 2/3: extracting description, employees, legal structure",
        )
        _merge_facts(profile, facts)
        _append_source_text(source_pages, profile.queried_website, "facts", facts.source_text)
        _report_stage_fields(progress, "Stage 2/3 complete", facts, exclude={"source_text"})

        classification = await _run_stage(
            browser_use,
            llm,
            browser,
            _classification_task(lookup_input, settings, country),
            BrowserUseClassificationExtraction,
            settings.browser_use_max_steps,
            progress,
            "Stage 3/3: classifying sector, company type, industry",
        )
        _merge_classification(profile, classification)
        _append_source_text(source_pages, profile.queried_website, "classification", classification.source_text)
        _report_stage_fields(progress, "Stage 3/3 complete", classification, exclude={"source_text"})
    finally:
        await _close_browser(browser)

    await _validate_browser_use_profile(profile, source_pages, settings, country, progress)
    registry_results = await query_country_registry(
        country,
        lookup_input.name,
        settings.request_timeout,
        context_text=_registry_context_text(profile, source_pages),
        progress=progress,
        scope="call_registries",
    )
    registry_profile = _build_registry_profile(
        lookup_input.name,
        country,
        registry_results,
        progress,
        "finalize_profile",
    )
    report(progress, "finalize_profile", "Browser-use extraction complete.")
    return LookupResult(
        website_profile=profile,
        registry_profile=registry_profile,
        registry_message=_registry_result_message(country, registry_results, registry_profile),
    )


def _validate_browser_use_settings(settings: Settings) -> None:
    provider = (settings.llm_provider or "").strip().lower()
    if provider != "ollama":
        raise ValueError(
            "--browser-use local mode requires ORG_AGENT_LLM_PROVIDER=ollama. "
            "The Browser Use API and Cloud SDK are not used."
        )


def _disable_browser_use_telemetry() -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")


def _load_browser_use():
    try:
        import browser_use  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The browser-use package is not installed. Run `uv sync` before using --browser-use."
        ) from exc
    return browser_use


def _build_browser_use_llm(browser_use, settings: Settings):
    kwargs = {"model": settings.llm_model}
    if settings.ollama_base_url:
        kwargs["host"] = settings.ollama_base_url
    return browser_use.ChatOllama(**kwargs)


def _browser_use_headless(settings: Settings) -> bool:
    if settings.browser_use_headless is not None:
        return settings.browser_use_headless
    return settings.playwright_headless


async def _run_stage(
    browser_use,
    llm,
    browser,
    task: str,
    output_model: type[StructuredOutput],
    max_steps: int,
    progress: ProgressCallback | None,
    stage_message: str,
) -> StructuredOutput:
    report(progress, "browser_use", stage_message)
    agent = browser_use.Agent(
        task=task,
        llm=llm,
        browser=browser,
        output_model_schema=output_model,
        use_vision=False,
    )
    try:
        history = await agent.run(max_steps=max_steps)
    except Exception as exc:  # noqa: BLE001 - surface browser-agent failures with context
        raise RuntimeError(f"browser-use stage failed: {stage_message}: {exc}") from exc

    output = getattr(history, "structured_output", None)
    if output is not None:
        return output_model.model_validate(output)

    final_result = getattr(history, "final_result", lambda: None)()
    if not final_result:
        return output_model()
    try:
        return output_model.model_validate_json(str(final_result))
    except ValidationError:
        return output_model()


async def _close_browser(browser) -> None:
    close = getattr(browser, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def _contact_task(lookup_input: LookupInput, country: str | None) -> str:
    return (
        f"Go to the official website {lookup_input.website} for {lookup_input.name}. "
        "Find contact, footer, imprint, legal notice, about, or location pages as needed. "
        "Extract only address, phone, email, and headquarters/address country. "
        f"{_country_focus_instruction(country)}"
        "Do not guess. Use null for missing fields. Include short source_text containing the exact "
        "page text snippets that support the extracted contact fields."
    )


def _facts_task(lookup_input: LookupInput, country: str | None) -> str:
    return (
        f"Use the official website {lookup_input.website} for {lookup_input.name}. "
        "Extract only a neutral factual English description, employee count, and legal structure. "
        "The description must begin exactly with: '[Account Name] is engaged in ...'. "
        "Extract employees as an integer only if the website states or clearly estimates a count. "
        "Extract legal_structure only if stated explicitly. "
        f"{_country_focus_instruction(country)}"
        "Do not guess. Use null for missing fields. Include short source_text snippets that support "
        "the extracted values."
    )


def _classification_task(lookup_input: LookupInput, settings: Settings, country: str | None) -> str:
    return (
        f"Use the official website {lookup_input.website} for {lookup_input.name}. "
        "Classify the organization using only website evidence. "
        f"Choose sector from exactly one of these labels: {_csv_values(settings.sectors_csv)}. "
        f"Choose company_type from exactly one of these labels: {_csv_values(settings.company_types_csv)}. "
        f"Choose up to {settings.max_industries} industry label(s) from these labels: "
        f"{_csv_values(settings.industries_csv, limit=settings.industry_shortlist_size)}. "
        f"{_country_focus_instruction(country)}"
        "Do not invent labels. Use null for missing fields. For multiple industries, return a comma-separated string. "
        "Include short source_text snippets that support the classification."
    )


def _country_focus_instruction(country: str | None) -> str:
    if not country:
        return ""
    return f"Prioritize information for country code {country.upper()}. "


def _csv_values(path: str, limit: int | None = None) -> str:
    values = _load_industries(Path(path))
    if limit is not None:
        values = values[:limit]
    return "; ".join(values)


def _merge_contact(profile: OrganizationProfile, extraction: BrowserUseContactExtraction) -> None:
    profile.address = extraction.address or profile.address
    profile.phone = extraction.phone or profile.phone
    profile.email = extraction.email or profile.email
    profile.address_country = extraction.address_country or profile.address_country


def _merge_facts(profile: OrganizationProfile, extraction: BrowserUseFactsExtraction) -> None:
    profile.description = extraction.description or profile.description
    profile.employees = extraction.employees or profile.employees
    profile.legal_structure = extraction.legal_structure or profile.legal_structure


def _merge_classification(
    profile: OrganizationProfile,
    extraction: BrowserUseClassificationExtraction,
) -> None:
    profile.sector = extraction.sector or profile.sector
    profile.company_type = extraction.company_type or profile.company_type
    profile.industry = extraction.industry or profile.industry


def _append_source_text(
    source_pages: list[WebsitePage],
    website: str | None,
    title: str,
    source_text: str | None,
) -> None:
    if not source_text:
        return
    source_pages.append(WebsitePage(url=website or "browser-use", title=title, text=source_text))


def _report_stage_fields(
    progress: ProgressCallback | None,
    prefix: str,
    extraction: BaseModel,
    exclude: set[str],
) -> None:
    fields = [
        field
        for field, value in extraction.model_dump().items()
        if field not in exclude and value not in {None, ""}
    ]
    if fields:
        report(progress, "browser_use", f"{prefix}: {', '.join(fields)} found")
    else:
        report(progress, "browser_use", f"{prefix}: no fields found")


async def _validate_browser_use_profile(
    profile: OrganizationProfile,
    source_pages: list[WebsitePage],
    settings: Settings,
    country: str | None,
    progress: ProgressCallback | None,
) -> None:
    report(progress, "validate_profile", "Validating browser-use extracted profile...")
    derived_address_country = _derive_profile_country_from_address(profile)
    if derived_address_country:
        report(
            progress,
            "validate_profile",
            f'Extracted country "{derived_address_country}" from address: {profile.address}',
        )
    _normalize_profile_address_country(profile)
    _normalize_profile_country(profile)
    email_before = profile.email
    _validate_profile_email(profile, source_pages)
    if email_before and profile.email is None:
        report(progress, "validate_profile", f"Removed invalid or unconfirmed email address: {email_before}")
    elif email_before != profile.email and profile.email:
        report(progress, "validate_profile", f"Normalized email: {email_before} -> {profile.email}")
    elif profile.email:
        report(progress, "validate_profile", "Validated email from browser-use source text.")

    address_llm = build_chat_model(settings)
    await _fragment_profile_address(address_llm, profile, country, progress, "validate_profile")


def _registry_context_text(profile: OrganizationProfile, source_pages: list[WebsitePage]) -> str:
    parts = [
        value
        for value in (
            profile.queried_website,
            profile.legal_structure,
            profile.description,
            profile.sector,
            profile.company_type,
            profile.industry,
        )
        if value
    ]
    parts.extend(page.text for page in source_pages)
    return "\n".join(parts)
