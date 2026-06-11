from __future__ import annotations

import json
import re
from csv import reader as csv_reader
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import pycountry
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from org_agent.llm import build_chat_model
from org_agent.models import (
    AgentState,
    CompanyTypeSelection,
    CrawlDecision,
    CrawlNode,
    CrawlTarget,
    EvidenceEntry,
    IndustrySelection,
    LegalStructureSelection,
    LookupResult,
    LookupInput,
    OrganizationProfile,
    OrganizationProfilePatch,
    PageAnalysis,
    PageExtraction,
    SectorSelection,
    WebsiteLink,
    WebsitePage,
)
from org_agent.progress import ProgressCallback, report
from org_agent.registry import normalize_country_code, query_country_registry
from org_agent.settings import Settings
from org_agent.website import (
    INFORMATION_LINK_SIGNALS,
    fetch_page_with_playwright,
    filter_candidate_links,
    normalize_url,
    report_crawl_tree,
    without_fragment,
)

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


EXTRACTABLE_PROFILE_FIELDS = (
    "address",
    "phone",
    "email",
    "country",
    "company_size",
)

DESCRIPTION_PROFILE_FIELD = "description"
LEGAL_STRUCTURE_PROFILE_FIELD = "legal_structure"
SECTOR_PROFILE_FIELD = "sector"
COMPANY_TYPE_PROFILE_FIELD = "company_type"
INDUSTRY_PROFILE_FIELD = "industry"
LINK_SELECTION_MAX_CANDIDATES = 25
ADDRESS_FIELD_PREFIX = "address_"

NO_REGISTRY_LEGAL_ADDRESS_MESSAGE = (
    "No third-party registry was attached; legal_address can only be provided by a registry."
)
NO_REGISTRY_PURPOSE_MESSAGE = (
    "No third-party registry was attached; purpose can only be provided by a registry."
)
NO_REGISTRY_REGISTRATION_ID_MESSAGE = (
    "No third-party registry was attached; registration_id can only be provided by a registry."
)
NO_REGISTRY_REGION_MESSAGE = (
    "No third-party registry was attached; region can only be provided by a registry."
)
NO_REGISTRY_OFFICIAL_COMPANY_NAME_MESSAGE = (
    "No third-party registry was attached; official_company_name can only be provided by a registry."
)


def build_graph(
    settings: Settings,
    country: str | None = None,
    progress: ProgressCallback | None = None,
):
    llm = build_chat_model(settings)
    page_extraction_parser = PydanticOutputParser(pydantic_object=PageExtraction)
    crawl_decision_parser = PydanticOutputParser(pydantic_object=CrawlDecision)
    legal_structure_selection_parser = PydanticOutputParser(pydantic_object=LegalStructureSelection)
    industry_selection_parser = PydanticOutputParser(pydantic_object=IndustrySelection)
    sector_selection_parser = PydanticOutputParser(pydantic_object=SectorSelection)
    company_type_selection_parser = PydanticOutputParser(pydantic_object=CompanyTypeSelection)

    async def initialize(state: AgentState) -> AgentState:
        report(progress, "initialize", f"Looking up: {state.input.name}")
        state.website = str(state.input.website) if state.input.website else None
        if state.website:
            report(progress, "initialize", f"Using provided website: {state.website}")
        state.profile = OrganizationProfile(
            queried_name=state.input.name,
            queried_website=state.website,
            queried_country=_queried_country_value(country),
        )
        return state

    async def call_registries(state: AgentState) -> AgentState:
        if not country:
            state.registry_message = (
                "Registry lookup was not called because no country registry was selected."
            )
            report(
                progress,
                "call_registries",
                "Skipped registry lookup because no country registry was selected.",
            )
            return state
        if not _registry_credentials_available(country):
            state.registry_message = _missing_registry_credentials_message(country)
            report(progress, "call_registries", state.registry_message)
            return state
        state.registry_results = await query_country_registry(
            country,
            state.input.name,
            settings.request_timeout,
            context_text=_registry_context_text(state),
            progress=progress,
            scope="call_registries",
        )
        if state.profile is not None:
            for result in state.registry_results:
                _report_registry_result_fields(progress, "call_registries", result.profile_patch)
        report(progress, "call_registries", f"Collected {len(state.registry_results)} registry result(s).")
        return state

    async def seed_crawl(state: AgentState) -> AgentState:
        if not state.website:
            return state
        root_url = normalize_url(state.website)
        state.website = root_url
        if state.profile:
            state.profile.queried_website = root_url
        state.pending_urls = [CrawlTarget(url=root_url, depth=0, node_id=0)]
        state.queued_urls = {root_url}
        state.crawl_nodes = [
            CrawlNode(id=0, parent_id=None, requested_url=root_url, label="root", depth=0)
        ]
        state.next_crawl_node_id = 1
        return state

    async def crawl_page(state: AgentState) -> AgentState:
        state.current_page = None
        state.raw_links = []
        state.candidate_links = []
        state.page_analysis = None
        state.should_continue_crawl = False
        if not state.pending_urls or len(state.website_pages) >= settings.crawl_max_pages:
            return state

        target = state.pending_urls.pop(0)
        state.current_crawl_node_id = target.node_id
        state.current_crawl_depth = target.depth
        node = _crawl_node(state, target.node_id)
        normalized_url = without_fragment(target.url)
        if normalized_url in state.visited_urls:
            node.status = "skipped"
            node.reason = "already visited"
            return state

        page, links, error = await fetch_page_with_playwright(
            target.url,
            headless=settings.playwright_headless,
            slow_mo=settings.playwright_slow_mo,
            progress=progress,
            scope="crawl_page",
        )
        if error or page is None:
            node.status = "skipped"
            node.reason = error or "page could not be loaded"
            return state

        final_url = without_fragment(page.url)
        node.final_url = final_url
        if final_url in state.visited_urls:
            node.status = "skipped"
            node.reason = f"redirected to already captured {final_url}"
            return state

        state.visited_urls.add(final_url)
        state.current_page = page
        state.raw_links = links
        state.website_pages.append(page)
        node.status = "captured"
        node.char_count = len(page.text)
        return state

    async def filter_links(state: AgentState) -> AgentState:
        if not state.current_page or not state.website:
            return state
        state.candidate_links = filter_candidate_links(
            state.raw_links,
            root_url=state.website,
            current_url=state.current_page.url,
        )
        _report_candidate_links(
            progress,
            "filter_links",
            state.current_page.url,
            state.candidate_links,
            len(state.raw_links),
        )
        return state

    async def analyze_page(state: AgentState) -> AgentState:
        if not state.current_page or state.profile is None:
            return state
        requested_fields = _missing_profile_fields(state.profile)
        if requested_fields:
            extraction = await _extract_page_info(
                llm,
                page_extraction_parser,
                state.input.name,
                state.profile.queried_website,
                requested_fields,
                state.current_page,
                progress,
                "analyze_page",
            )
            _keep_requested_extraction_fields(extraction, requested_fields)
            _set_website_evidence_source(extraction.evidence, state.current_page.url)
        else:
            extraction = PageExtraction(reasoning="No missing profile fields remain before this page.")
        filled_fields = _merge_profile_patch(state.profile, extraction.profile_patch)
        _report_filled_fields(progress, "analyze_page", filled_fields)
        _extend_evidence_dedup(state.profile.evidence, extraction.evidence)
        report(progress, "analyze_page", f"Extraction: {extraction.reasoning}")

        if state.profile.description in {None, ""}:
            description = await _extract_description(
                llm,
                settings.description_system_prompt,
                state.input.name,
                state.current_page,
                progress,
                "analyze_page",
            )
            was_empty = state.profile.description in {None, ""}
            state.profile.description = description
            if was_empty and description not in {None, ""}:
                _report_filled_fields(progress, "analyze_page", [("description", description)])
            state.profile.evidence.append(
                EvidenceEntry(
                    field="description",
                    value=description,
                    source=state.current_page.url,
                    reasoning="Generated by the dedicated description prompt from the current page text.",
                )
            )

        if state.profile.legal_structure in {None, ""}:
            legal_structure = await _extract_legal_structure(
                llm,
                legal_structure_selection_parser,
                state.current_page.text,
                settings.legal_structures_csv,
                progress,
                "analyze_page",
            )
            if legal_structure:
                state.profile.legal_structure = legal_structure
                _report_filled_fields(progress, "analyze_page", [("legal_structure", legal_structure)])
                state.profile.evidence.append(
                    EvidenceEntry(
                        field="legal_structure",
                        value=legal_structure,
                        source=state.current_page.url,
                        reasoning="Selected from the configured legal structure list using the dedicated legal structure prompt.",
                    )
                )

        if state.profile.sector in {None, ""}:
            sector = await _extract_sector(
                llm,
                sector_selection_parser,
                state.profile.description or "",
                state.current_page.text,
                settings.sectors_csv,
                progress,
                "analyze_page",
            )
            if sector:
                state.profile.sector = sector
                _report_filled_fields(progress, "analyze_page", [("sector", sector)])
                state.profile.evidence.append(
                    EvidenceEntry(
                        field="sector",
                        value=sector,
                        source=state.current_page.url,
                        reasoning="Selected from the configured sector list using the dedicated sector prompt.",
                    )
                )

        if state.profile.company_type in {None, ""}:
            company_type = await _extract_company_type(
                llm,
                company_type_selection_parser,
                state.profile.description or "",
                state.current_page.text,
                settings.company_types_csv,
                progress,
                "analyze_page",
            )
            if company_type:
                state.profile.company_type = company_type
                _report_filled_fields(progress, "analyze_page", [("company_type", company_type)])
                state.profile.evidence.append(
                    EvidenceEntry(
                        field="company_type",
                        value=company_type,
                        source=state.current_page.url,
                        reasoning="Selected from the configured company type list using the dedicated company type prompt.",
                    )
                )

        if state.profile.industry in {None, ""}:
            industries = await _extract_industries(
                llm,
                industry_selection_parser,
                state.profile.description or "",
                settings.industries_csv,
                settings.max_industries,
                settings.industry_shortlist_size,
                progress,
                "analyze_page",
            )
            if industries:
                industry_value = ", ".join(industries)
                state.profile.industry = industry_value
                _report_filled_fields(progress, "analyze_page", [("industry", industry_value)])
                state.profile.evidence.append(
                    EvidenceEntry(
                        field="industry",
                        value=industry_value,
                        source=state.current_page.url,
                        reasoning="Selected from the configured industry list using the dedicated industry prompt.",
                    )
                )

        remaining_missing_fields = _missing_crawl_fields(state.profile)

        decision = await _select_next_links(
            llm,
            crawl_decision_parser,
            state.input.name,
            remaining_missing_fields,
            state.candidate_links,
            progress,
            "analyze_page",
        )

        selected_urls = _normalize_selected_urls(decision.selected_urls[:3], state.candidate_links)
        _report_selected_links(progress, "analyze_page", state.candidate_links, selected_urls)
        analysis = PageAnalysis(
            profile_patch=OrganizationProfilePatch.model_validate(extraction.profile_patch.model_dump()),
            evidence=extraction.evidence,
            missing_fields=remaining_missing_fields,
            selected_urls=decision.selected_urls[:3],
            is_complete=decision.is_complete,
            reasoning=decision.reasoning,
        )
        state.page_analysis = analysis
        node = _crawl_node(state, state.current_crawl_node_id)
        node.reason = extraction.reasoning
        _report_page_analysis(progress, "analyze_page", analysis)

        queued_selected_link = False
        if state.current_crawl_depth < settings.crawl_max_depth:
            for link in state.candidate_links:
                if link.url not in selected_urls:
                    continue
                if link.url in state.queued_urls or link.url in state.visited_urls:
                    continue
                state.crawl_nodes.append(
                    CrawlNode(
                        id=state.next_crawl_node_id,
                        parent_id=node.id,
                        requested_url=link.url,
                        label=link.text,
                        depth=state.current_crawl_depth + 1,
                    )
                )
                state.pending_urls.append(
                    CrawlTarget(
                        url=link.url,
                        depth=state.current_crawl_depth + 1,
                        node_id=state.next_crawl_node_id,
                    )
                )
                state.queued_urls.add(link.url)
                state.next_crawl_node_id += 1
                queued_selected_link = True

        state.should_continue_crawl = _should_continue_crawl(
            state,
            settings,
            queued_selected_link=queued_selected_link,
        )
        return state

    async def validate_profile(state: AgentState) -> AgentState:
        if state.profile is None:
            report(progress, "validate_profile", "Skipped profile validation: no profile extracted.")
            return state
        report(progress, "validate_profile", "Validating extracted profile...")

        country_before = state.profile.country
        email_before = state.profile.email
        had_website_pages = bool(state.website_pages)

        _normalize_profile_country(state.profile)
        if country_before != state.profile.country:
            report(
                progress,
                "validate_profile",
                f"Normalized country: {country_before} -> {state.profile.country}",
            )

        _validate_profile_email(state.profile, state.website_pages)
        if email_before in {None, ""}:
            report(progress, "validate_profile", "Skipped email validation: no email extracted.")
        elif not had_website_pages:
            report(progress, "validate_profile", "Skipped email validation: no crawled website pages.")
        else:
            stripped_email = email_before.strip()
            if not stripped_email:
                report(progress, "validate_profile", "Removed blank email.")
            elif state.profile.email is None:
                report(
                    progress,
                    "validate_profile",
                    f"Removed email not found in crawled website pages: {stripped_email}",
                )
            else:
                if email_before != state.profile.email:
                    report(
                        progress,
                        "validate_profile",
                        f"Trimmed email whitespace: {email_before} -> {state.profile.email}",
                    )
                report(progress, "validate_profile", "Validated email: Found in crawled text")
        await _fragment_profile_address(llm, state.profile, country, progress, "validate_profile")
        return state

    async def finalize_profile(state: AgentState) -> AgentState:
        if state.profile is None:
            state.profile = OrganizationProfile(
                queried_name=state.input.name,
                queried_website=state.website,
                queried_country=_queried_country_value(country),
            )
        state.profile.queried_name = state.input.name
        state.profile.queried_country = _queried_country_value(country)
        if state.website and not state.profile.queried_website:
            state.profile.queried_website = state.website
        _write_crawl_text_logs(state, settings, progress, "finalize_profile")
        for error in state.errors:
            state.profile.evidence.append(
                EvidenceEntry(field="general", value=None, source="agent", reasoning=error)
            )
        report_crawl_tree(state.crawl_nodes, progress, scope="finalize_profile")
        report(progress, "finalize_profile", "Extraction complete.")
        return state

    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize)
    graph.add_node("call_registries", call_registries)
    graph.add_node("seed_crawl", seed_crawl)
    graph.add_node("crawl_page", crawl_page)
    graph.add_node("filter_links", filter_links)
    graph.add_node("analyze_page", analyze_page)
    graph.add_node("validate_profile", validate_profile)
    graph.add_node("finalize_profile", finalize_profile)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "seed_crawl")
    graph.add_edge("seed_crawl", "crawl_page")
    graph.add_edge("crawl_page", "filter_links")
    graph.add_edge("filter_links", "analyze_page")
    graph.add_conditional_edges(
        "analyze_page",
        lambda state: "crawl_page" if state.should_continue_crawl else "validate_profile",
        {"crawl_page": "crawl_page", "validate_profile": "validate_profile"},
    )
    graph.add_edge("validate_profile", "call_registries")
    graph.add_edge("call_registries", "finalize_profile")
    graph.add_edge("finalize_profile", END)
    return graph.compile()


async def run_lookup(
    lookup_input: LookupInput,
    settings: Settings,
    country: str | None = None,
    progress: ProgressCallback | None = None,
) -> LookupResult:
    graph = build_graph(settings, country=country, progress=progress)
    final_state = await graph.ainvoke(AgentState(input=lookup_input))
    state = AgentState.model_validate(final_state)
    if state.profile is None:
        raise RuntimeError("Lookup did not produce an organization profile.")
    registry_profile = _build_registry_profile(
        state.input.name,
        country,
        state.registry_results,
        progress,
        "finalize_profile",
    )
    registry_message = state.registry_message or _registry_result_message(
        country,
        state.registry_results,
        registry_profile,
    )
    return LookupResult(
        website_profile=state.profile,
        registry_profile=registry_profile,
        registry_message=registry_message,
    )


async def _extract_page_info(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    organization_name: str,
    website: str | None,
    requested_fields: list[str],
    current_page: WebsitePage,
    progress: ProgressCallback | None,
    scope: str,
) -> PageExtraction:
    formatted_fields = "\n".join(f"- {field}" for field in requested_fields)
    company_size_guidance = ""
    if "company_size" in requested_fields:
        company_size_guidance = (
            "For company_size, extract the number of employees as an integer. "
            "Rounded estimates are allowed when the page states an approximate employee count, "
            "for example 'around 100 employees' -> 100. Use null if no employee count is stated. "
            "Do not infer company_size from sector, industry, revenue, office count, or vague size labels. "
        )
    prompt = (
        "You are extracting factual information about an organization from a website page.\n"
        "Extract only these missing fields from the current page:\n"
        f"{formatted_fields}\n"
        "Do not extract, mention, or fill any fields that are not listed above. "
        "Use null for listed fields that are not present on the current page.\n"
        f"{company_size_guidance}"
        "Write brief evidence entries for each extracted value. "
        "Write a one-sentence factual explanation as reasoning.\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization name: {organization_name}\n\n"
        f"Known website: {website or 'unknown'}\n\n"
        f"Current page:\n{current_page.model_dump_json(indent=2)}"
    )
    messages = [
        SystemMessage(
            content="You extract factual organization data from web pages. Return structured output only."
        ),
        HumanMessage(content=prompt),
    ]
    try:
        structured_llm = llm.with_structured_output(PageExtraction)
        result = await structured_llm.ainvoke(messages)
        return _parse_structured_result(result, PageExtraction, parser)
    except Exception as exc:  # noqa: BLE001
        report(progress, scope, f"Structured extraction failed; retrying with JSON prompt. {exc}")
        return await _retry_json_parse(llm, PageExtraction, parser, messages, progress=progress, scope=scope)


async def _extract_description(
    llm: BaseChatModel,
    system_prompt: str,
    organization_name: str,
    current_page: WebsitePage,
    progress: ProgressCallback | None,
    scope: str,
) -> str:
    prompt = system_prompt.replace("[Account Name]", organization_name)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=current_page.text),
    ]
    report(progress, scope, "Calling LLM for description...")
    response = await llm.ainvoke(messages)
    return str(response.content)


async def _extract_industries(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    description: str,
    industries_csv: str,
    max_industries: int,
    shortlist_size: int,
    progress: ProgressCallback | None,
    scope: str,
) -> list[str]:
    industries = _load_industries(Path(industries_csv))
    if not industries:
        return []

    max_count = max(1, max_industries)
    shortlist_count = max(1, shortlist_size)
    candidates = industries
    if len(industries) > shortlist_count:
        report(
            progress,
            scope,
            f"More than {shortlist_count} industries were found in {industries_csv}. "
            "Therefore, industries are pre-filtered.",
        )
        report(
            progress,
            scope,
            f"Shortlisting {shortlist_count} of {len(industries)} configured industries...",
        )
        candidates = _shortlist_industries_by_embedding(description, industries, shortlist_count)
    else:
        report(
            progress,
            scope,
            f"{shortlist_count} or fewer industries were found in {industries_csv}. "
            "Therefore, industries are not pre-filtered.",
        )

    prompt = (
        f"Choose at most {max_count} industries from the candidate list. The industries need to match the organization, so can choose fewer industries if required."
        "Only choose industries that fit the organization description. "
        "Return JSON only. Do not include Markdown, commentary, explanations, or code fences.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization description:\n{description}\n\n"
        f"Candidate industries:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(
            content="You classify organizations into canonical industries. Return structured output only."
        ),
        HumanMessage(content=prompt),
    ]
    try:
        report(progress, scope, "Calling LLM for industry selection...")
        industry_llm = _industry_selection_llm(llm)
        result = await industry_llm.ainvoke(messages)
        selection = _parse_industry_selection_result(result, parser)
    except Exception as exc:  # noqa: BLE001
        report(progress, scope, f"Structured industry selection failed; retrying with JSON prompt. {exc}")
        selection = await _retry_json_parse(
            llm,
            IndustrySelection,
            parser,
            messages,
            progress=progress,
            scope=scope,
        )

    return _validate_selected_industries(selection.industries, industries, max_count)


async def _extract_legal_structure(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    page_text: str,
    legal_structures_csv: str,
    progress: ProgressCallback | None,
    scope: str,
) -> str | None:
    legal_structures = _load_industries(Path(legal_structures_csv))
    if not legal_structures:
        return None

    prompt = (
        "Choose exactly one legal structure from the candidate list. "
        "Only choose a legal structure that is supported by the current page text. "
        "Return JSON only. Do not include Markdown, commentary, explanations, or code fences.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Current page text:\n{page_text}\n\n"
        f"Candidate legal structures:\n{json.dumps(legal_structures, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(
            content="You classify organizations into canonical legal structures. Return structured output only."
        ),
        HumanMessage(content=prompt),
    ]
    try:
        report(progress, scope, "Calling LLM for legal_structure selection...")
        legal_structure_llm = _legal_structure_selection_llm(llm)
        result = await legal_structure_llm.ainvoke(messages)
        selection = _parse_legal_structure_selection_result(result, parser)
    except Exception as exc:  # noqa: BLE001
        report(progress, scope, f"Structured legal_structure selection failed; retrying with JSON prompt. {exc}")
        selection = await _retry_json_parse(
            llm,
            LegalStructureSelection,
            parser,
            messages,
            progress=progress,
            scope=scope,
        )

    return _validate_selected_legal_structure(selection.legal_structure, legal_structures)


async def _extract_sector(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    description: str,
    page_text: str,
    sectors_csv: str,
    progress: ProgressCallback | None,
    scope: str,
) -> str | None:
    sectors = _load_industries(Path(sectors_csv))
    if not sectors:
        return None

    prompt = (
        "Choose exactly one economic sector from the candidate list. "
        "Only choose a sector that fits the organization description and page context. "
        "Return JSON only. Do not include Markdown, commentary, explanations, or code fences.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization description:\n{description}\n\n"
        f"Current page text:\n{page_text}\n\n"
        f"Candidate sectors:\n{json.dumps(sectors, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(
            content="You classify organizations into canonical economic sectors. Return structured output only."
        ),
        HumanMessage(content=prompt),
    ]
    try:
        report(progress, scope, "Calling LLM for sector selection...")
        sector_llm = _sector_selection_llm(llm)
        result = await sector_llm.ainvoke(messages)
        selection = _parse_sector_selection_result(result, parser)
    except Exception as exc:  # noqa: BLE001
        report(progress, scope, f"Structured sector selection failed; retrying with JSON prompt. {exc}")
        selection = await _retry_json_parse(
            llm,
            SectorSelection,
            parser,
            messages,
            progress=progress,
            scope=scope,
        )

    return _validate_selected_sector(selection.sector, sectors)


async def _extract_company_type(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    description: str,
    page_text: str,
    company_types_csv: str,
    progress: ProgressCallback | None,
    scope: str,
) -> str | None:
    company_types = _load_industries(Path(company_types_csv))
    if not company_types:
        return None

    prompt = (
        "Choose exactly one company type from the candidate list. "
        "Only choose a company type that fits the organization description and page context. "
        "Return JSON only. Do not include Markdown, commentary, explanations, or code fences.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization description:\n{description}\n\n"
        f"Current page text:\n{page_text}\n\n"
        f"Candidate company types:\n{json.dumps(company_types, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(
            content="You classify organizations into canonical company types. Return structured output only."
        ),
        HumanMessage(content=prompt),
    ]
    try:
        report(progress, scope, "Calling LLM for company_type selection...")
        company_type_llm = _company_type_selection_llm(llm)
        result = await company_type_llm.ainvoke(messages)
        selection = _parse_company_type_selection_result(result, parser)
    except Exception as exc:  # noqa: BLE001
        report(progress, scope, f"Structured company_type selection failed; retrying with JSON prompt. {exc}")
        selection = await _retry_json_parse(
            llm,
            CompanyTypeSelection,
            parser,
            messages,
            progress=progress,
            scope=scope,
        )

    return _validate_selected_company_type(selection.company_type, company_types)


def _load_industries(path: Path) -> list[str]:
    rows = csv_reader(path.read_text(encoding="utf-8").splitlines())
    industries: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for value in row:
            industry = value.strip()
            if not industry or industry in seen:
                continue
            industries.append(industry)
            seen.add(industry)
    return industries


def _queried_country_value(country: str | None) -> str:
    country_code = _country_code(normalize_country_code(country))
    return country_code.upper() if country_code else "not specified"


def _registry_credentials_available(country: str | None) -> bool:
    normalized_country = normalize_country_code(country)
    if normalized_country == "ch":
        from org_agent.countries.ch import registry as ch_registry

        return ch_registry.has_credentials()
    return True


def _missing_registry_credentials_message(country: str | None) -> str:
    registry_country = _queried_country_value(country)
    return f"Registry lookup was not called because {registry_country} registry credentials are missing."


def _registry_result_message(
    country: str | None,
    registry_results: list,
    registry_profile: OrganizationProfile | None,
) -> str | None:
    if registry_profile is not None:
        return None
    if not country:
        return "Registry lookup was not called because no country registry was selected."
    if any(getattr(result, "status_code", None) == 0 for result in registry_results):
        return "Registry lookup failed; see registry trace output."
    return "Registry lookup completed but did not produce registry profile fields."


async def _fragment_profile_address(
    llm: BaseChatModel,
    profile: OrganizationProfile,
    country: str | None,
    progress: ProgressCallback | None,
    scope: str,
) -> None:
    if not profile.address:
        return

    config_path, config_reason = _address_fields_config_selection(country, profile.country)
    report(progress, scope, config_reason)
    if config_path is None:
        return

    config = _load_address_fields_config(config_path)
    report(progress, scope, f"Therefore, using address fields config: {config_path}")

    prompt = _address_fragmentation_prompt(profile.address, config)
    messages = [
        SystemMessage(content="You fragment postal addresses into configured JSON fields."),
        HumanMessage(content=prompt),
    ]
    report(progress, scope, "Calling LLM for address fragmentation...")
    response = await _address_fragmentation_llm(llm).ainvoke(messages)
    try:
        extracted = _parse_address_fragmentation_response(response)
    except ValueError as exc:
        report(progress, scope, f"Address fragmentation LLM output could not be parsed: {exc}")
        return

    report(
        progress,
        scope,
        f"Address fragmentation LLM output: {json.dumps(extracted, ensure_ascii=False, sort_keys=True)}",
    )
    profile.address_fields = _validated_address_fields(profile.address, extracted, config)


def _address_fields_config_path(country: str | None, profile_country: str | None) -> Path | None:
    config_path, _reason = _address_fields_config_selection(country, profile_country)
    return config_path


def _address_fields_config_selection(
    country: str | None,
    profile_country: str | None,
) -> tuple[Path | None, str]:
    explicit_country = (country or "").strip()
    if explicit_country:
        country_code = _country_code(explicit_country)
        if country_code is None:
            return (
                None,
                "Skipped address fragmentation because explicit "
                f"--country={explicit_country} could not be resolved to a country code.",
            )
        path = _address_fields_path(country_code)
        if path.exists():
            return (
                path,
                f"Address fields country source: explicit --country={explicit_country} -> {country_code}.",
            )
        return (
            None,
            "Skipped address fragmentation because explicit "
            f"--country={explicit_country} resolved to {country_code}, but no address fields "
            f"config exists for country: {country_code}.",
        )

    extracted_country = (profile_country or "").strip()
    if extracted_country:
        country_code = _country_code(extracted_country)
        if country_code is None:
            return (
                None,
                "Skipped address fragmentation because extracted "
                f"profile.country={extracted_country} could not be resolved to a country code.",
            )
        path = _address_fields_path(country_code)
        if path.exists():
            return (
                path,
                "Address fields country source: extracted "
                f"profile.country={extracted_country} -> {country_code}.",
            )
        return (
            None,
            "Skipped address fragmentation because extracted "
            f"profile.country={extracted_country} resolved to {country_code}, but no address fields "
            f"config exists for country: {country_code}.",
        )

    return (
        None,
        "Skipped address fragmentation because no country was provided or extracted.",
    )


def _address_fields_path(country_code: str) -> Path:
    return Path(__file__).parent / "countries" / country_code / "address_fields.json"


def _country_code(country: str | None) -> str | None:
    normalized = (country or "").strip()
    if not normalized:
        return None

    if len(normalized) == 2:
        return normalized.lower()

    try:
        match = pycountry.countries.lookup(normalized)
    except LookupError:
        return None
    return match.alpha_2.lower()


def _load_address_fields_config(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Address fields config must be a JSON object: {path}")

    config: dict[str, dict[str, object]] = {}
    for field, field_config in data.items():
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"Address fields config contains an invalid field name: {path}")
        if not isinstance(field_config, dict):
            raise ValueError(f"Address field {field!r} must be a JSON object: {path}")
        if set(field_config) != {"prompt", "validation"}:
            raise ValueError(
                f"Address field {field!r} must contain exactly prompt and validation: {path}"
            )
        prompt = field_config["prompt"]
        validation = field_config["validation"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Address field {field!r} prompt must be a non-empty string: {path}")
        if not isinstance(validation, bool):
            raise ValueError(f"Address field {field!r} validation must be a boolean: {path}")
        config[field.strip()] = {"prompt": prompt.strip(), "validation": validation}
    return config


def _address_fragmentation_prompt(address: str, config: dict[str, dict[str, object]]) -> str:
    fields = "\n".join(
        f"- {ADDRESS_FIELD_PREFIX}{field}: {field_config['prompt']}"
        for field, field_config in config.items()
    )
    return (
        "Fragment the address into custom address fields.\n\n"
        "Return JSON only. Do not include Markdown, commentary, explanations, or code fences.\n"
        "Only return configured fields.\n"
        "Prefix every returned field name with address_.\n"
        "Use null for fields that are not present.\n"
        "Do not invent values.\n\n"
        f"Address:\n{address}\n\n"
        f"Configured fields:\n{fields}"
    )


def _address_fragmentation_llm(llm: BaseChatModel):
    if type(llm).__name__ == "ChatOllama":
        return llm.bind(format="json")
    return llm


def _parse_address_fragmentation_response(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    content = str(getattr(response, "content", response)).strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").removesuffix("```").strip()
    elif content.startswith("```"):
        content = content.removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def _validated_address_fields(
    address: str,
    extracted: dict[str, Any],
    config: dict[str, dict[str, object]],
) -> dict[str, str]:
    address_fields: dict[str, str] = {}
    for configured_field, field_config in config.items():
        output_field = f"{ADDRESS_FIELD_PREFIX}{configured_field}"
        value = extracted.get(output_field, extracted.get(configured_field))
        if value in {None, ""}:
            continue
        value = str(value).strip()
        if not value:
            continue
        if field_config["validation"] and not _address_contains_value(address, value):
            continue
        address_fields[output_field] = value
    return address_fields


def _address_contains_value(address: str, value: str) -> bool:
    return _normalize_address_fragment(value) in _normalize_address_fragment(address)


def _normalize_address_fragment(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _shortlist_industries_by_embedding(
    description: str,
    industries: list[str],
    shortlist_size: int,
) -> list[str]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    texts = [f"query: {description}", *(f"passage: {industry}" for industry in industries)]
    embeddings = model.encode(texts, normalize_embeddings=True)
    description_embedding = embeddings[0]
    industry_embeddings = embeddings[1:]
    scores = industry_embeddings @ description_embedding
    ranked_indices = sorted(range(len(industries)), key=lambda index: float(scores[index]), reverse=True)
    return [industries[index] for index in ranked_indices[:shortlist_size]]


def _industry_selection_llm(llm: BaseChatModel):
    if type(llm).__name__ == "ChatOllama":
        return llm.bind(format="json")
    return llm.with_structured_output(IndustrySelection)


def _legal_structure_selection_llm(llm: BaseChatModel):
    if type(llm).__name__ == "ChatOllama":
        return llm.bind(format="json")
    return llm.with_structured_output(LegalStructureSelection)


def _sector_selection_llm(llm: BaseChatModel):
    if type(llm).__name__ == "ChatOllama":
        return llm.bind(format="json")
    return llm.with_structured_output(SectorSelection)


def _company_type_selection_llm(llm: BaseChatModel):
    if type(llm).__name__ == "ChatOllama":
        return llm.bind(format="json")
    return llm.with_structured_output(CompanyTypeSelection)


def _parse_industry_selection_result(
    result: object,
    parser: PydanticOutputParser,
) -> IndustrySelection:
    return _parse_structured_result(result, IndustrySelection, parser)


def _parse_legal_structure_selection_result(
    result: object,
    parser: PydanticOutputParser,
) -> LegalStructureSelection:
    return _parse_structured_result(result, LegalStructureSelection, parser)


def _parse_sector_selection_result(
    result: object,
    parser: PydanticOutputParser,
) -> SectorSelection:
    return _parse_structured_result(result, SectorSelection, parser)


def _parse_company_type_selection_result(
    result: object,
    parser: PydanticOutputParser,
) -> CompanyTypeSelection:
    return _parse_structured_result(result, CompanyTypeSelection, parser)


def _validate_selected_industries(
    selected: list[str],
    industries: list[str],
    max_count: int,
) -> list[str]:
    allowed = set(industries)
    valid: list[str] = []
    for industry in selected:
        if industry not in allowed or industry in valid:
            continue
        valid.append(industry)
        if len(valid) >= max_count:
            break
    return valid


def _validate_selected_legal_structure(
    selected: str | None,
    legal_structures: list[str],
) -> str | None:
    if selected in set(legal_structures):
        return selected
    return None


def _validate_selected_sector(selected: str | None, sectors: list[str]) -> str | None:
    if selected in set(sectors):
        return selected
    return None


def _validate_selected_company_type(
    selected: str | None,
    company_types: list[str],
) -> str | None:
    if selected in set(company_types):
        return selected
    return None


async def _select_next_links(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    organization_name: str,
    missing_fields: list[str],
    candidate_links: list[WebsiteLink],
    progress: ProgressCallback | None,
    scope: str,
) -> CrawlDecision:
    available_links = [link.model_dump() for link in candidate_links[:LINK_SELECTION_MAX_CANDIDATES]]
    formatted_missing_fields = "\n".join(f"- {field}" for field in missing_fields) or "- none"
    missing_summary = ", ".join(missing_fields) or "none"
    report(
        progress,
        scope,
        f"Selecting next links from {len(available_links)} candidate(s); missing fields: {missing_summary}",
    )
    prompt = (
        "You are choosing website links to visit while collecting company information.\n"
        "Prioritize links that are likely to contain these still-missing fields:\n"
        f"{formatted_missing_fields}\n"
        "Select up to 3 URLs most likely to contain factual company or organization data, "
        "such as about/company details, contact information, legal/imprint information, "
        "privacy information, registration details, address, email, or phone.\n"
        "Only select URLs from the candidate links. Do not invent URLs.\n"
        "If no candidate link looks useful, return no selected URLs and set is_complete to true; "
        "otherwise set is_complete to false.\n"
        "Return JSON only. Do not include Markdown, commentary, explanations, or code fences.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization: {organization_name}\n\n"
        f"Candidate links:\n{json.dumps(available_links, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(content="You decide which links to follow. Return structured output only."),
        HumanMessage(content=prompt),
    ]
    try:
        report(progress, scope, "Calling LLM for link selection...")
        structured_llm = _link_selection_llm(llm)
        result = await structured_llm.ainvoke(messages)
        decision = _parse_crawl_decision_result(result, parser)
    except Exception as exc:  # noqa: BLE001
        report(progress, scope, f"Structured link selection failed; retrying with JSON prompt. {exc}")
        decision = await _retry_json_parse(
            llm,
            CrawlDecision,
            parser,
            messages,
            progress=progress,
            scope=scope,
        )
    decision.selected_urls = decision.selected_urls[:3]
    report(progress, scope, f"LLM link selection returned {len(decision.selected_urls)} URL(s).")
    return decision


def _link_selection_llm(llm: BaseChatModel):
    if type(llm).__name__ == "ChatOllama":
        return llm.bind(format="json")
    return llm.with_structured_output(CrawlDecision)


def _parse_crawl_decision_result(result: object, parser: PydanticOutputParser) -> CrawlDecision:
    return _parse_structured_result(result, CrawlDecision, parser)


async def _retry_json_parse(
    llm: BaseChatModel,
    model_type: type[StructuredModel],
    parser: PydanticOutputParser,
    messages: list,
    progress: ProgressCallback | None = None,
    scope: str = "analyze_page",
) -> StructuredModel:
    response = await llm.ainvoke(messages)
    try:
        return _parse_structured_result(response, model_type, parser)
    except OutputParserException:
        report(progress, scope, "Repairing invalid JSON response...")
        repair_response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Convert the previous answer into valid JSON only. Do not include "
                        "Markdown, commentary, explanations, or code fences."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Schema instructions:\n{parser.get_format_instructions()}\n\n"
                        f"Previous answer:\n{response.content}"
                    )
                ),
            ]
        )
        return _parse_structured_result(repair_response, model_type, parser)


def _parse_structured_result(
    result: object,
    model_type: type[StructuredModel],
    parser: PydanticOutputParser,
) -> StructuredModel:
    if isinstance(result, model_type):
        return result
    if isinstance(result, dict):
        return _model_validate_with_properties_fallback(model_type, result)

    content = getattr(result, "content", result)
    try:
        return parser.parse(str(content))
    except OutputParserException as exc:
        try:
            parsed_content = json.loads(str(content))
        except json.JSONDecodeError:
            raise exc
        if isinstance(parsed_content, dict) and "properties" in parsed_content:
            try:
                return model_type.model_validate(parsed_content["properties"])
            except ValidationError:
                raise exc
        raise exc


def _model_validate_with_properties_fallback(
    model_type: type[StructuredModel],
    result: dict,
) -> StructuredModel:
    try:
        return model_type.model_validate(result)
    except ValidationError:
        if "properties" in result:
            return model_type.model_validate(result["properties"])
        raise


def _merge_profile_patch(
    profile: OrganizationProfile,
    patch: OrganizationProfilePatch,
) -> list[tuple[str, str]]:
    filled_fields: list[tuple[str, str]] = []
    for field, value in patch.model_dump().items():
        if value not in {None, ""}:
            was_empty = getattr(profile, field) in {None, ""}
            setattr(profile, field, value)
            if was_empty:
                filled_fields.append((field, str(value)))
    return filled_fields


def _normalize_profile_country(profile: OrganizationProfile) -> None:
    profile.country = _normalize_country(profile.country)
    for entry in profile.evidence:
        if entry.field == "country":
            entry.value = _normalize_country(entry.value)


def _normalize_country(value: str | None) -> str | None:
    if value is None:
        return None

    country = value.strip()
    if not country:
        return None

    upper_country = country.upper()
    if len(upper_country) == 2:
        match = pycountry.countries.get(alpha_2=upper_country)
        if match is not None:
            return match.name

    if len(upper_country) == 3:
        match = pycountry.countries.get(alpha_3=upper_country)
        if match is not None:
            return match.name

    try:
        return pycountry.countries.lookup(country).name
    except LookupError:
        pass

    try:
        return pycountry.countries.search_fuzzy(country)[0].name
    except LookupError:
        return country


def _validate_profile_email(profile: OrganizationProfile, website_pages: list[WebsitePage]) -> None:
    if not profile.email or not website_pages:
        return

    email = profile.email.strip()
    if not email:
        profile.email = None
        profile.evidence = [entry for entry in profile.evidence if entry.field != "email"]
        return

    source_text = "\n".join(page.text for page in website_pages).lower()
    if email.lower() in source_text:
        profile.email = email
        return

    profile.email = None
    profile.evidence = [entry for entry in profile.evidence if entry.field != "email"]


def _report_filled_fields(
    progress: ProgressCallback | None,
    step: str,
    filled_fields: list[tuple[str, str]],
) -> None:
    for field, value in filled_fields:
        report(progress, step, f"Filled {field}: {_truncate_progress_value(value)}")


def _report_registry_result_fields(
    progress: ProgressCallback | None,
    step: str,
    patch: OrganizationProfilePatch,
) -> None:
    for field, value in patch.model_dump().items():
        if value not in {None, ""}:
            report(progress, step, f"Registry returned {field}: {_truncate_progress_value(str(value))}")


def _build_registry_profile(
    queried_name: str,
    queried_country: str | None,
    registry_results: list,
    progress: ProgressCallback | None = None,
    step: str = "finalize_profile",
) -> OrganizationProfile | None:
    if not registry_results:
        return None

    profile = OrganizationProfile(
        queried_name=queried_name,
        queried_country=_queried_country_value(queried_country),
    )
    for result in registry_results:
        filled_fields = _merge_profile_patch(profile, result.profile_patch)
        _report_filled_fields(progress, step, filled_fields)
        _extend_evidence_dedup(profile.evidence, _matching_profile_evidence(profile, result.evidence))

    _normalize_profile_country(profile)
    if not any(getattr(profile, field) not in {None, ""} for field in OrganizationProfilePatch.model_fields):
        return None
    return profile


def _registry_context_text(state: AgentState) -> str:
    parts: list[str] = []
    if state.profile is not None:
        for value in (
            state.profile.queried_website,
            state.profile.legal_structure,
            state.profile.description,
            state.profile.sector,
            state.profile.company_type,
            state.profile.industry,
        ):
            if value:
                parts.append(value)
    parts.extend(page.text for page in state.website_pages)
    return "\n".join(parts)


def _merge_profile_patch_missing_only(
    profile: OrganizationProfile,
    patch: OrganizationProfilePatch,
) -> list[tuple[str, str]]:
    filled_fields: list[tuple[str, str]] = []
    for field, value in patch.model_dump().items():
        if value in {None, ""} or getattr(profile, field) not in {None, ""}:
            continue
        setattr(profile, field, value)
        filled_fields.append((field, str(value)))
    return filled_fields


def _matching_profile_evidence(
    profile: OrganizationProfile,
    evidence: list[EvidenceEntry],
) -> list[EvidenceEntry]:
    matching: list[EvidenceEntry] = []
    for entry in evidence:
        if not hasattr(profile, entry.field):
            continue
        if entry.value is None or getattr(profile, entry.field) == entry.value:
            matching.append(entry)
    return matching


def _truncate_progress_value(value: str, max_length: int = 60) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}..."


def _extend_evidence_dedup(existing: list[EvidenceEntry], incoming: list[EvidenceEntry]) -> None:
    known = {(e.field, e.value) for e in existing if e.value is not None}
    for entry in incoming:
        if entry.value is not None and (entry.field, entry.value) not in known:
            existing.append(entry)


def _missing_profile_fields(profile: OrganizationProfile) -> list[str]:
    return [
        field
        for field in EXTRACTABLE_PROFILE_FIELDS
        if getattr(profile, field) in {None, ""}
    ]


def _missing_crawl_fields(profile: OrganizationProfile) -> list[str]:
    fields = _missing_profile_fields(profile)
    if profile.legal_structure in {None, ""}:
        fields.insert(0, LEGAL_STRUCTURE_PROFILE_FIELD)
    if profile.industry in {None, ""}:
        fields.insert(1, INDUSTRY_PROFILE_FIELD)
    if profile.sector in {None, ""}:
        fields.insert(2, SECTOR_PROFILE_FIELD)
    if profile.company_type in {None, ""}:
        fields.insert(3, COMPANY_TYPE_PROFILE_FIELD)
    if profile.description in {None, ""}:
        fields.insert(2, DESCRIPTION_PROFILE_FIELD)
    return fields


def _keep_requested_extraction_fields(extraction: PageExtraction, requested_fields: list[str]) -> None:
    requested = set(requested_fields)
    for field in extraction.profile_patch.__class__.model_fields:
        if field not in requested:
            setattr(extraction.profile_patch, field, None)
    extraction.evidence = [entry for entry in extraction.evidence if entry.field in requested]
    extraction.missing_fields = [field for field in extraction.missing_fields if field in requested]


def _fill_registry_only_field_messages(
    profile: OrganizationProfile,
    has_registry_results: bool = False,
) -> None:
    if has_registry_results:
        return

    registry_only_fields = {
        "official_company_name": NO_REGISTRY_OFFICIAL_COMPANY_NAME_MESSAGE,
        "registration_id": NO_REGISTRY_REGISTRATION_ID_MESSAGE,
        "legal_address": NO_REGISTRY_LEGAL_ADDRESS_MESSAGE,
        "purpose": NO_REGISTRY_PURPOSE_MESSAGE,
        "region": NO_REGISTRY_REGION_MESSAGE,
    }
    for field, message in registry_only_fields.items():
        if getattr(profile, field):
            continue
        setattr(profile, field, message)
        profile.evidence.append(
            EvidenceEntry(
                field=field,
                value=message,
                source="agent",
                reasoning=f"No third-party registry was attached; {field} is not extracted from websites.",
            )
        )


def _set_website_evidence_source(evidence: list[EvidenceEntry], page_url: str) -> None:
    for entry in evidence:
        entry.source = page_url


def _write_crawl_text_logs(
    state: AgentState,
    settings: Settings,
    progress: ProgressCallback | None,
    scope: str,
) -> None:
    if not settings.crawl_log_enabled or not settings.crawl_log_dir or not state.website_pages:
        return

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(settings.crawl_log_dir) / f"{timestamp}-{_slugify(state.input.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for index, page in enumerate(state.website_pages, start=1):
        page_slug = _slugify(page.title or page.url)
        page_path = run_dir / f"{index:03d}-{page_slug}.txt"
        page_path.write_text(_format_page_log(page), encoding="utf-8")

    report(progress, scope, f"Saved crawl text logs to {run_dir}")


def _format_page_log(page: WebsitePage) -> str:
    title = page.title or ""
    return (
        f"URL: {page.url}\n"
        f"Title: {title}\n"
        f"Chars: {len(page.text)}\n"
        "\n"
        f"{page.text}\n"
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:80] or "page"


def _should_continue_crawl(
    state: AgentState,
    settings: Settings,
    queued_selected_link: bool = False,
) -> bool:
    if len(state.website_pages) >= settings.crawl_max_pages:
        return False
    if not state.pending_urls:
        return False
    if queued_selected_link:
        return True
    if state.page_analysis and state.page_analysis.is_complete and _has_minimum_profile(state.profile):
        return False
    return True


def _has_minimum_profile(profile: OrganizationProfile | None) -> bool:
    if profile is None:
        return False
    return bool(
        profile.queried_website
        and profile.legal_structure
        and profile.description
        and profile.sector
        and profile.company_type
        and profile.industry
        and (profile.address or profile.email or profile.phone)
        and profile.evidence
    )


def _crawl_node(state: AgentState, node_id: int | None) -> CrawlNode:
    for node in state.crawl_nodes:
        if node.id == node_id:
            return node
    raise ValueError(f"Crawl node not found: {node_id}")


def _normalize_selected_urls(selected_urls: list[str], links: list[WebsiteLink]) -> set[str]:
    available = {without_fragment(link.url): link.url for link in links}
    selected: set[str] = set()
    for url in selected_urls[:3]:
        normalized = without_fragment(url)
        if normalized in available:
            selected.add(available[normalized])
    return selected


def _report_candidate_links(
    progress: ProgressCallback | None,
    scope: str,
    page_url: str,
    links: list[WebsiteLink],
    raw_count: int,
) -> None:
    if progress is None:
        return
    report(
        progress,
        scope,
        "Filtered and ordered links given to LLM in analyze_page node "
        f"from {page_url}: {len(links)} candidate(s), filtered from {raw_count} raw link(s).",
    )
    report(progress, scope, f"Configured link keywords: {', '.join(INFORMATION_LINK_SIGNALS)}")
    if not links:
        report(progress, scope, "  No candidate links.")
        return
    for index, link in enumerate(links[:LINK_SELECTION_MAX_CANDIDATES], start=1):
        report(progress, scope, f"  [{index}] {link.text} -> {link.url} ({link.area})")


def _report_selected_links(
    progress: ProgressCallback | None,
    scope: str,
    links: list[WebsiteLink],
    selected_urls: set[str],
) -> None:
    if progress is None:
        return
    selected_links = [link for link in links if link.url in selected_urls]
    report(progress, scope, f"LLM selected {len(selected_links)} link(s).")
    for index, link in enumerate(selected_links, start=1):
        report(progress, scope, f"  [{index}] {link.text} -> {link.url} ({link.area})")


def _report_page_analysis(
    progress: ProgressCallback | None,
    scope: str,
    analysis: PageAnalysis,
) -> None:
    if progress is None:
        return
    report(progress, scope, f"Crawl decision: {analysis.reasoning}")
    if analysis.missing_fields:
        report(progress, scope, f"Missing fields: {', '.join(analysis.missing_fields)}")
    if analysis.is_complete:
        report(progress, scope, "LLM selected: stop crawling")
