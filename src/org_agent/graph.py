from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph

from org_agent.llm import build_chat_model
from org_agent.models import (
    AgentState,
    AppConfig,
    CrawlDecision,
    CrawlNode,
    CrawlTarget,
    EvidenceEntry,
    LookupInput,
    OrganizationProfile,
    OrganizationProfilePatch,
    PageAnalysis,
    PageExtraction,
    SearchResult,
    WebsiteLink,
    WebsitePage,
)
from org_agent.progress import ProgressCallback, report
from org_agent.registry import query_registries
from org_agent.search import build_search_provider
from org_agent.settings import Settings
from org_agent.website import (
    fetch_page_with_playwright,
    filter_candidate_links,
    normalize_url,
    report_crawl_tree,
    without_fragment,
)


EXTRACTABLE_PROFILE_FIELDS = (
    "legal_form",
    "industry",
    "description",
    "address",
    "phone",
    "email",
    "country",
)

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
    app_config: AppConfig,
    progress: ProgressCallback | None = None,
):
    llm = build_chat_model(settings)
    search_provider = build_search_provider(settings)
    page_extraction_parser = PydanticOutputParser(pydantic_object=PageExtraction)
    crawl_decision_parser = PydanticOutputParser(pydantic_object=CrawlDecision)

    async def initialize(state: AgentState) -> AgentState:
        report(progress, "input", f"Looking up: {state.input.name}")
        state.website = str(state.input.website) if state.input.website else None
        if state.website:
            report(progress, "input", f"Using provided website: {state.website}")
        state.profile = OrganizationProfile(queried_name=state.input.name, website=state.website)
        return state

    async def discover_website(state: AgentState) -> AgentState:
        if state.website:
            report(progress, "search", "Skipped website discovery because --website was provided.")
            return state

        try:
            query = f"{state.input.name} official website"
            report(progress, "search", f"Searching for official website: {query}")
            state.search_results = await search_provider.search(query)
            report(progress, "search", f"Received {len(state.search_results)} search result(s).")
            for result in state.search_results[:3]:
                report(progress, "search", f"Candidate: {result.title} -> {result.url}")
            state.website = _choose_website(state.search_results)
            if state.profile and state.website:
                state.profile.website = state.website
            if state.website is None:
                state.errors.append("No website was provided and search did not find a usable candidate.")
                report(progress, "search", "No usable website candidate was selected.")
            else:
                report(progress, "search", f"Selected website candidate: {state.website}")
        except Exception as exc:  # noqa: BLE001 - keep stateful error context for CLI output
            state.errors.append(f"Website discovery failed: {exc}")
            report(progress, "search", f"Website discovery failed: {exc}")
        return state

    async def call_registries(state: AgentState) -> AgentState:
        enabled_count = sum(1 for registry in app_config.registries if registry.enabled)
        if enabled_count == 0:
            return state
            report(progress, "registry", f"Querying {enabled_count} configured registry endpoint(s).")
        state.registry_results = await query_registries(
            state.input.name,
            app_config,
            settings.request_timeout,
            progress=progress,
        )
        if state.profile is not None:
            for result in state.registry_results:
                filled_fields = _merge_profile_patch(state.profile, result.profile_patch)
                _report_filled_fields(progress, "registry", filled_fields)
                _extend_evidence_dedup(state.profile.evidence, result.evidence)
        report(progress, "registry", f"Collected {len(state.registry_results)} registry result(s).")
        return state

    async def seed_crawl(state: AgentState) -> AgentState:
        if not state.website:
            return state
        root_url = normalize_url(state.website)
        state.website = root_url
        if state.profile:
            state.profile.website = root_url
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
        _report_candidate_links(progress, state.current_page.url, state.candidate_links, len(state.raw_links))
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
                state.profile.website,
                requested_fields,
                state.current_page,
                state.registry_results,
                progress,
            )
            _keep_requested_extraction_fields(extraction, requested_fields)
            _set_website_evidence_source(extraction.evidence, state.current_page.url)
        else:
            extraction = PageExtraction(reasoning="No missing profile fields remain before this page.")
        filled_fields = _merge_profile_patch(state.profile, extraction.profile_patch)
        _report_filled_fields(progress, "website", filled_fields)
        _extend_evidence_dedup(state.profile.evidence, extraction.evidence)
        remaining_missing_fields = _missing_profile_fields(state.profile)
        report(progress, "website", f"Extraction: {extraction.reasoning}")

        decision = await _select_next_links(
            llm,
            crawl_decision_parser,
            state.input.name,
            remaining_missing_fields,
            state.candidate_links,
            progress,
        )

        selected_urls = _normalize_selected_urls(decision.selected_urls[:3], state.candidate_links)
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
        _report_page_analysis(progress, analysis)

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

        state.should_continue_crawl = _should_continue_crawl(state, settings)
        return state

    async def finalize_profile(state: AgentState) -> AgentState:
        if state.profile is None:
            state.profile = OrganizationProfile(queried_name=state.input.name, website=state.website)
        state.profile.queried_name = state.input.name
        if state.website and not state.profile.website:
            state.profile.website = state.website
        _fill_registry_only_field_messages(state.profile, app_config)
        _write_crawl_text_logs(state, settings, progress)
        for error in state.errors:
            state.profile.evidence.append(
                EvidenceEntry(field="general", value=None, source="agent", reasoning=error)
            )
        report_crawl_tree(state.crawl_nodes, progress)
        report(progress, "extract", "Extraction complete.")
        return state

    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize)
    graph.add_node("discover_website", discover_website)
    graph.add_node("call_registries", call_registries)
    graph.add_node("seed_crawl", seed_crawl)
    graph.add_node("crawl_page", crawl_page)
    graph.add_node("filter_links", filter_links)
    graph.add_node("analyze_page", analyze_page)
    graph.add_node("finalize_profile", finalize_profile)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "discover_website")
    graph.add_edge("discover_website", "call_registries")
    graph.add_edge("call_registries", "seed_crawl")
    graph.add_edge("seed_crawl", "crawl_page")
    graph.add_edge("crawl_page", "filter_links")
    graph.add_edge("filter_links", "analyze_page")
    graph.add_conditional_edges(
        "analyze_page",
        lambda state: "crawl_page" if state.should_continue_crawl else "finalize_profile",
        {"crawl_page": "crawl_page", "finalize_profile": "finalize_profile"},
    )
    graph.add_edge("finalize_profile", END)
    return graph.compile()


async def run_lookup(
    lookup_input: LookupInput,
    settings: Settings,
    app_config: AppConfig,
    progress: ProgressCallback | None = None,
) -> OrganizationProfile:
    graph = build_graph(settings, app_config, progress=progress)
    final_state = await graph.ainvoke(AgentState(input=lookup_input))
    state = AgentState.model_validate(final_state)
    if state.profile is None:
        raise RuntimeError("Lookup did not produce an organization profile.")
    return state.profile


def _choose_website(search_results: list[SearchResult]) -> str | None:
    for result in search_results:
        if result.url.startswith(("http://", "https://")):
            return result.url
    return None


async def _extract_page_info(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    organization_name: str,
    website: str | None,
    requested_fields: list[str],
    current_page: WebsitePage,
    registry_results: list,
    progress: ProgressCallback | None,
) -> PageExtraction:
    formatted_fields = "\n".join(f"- {field}" for field in requested_fields)
    prompt = (
        "You are extracting factual information about an organization from a website page.\n"
        "Extract only these missing fields from the current page:\n"
        f"{formatted_fields}\n"
        "Do not extract, mention, or fill any fields that are not listed above. "
        "Use null for listed fields that are not present on the current page.\n"
        "Write brief evidence entries for each extracted value. "
        "Write a one-sentence factual explanation as reasoning.\n"
        "Do not include advertising language in the description; write a factual summary.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization name: {organization_name}\n\n"
        f"Known website: {website or 'unknown'}\n\n"
        f"Current page:\n{current_page.model_dump_json(indent=2)}\n\n"
        f"Registry results:\n{json.dumps([r.model_dump() for r in registry_results], ensure_ascii=False, indent=2)}"
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
        return result if isinstance(result, PageExtraction) else PageExtraction.model_validate(result)
    except Exception as exc:  # noqa: BLE001
        report(progress, "website", f"Structured extraction failed; retrying with JSON prompt. {exc}")
        return await _retry_json_parse(llm, parser, messages, progress=progress)


async def _select_next_links(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    organization_name: str,
    missing_fields: list[str],
    candidate_links: list[WebsiteLink],
    progress: ProgressCallback | None,
) -> CrawlDecision:
    available_links = [link.model_dump() for link in candidate_links[:80]]
    formatted_missing_fields = "\n".join(f"- {field}" for field in missing_fields) or "- none"
    missing_summary = ", ".join(missing_fields) or "none"
    report(
        progress,
        "website",
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
        report(progress, "website", "Calling LLM for link selection...")
        structured_llm = _link_selection_llm(llm)
        result = await structured_llm.ainvoke(messages)
        decision = _parse_crawl_decision_result(result, parser)
    except Exception as exc:  # noqa: BLE001
        report(progress, "website", f"Structured link selection failed; retrying with JSON prompt. {exc}")
        decision = await _retry_json_parse(llm, parser, messages, progress=progress)
    decision.selected_urls = decision.selected_urls[:3]
    report(progress, "website", f"LLM link selection returned {len(decision.selected_urls)} URL(s).")
    return decision


def _link_selection_llm(llm: BaseChatModel):
    if type(llm).__name__ == "ChatOllama":
        return llm.bind(format="json")
    return llm.with_structured_output(CrawlDecision)


def _parse_crawl_decision_result(result: object, parser: PydanticOutputParser) -> CrawlDecision:
    if isinstance(result, CrawlDecision):
        return result
    if isinstance(result, dict):
        return CrawlDecision.model_validate(result)
    content = getattr(result, "content", result)
    return parser.parse(str(content))


async def _retry_json_parse(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    messages: list,
    progress: ProgressCallback | None = None,
) -> PageExtraction | CrawlDecision:
    response = await llm.ainvoke(messages)
    try:
        return parser.parse(str(response.content))
    except OutputParserException:
        report(progress, "website", "Repairing invalid JSON response...")
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
        return parser.parse(str(repair_response.content))


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


def _report_filled_fields(
    progress: ProgressCallback | None,
    step: str,
    filled_fields: list[tuple[str, str]],
) -> None:
    for field, value in filled_fields:
        report(progress, step, f"Filled {field}: {_truncate_progress_value(value)}")


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


def _keep_requested_extraction_fields(extraction: PageExtraction, requested_fields: list[str]) -> None:
    requested = set(requested_fields)
    for field in extraction.profile_patch.__class__.model_fields:
        if field not in requested:
            setattr(extraction.profile_patch, field, None)
    extraction.evidence = [entry for entry in extraction.evidence if entry.field in requested]
    extraction.missing_fields = [field for field in extraction.missing_fields if field in requested]


def _fill_registry_only_field_messages(profile: OrganizationProfile, app_config: AppConfig) -> None:
    has_enabled_registry = any(registry.enabled for registry in app_config.registries)
    if has_enabled_registry:
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

    report(progress, "website", f"Saved crawl text logs to {run_dir}")


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


def _should_continue_crawl(state: AgentState, settings: Settings) -> bool:
    if len(state.website_pages) >= settings.crawl_max_pages:
        return False
    if not state.pending_urls:
        return False
    if state.page_analysis and state.page_analysis.is_complete and _has_minimum_profile(state.profile):
        return False
    return True


def _has_minimum_profile(profile: OrganizationProfile | None) -> bool:
    if profile is None:
        return False
    return bool(
        profile.website
        and profile.description
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
    page_url: str,
    links: list[WebsiteLink],
    raw_count: int,
) -> None:
    if progress is None:
        return
    report(
        progress,
        "website",
        f"Links given to LLM from {page_url}: {len(links)} candidate(s), filtered from {raw_count} raw link(s).",
    )
    if not links:
        report(progress, "website", "  No candidate links.")
        return
    for index, link in enumerate(links[:80], start=1):
        report(progress, "website", f"  [{index}] {link.text} -> {link.url} ({link.area})")


def _report_page_analysis(progress: ProgressCallback | None, analysis: PageAnalysis) -> None:
    if progress is None:
        return
    report(progress, "website", f"Crawl decision: {analysis.reasoning}")
    if analysis.missing_fields:
        report(progress, "website", f"Missing fields: {', '.join(analysis.missing_fields)}")
    if analysis.is_complete:
        report(progress, "website", "LLM selected: stop crawling")
        return
    if not analysis.selected_urls:
        report(progress, "website", "LLM selected: no links")
        return
    report(progress, "website", "LLM selected link(s):")
    for index, url in enumerate(analysis.selected_urls, start=1):
        report(progress, "website", f"  [{index}] {url}")
