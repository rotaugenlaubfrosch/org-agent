from __future__ import annotations

import json

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
        state.profile = OrganizationProfile(name=state.input.name, website=state.website)
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
        extraction = await _extract_page_info(
            llm,
            page_extraction_parser,
            state.input.name,
            state.profile,
            state.current_page,
            state.profile.evidence,
            state.registry_results,
            progress,
        )
        _merge_profile_patch(state.profile, extraction.profile_patch)
        _extend_evidence_dedup(state.profile.evidence, extraction.evidence)
        report(progress, "website", f"Extraction: {extraction.reasoning}")

        decision = await _select_next_links(
            llm,
            crawl_decision_parser,
            state.input.name,
            state.profile,
            extraction.missing_fields,
            state.candidate_links,
            progress,
        )

        selected_urls = _normalize_selected_urls(decision.selected_urls[:3], state.candidate_links)
        analysis = PageAnalysis(
            profile_patch=extraction.profile_patch,
            evidence=extraction.evidence,
            missing_fields=extraction.missing_fields,
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
            state.profile = OrganizationProfile(name=state.input.name, website=state.website)
        state.profile.name = state.input.name
        if state.website and not state.profile.website:
            state.profile.website = state.website
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
    partial_profile: OrganizationProfile,
    current_page: WebsitePage,
    prior_evidence: list[EvidenceEntry],
    registry_results: list,
    progress: ProgressCallback | None,
) -> PageExtraction:
    prompt = (
        "You are extracting factual information about an organization from a website page.\n"
        "Target fields: name, website, registration_id, legal_form, industry, description, "
        "address, phone, email, country, region.\n"
        "Only fill fields that are not already present in the partial profile. Use null for unknown fields.\n"
        "Write brief evidence entries for each extracted value. Use the page URL as the source. "
        "Write a one-sentence factual explanation as reasoning.\n"
        "Do not re-add evidence for field-value pairs that already appear in the previously extracted evidence.\n"
        "Do not include advertising language in the description; write a factual summary.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization name: {organization_name}\n\n"
        f"Current partial profile:\n{partial_profile.model_dump_json(indent=2)}\n\n"
        f"Current page:\n{current_page.model_dump_json(indent=2)}\n\n"
        f"Previously extracted evidence:\n{json.dumps([e.model_dump() for e in prior_evidence], ensure_ascii=False, indent=2)}\n\n"
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
        return await _retry_json_parse(llm, parser, messages)


async def _select_next_links(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    organization_name: str,
    profile: OrganizationProfile,
    missing_fields: list[str],
    candidate_links: list[WebsiteLink],
    progress: ProgressCallback | None,
) -> CrawlDecision:
    available_links = [link.model_dump() for link in candidate_links[:80]]
    prompt = (
        "You are deciding which website links to follow next while building an organization profile.\n"
        "Decide:\n"
        "1. Is the profile complete enough to stop? A good stopping point has: website, description, "
        "industry, and at least one of address, email, or phone. Do not keep crawling just because "
        "registration_id or legal_form is missing.\n"
        "2. If not complete, select up to 3 URLs most likely to fill the missing fields. "
        "Prefer contact, imprint/legal, about/company, privacy, or registration pages.\n"
        "3. Never invent URLs. Only select from the candidate links.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization: {organization_name}\n\n"
        f"Current profile:\n{profile.model_dump_json(indent=2)}\n\n"
        f"Missing fields: {', '.join(missing_fields) if missing_fields else 'none'}\n\n"
        f"Candidate links:\n{json.dumps(available_links, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(content="You decide which links to follow. Return structured output only."),
        HumanMessage(content=prompt),
    ]
    try:
        structured_llm = llm.with_structured_output(CrawlDecision)
        result = await structured_llm.ainvoke(messages)
        decision = result if isinstance(result, CrawlDecision) else CrawlDecision.model_validate(result)
    except Exception as exc:  # noqa: BLE001
        report(progress, "website", f"Structured link selection failed; retrying with JSON prompt. {exc}")
        decision = await _retry_json_parse(llm, parser, messages)
    decision.selected_urls = decision.selected_urls[:3]
    return decision


async def _retry_json_parse(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    messages: list,
) -> PageExtraction | CrawlDecision:
    response = await llm.ainvoke(messages)
    try:
        return parser.parse(str(response.content))
    except OutputParserException:
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


def _merge_profile_patch(profile: OrganizationProfile, patch: OrganizationProfilePatch) -> None:
    for field, value in patch.model_dump().items():
        if value not in {None, ""}:
            setattr(profile, field, value)


def _extend_evidence_dedup(existing: list[EvidenceEntry], incoming: list[EvidenceEntry]) -> None:
    known = {(e.field, e.value) for e in existing if e.value is not None}
    for entry in incoming:
        if entry.value is not None and (entry.field, entry.value) not in known:
            existing.append(entry)


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
