from __future__ import annotations

import json
from urllib.parse import urlparse

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph

from org_agent.llm import build_chat_model
from org_agent.models import (
    AgentState,
    AppConfig,
    CrawlNode,
    CrawlTarget,
    EvidenceEntry,
    FieldExtraction,
    LinkFilterDecision,
    LookupInput,
    NextUrlDecision,
    OrganizationProfile,
    OrganizationProfilePatch,
    OrchestratorDecision,
    PageAnalysis,
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
    field_extraction_parser = PydanticOutputParser(pydantic_object=FieldExtraction)
    next_url_parser = PydanticOutputParser(pydantic_object=NextUrlDecision)
    link_filter_parser = PydanticOutputParser(pydantic_object=LinkFilterDecision)

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

    async def crawl_page(state: AgentState) -> AgentState:
        state.current_page = None
        state.raw_links = []
        state.candidate_links = []
        state.page_analysis = None
        state.orchestrator_decision = None
        state.next_url_decision = None
        state.should_continue_crawl = False
        if not state.pending_urls or len(state.website_pages) >= settings.crawl_max_pages:
            return state

        target = state.pending_urls.pop(0)
        state.current_crawl_node_id = target.node_id
        state.current_crawl_depth = target.depth
        node = _crawl_node(state, target.node_id)
        normalized_url = without_fragment(target.url)
        cached_page = state.page_cache.get(normalized_url)
        if cached_page is not None:
            state.current_page = cached_page
            state.raw_links = state.raw_link_cache.get(normalized_url, [])
            node.final_url = cached_page.url
            node.status = "captured"
            node.reason = "loaded from cache"
            node.char_count = len(cached_page.text)
            report(progress, "website", f"Using cached page: {normalized_url}")
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
        state.page_cache[final_url] = page
        state.raw_link_cache[final_url] = links
        state.website_pages.append(page)
        node.status = "captured"
        node.char_count = len(page.text)
        return state

    async def filter_links(state: AgentState) -> AgentState:
        if not state.current_page or not state.website:
            return state
        cached_links = state.candidate_link_cache.get(state.current_page.url)
        if cached_links is not None:
            state.candidate_links = cached_links
            _report_candidate_links(progress, state.current_page.url, state.candidate_links, len(state.raw_links))
            return state
        state.candidate_links = filter_candidate_links(
            state.raw_links,
            root_url=state.website,
            current_url=state.current_page.url,
        )
        state.candidate_links = await _filter_links_with_llm(
            llm,
            link_filter_parser,
            state.current_page,
            state.candidate_links,
            progress,
        )
        state.candidate_link_cache[state.current_page.url] = state.candidate_links
        _report_candidate_links(progress, state.current_page.url, state.candidate_links, len(state.raw_links))
        return state

    async def orchestrator(state: AgentState) -> AgentState:
        if state.profile is None:
            return state

        if state.current_page is None:
            if not state.website:
                state.orchestrator_decision = OrchestratorDecision(
                    action="finalize",
                    reasoning="No website is available to crawl.",
                )
                return state
            root_url = normalize_url(state.website)
            state.website = root_url
            state.profile.website = root_url
            _queue_url(state, root_url, label="root", depth=0, parent_id=None)
            state.orchestrator_decision = OrchestratorDecision(
                action="crawl_page",
                reasoning="Crawling the root website page first.",
            )
            report(progress, "orchestrator", state.orchestrator_decision.reasoning)
            return state

        if state.current_field_task and state.field_extraction is None:
            state.orchestrator_decision = OrchestratorDecision(
                action="run_field_extractors",
                field_tasks=[state.current_field_task],
                reasoning=f"Retrying {state.current_field_task} on the current page.",
            )
            report(progress, "orchestrator", state.orchestrator_decision.reasoning)
            return state

        if state.field_tasks:
            state.current_field_task = _pop_next_field_task(state)
            if state.current_field_task is None:
                state.field_tasks = []
            else:
                state.orchestrator_decision = OrchestratorDecision(
                    action="run_field_extractors",
                    field_tasks=[state.current_field_task],
                    reasoning=f"Continuing queued task: {state.current_field_task}.",
                )
                report(progress, "orchestrator", state.orchestrator_decision.reasoning)
                return state

        if len(state.website_pages) >= settings.crawl_max_pages:
            state.orchestrator_decision = OrchestratorDecision(
                action="finalize",
                reasoning="Reached the maximum page crawl limit.",
            )
            report(progress, "orchestrator", state.orchestrator_decision.reasoning)
            return state

        state.field_tasks = _determine_field_tasks(state)
        state.current_field_task = _pop_next_field_task(state)
        if state.current_field_task is not None:
            decision = OrchestratorDecision(
                action="run_field_extractors",
                field_tasks=[state.current_field_task, *state.field_tasks],
                reasoning=f"Missing fields require extraction; running {state.current_field_task}.",
            )
            report(progress, "route", f"Next field task: {state.current_field_task}")
        else:
            decision = OrchestratorDecision(
                action="finalize",
                reasoning="No remaining field tasks for the current page.",
            )

        state.orchestrator_decision = decision
        report(progress, "orchestrator", decision.reasoning)
        report(progress, "route", f"Orchestrator action: {decision.action}")
        return state

    async def merge_result(state: AgentState) -> AgentState:
        if not state.current_page or not state.profile or not state.field_extraction:
            return state
        _record_field_attempt(state, state.current_page.url, state.field_extraction.field_task)
        _merge_profile_patch(state.profile, state.field_extraction.profile_patch)
        _extend_evidence_dedup(state.profile.evidence, state.field_extraction.evidence)
        report(progress, "extract", f"{state.field_extraction.field_task}: {state.field_extraction.reasoning}")
        state.current_field_task = None
        state.field_extraction = None
        return state

    async def field_extractor(state: AgentState) -> AgentState:
        if not state.current_page or state.profile is None or not state.current_field_task:
            return state
        state.field_extraction = await _extract_field_info(
            llm,
            field_extraction_parser,
            state.input.name,
            state.current_field_task,
            state.current_page,
            progress,
        )
        node = _crawl_node(state, state.current_crawl_node_id)
        node.reason = state.field_extraction.reasoning
        return state

    async def next_url_selector(state: AgentState) -> AgentState:
        if not state.current_page or state.profile is None or not state.current_field_task:
            return state
        state.next_url_decision = await _select_next_url(
            llm,
            next_url_parser,
            state.input.name,
            state.profile,
            state.current_field_task,
            state.current_page,
            state.candidate_links,
            state.visited_urls,
            settings.crawl_max_pages,
            settings.crawl_max_depth,
            state.current_crawl_depth,
            progress,
        )
        if state.next_url_decision.should_crawl:
            if _queue_selected_url(state, state.next_url_decision.selected_url, settings, progress):
                report(progress, "route", f"Next URL selected: {state.next_url_decision.selected_url}")
            else:
                state.next_url_decision.should_crawl = False
                _record_field_attempt(state, state.current_page.url, state.current_field_task)
                state.current_field_task = None
        else:
            _record_field_attempt(state, state.current_page.url, state.current_field_task)
            state.current_field_task = None
        report(progress, "route", f"Next URL decision: {state.next_url_decision.reasoning}")
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
    graph.add_node("crawl_page", crawl_page)
    graph.add_node("filter_links", filter_links)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("field_extractor", field_extractor)
    graph.add_node("merge_result", merge_result)
    graph.add_node("next_url_selector", next_url_selector)
    graph.add_node("finalize_profile", finalize_profile)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "discover_website")
    graph.add_edge("discover_website", "call_registries")
    graph.add_edge("call_registries", "orchestrator")
    graph.add_edge("crawl_page", "filter_links")
    graph.add_edge("filter_links", "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "crawl_page": "crawl_page",
            "field_extractor": "field_extractor",
            "finalize_profile": "finalize_profile",
        },
    )
    graph.add_conditional_edges(
        "field_extractor",
        lambda state: "merge_result" if state.field_extraction and state.field_extraction.found else "next_url_selector",
        {"merge_result": "merge_result", "next_url_selector": "next_url_selector"},
    )
    graph.add_edge("merge_result", "orchestrator")
    graph.add_conditional_edges(
        "next_url_selector",
        lambda state: "crawl_page" if state.next_url_decision and state.next_url_decision.should_crawl else "orchestrator",
        {"crawl_page": "crawl_page", "orchestrator": "orchestrator"},
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


async def _filter_links_with_llm(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    current_page: WebsitePage,
    links: list[WebsiteLink],
    progress: ProgressCallback | None,
) -> list[WebsiteLink]:
    if len(links) <= 5:
        return links
    available_links = [link.model_dump() for link in links[:80]]
    prompt = (
        "Select up to 5 links most likely to contain factual company information for an organization lookup.\n"
        "Prefer contact/kontakt, consumer service, impressum/imprint/legal/agb, about/company/unternehmen/story pages.\n"
        "Do not select shop, product, campaign, login, social, or media links.\n"
        "Only select URLs from candidate_links. Do not invent URLs.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Current page URL: {current_page.url}\n"
        f"Current page title: {current_page.title}\n\n"
        f"Candidate links:\n{json.dumps(available_links, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(content="You rank company-information links. Return structured output only."),
        HumanMessage(content=prompt),
    ]
    _report_llm_prompt(progress, "filter_links", prompt)
    try:
        structured_llm = llm.with_structured_output(LinkFilterDecision)
        result = await structured_llm.ainvoke(messages)
        decision = result if isinstance(result, LinkFilterDecision) else LinkFilterDecision.model_validate(result)
    except Exception as exc:  # noqa: BLE001
        report(progress, "website", f"Structured link filtering failed; using first 5 deterministic links. {exc}")
        return links[:5]
    selected = _normalize_selected_urls(decision.selected_urls[:5], links)
    _report_llm_response(progress, "filter_links", decision.model_dump())
    if not selected:
        return links[:5]
    return [link for link in links if link.url in selected][:5]


async def _select_next_url(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    organization_name: str,
    profile: OrganizationProfile,
    field_task: str,
    current_page: WebsitePage,
    candidate_links: list[WebsiteLink],
    visited_urls: set[str],
    crawl_max_pages: int,
    crawl_max_depth: int,
    current_depth: int,
    progress: ProgressCallback | None,
) -> NextUrlDecision:
    available_links = [link.model_dump() for link in candidate_links[:80]]
    prompt = (
        "You choose the next website link to crawl for one missing organization field.\n"
        f"The missing field task is: {field_task}.\n"
        "Your only job is link selection. Do not extract profile fields.\n"
        "Choose should_crawl=false if no candidate link is likely to contain this field.\n"
        "If should_crawl=true, select exactly one URL from candidate_links. Do not invent URLs.\n"
        "Prefer contact/kontakt pages for address, phone, or email; impressum/imprint/legal/agb pages "
        "for legal form or registration ID; and about/company/unternehmen/story pages for industry or description.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization name: {organization_name}\n\n"
        f"Missing field task: {field_task}\n\n"
        f"Current profile:\n{profile.model_dump_json(indent=2)}\n\n"
        f"Missing fields:\n{json.dumps(_missing_fields(profile), ensure_ascii=False, indent=2)}\n\n"
        f"Current page URL: {current_page.url}\n"
        f"Current page title: {current_page.title}\n\n"
        f"Visited URLs:\n{json.dumps(sorted(visited_urls), ensure_ascii=False, indent=2)}\n\n"
        f"Crawl limits: max_pages={crawl_max_pages}, max_depth={crawl_max_depth}, current_depth={current_depth}\n\n"
        f"Candidate links:\n{json.dumps(available_links, ensure_ascii=False, indent=2)}"
    )
    messages = [
        SystemMessage(content="You select one next URL to crawl. Return structured output only."),
        HumanMessage(content=prompt),
    ]
    _report_llm_prompt(progress, "next_url_selector", prompt)
    try:
        structured_llm = llm.with_structured_output(NextUrlDecision)
        result = await structured_llm.ainvoke(messages)
        decision = result if isinstance(result, NextUrlDecision) else NextUrlDecision.model_validate(result)
    except Exception as exc:  # noqa: BLE001
        report(progress, "route", f"Structured URL selection failed; retrying with JSON prompt. {exc}")
        decision = await _retry_json_parse(llm, parser, messages)
    decision.selected_url = _normalize_requested_url(decision.selected_url, candidate_links)
    if decision.should_crawl and decision.selected_url is None:
        decision.should_crawl = False
    _report_llm_response(progress, "next_url_selector", decision.model_dump())
    return decision


async def _extract_field_info(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    organization_name: str,
    field_task: str,
    current_page: WebsitePage,
    progress: ProgressCallback | None,
) -> FieldExtraction:
    target, rule = _field_task_prompt(field_task)
    prompt = (
        "You extract exactly one kind of company information from one webpage.\n"
        f"Target information: {target}.\n"
        f"Rule: {rule}\n"
        "Use only the current page text. Do not guess. Do not extract unrelated information.\n"
        "Ignore navigation menus, language selectors, product categories, login/register labels, cookie text, and generic footer noise.\n"
        "If the target information is present, set found=true and fill only the matching fields in profile_patch. "
        "Use the current page URL as evidence source and write one short evidence sentence.\n"
        "If the target information is not present, set found=false, leave profile_patch empty, and return no evidence.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Organization name: {organization_name}\n\n"
        f"Current page URL: {current_page.url}\n"
        f"Current page title: {current_page.title}\n\n"
        f"Current page text:\n{current_page.text}"
    )
    messages = [
        SystemMessage(content="You extract one assigned field task. Return structured output only."),
        HumanMessage(content=prompt),
    ]
    _report_llm_prompt(progress, f"field_extractor:{field_task}", prompt)
    try:
        structured_llm = llm.with_structured_output(FieldExtraction)
        result = await structured_llm.ainvoke(messages)
        extraction = result if isinstance(result, FieldExtraction) else FieldExtraction.model_validate(result)
    except Exception as exc:  # noqa: BLE001
        report(progress, "extract", f"Structured field extraction failed; retrying with JSON prompt. {exc}")
        extraction = await _retry_json_parse(llm, parser, messages)
    extraction.field_task = field_task
    extraction.requested_url = None
    _report_llm_response(progress, f"field_extractor:{field_task}", extraction.model_dump())
    return extraction


async def _retry_json_parse(
    llm: BaseChatModel,
    parser: PydanticOutputParser,
    messages: list,
) -> LinkFilterDecision | NextUrlDecision | FieldExtraction:
    response = await llm.ainvoke(messages)
    try:
        parsed = parser.parse(str(response.content))
        return parsed
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


def _report_llm_prompt(progress: ProgressCallback | None, label: str, prompt: str) -> None:
    report(progress, "llm", f"Prompt to {label}: {_shorten_debug_text(prompt)}")


def _report_llm_response(progress: ProgressCallback | None, label: str, response: dict) -> None:
    payload = json.dumps(response, ensure_ascii=False, indent=2)
    report(progress, "llm", f"Response from {label}: {_shorten_debug_text(payload)}")


def _shorten_debug_text(text: str, limit: int = 1600) -> str:
    compact = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}\n... <truncated {len(compact) - limit} chars>"


def _field_task_prompt(field_task: str) -> tuple[str, str]:
    if field_task == "address":
        return (
            "company physical address, plus country and region if stated",
            "A valid address must include a street, postal code, city, PO box, or equivalent physical location. Menu labels are not an address.",
        )
    if field_task == "phone":
        return (
            "company phone number",
            "Extract only a company contact phone number. Do not extract product numbers, IDs, years, or prices.",
        )
    if field_task == "email":
        return (
            "company email address",
            "Extract only a company contact email address.",
        )
    if field_task == "legal":
        return (
            "official company name and legal form",
            "Extract the official legal company name and legal form such as AG, GmbH, Ltd, Inc, LLC, SA, or equivalent.",
        )
    if field_task == "registration_id":
        return (
            "official registration, commercial register, tax, VAT, UID, or company identification number",
            "Extract only an official organization identifier, not phone numbers, postal codes, years, or product IDs.",
        )
    if field_task == "industry":
        return (
            "company industry and short factual description",
            "Return a neutral industry label and a short factual description without marketing language.",
        )
    return (field_task, "Extract only the requested information if it is explicitly present.")


def _merge_profile_patch(profile: OrganizationProfile, patch: OrganizationProfilePatch) -> None:
    for field, value in patch.model_dump().items():
        if value not in {None, ""}:
            setattr(profile, field, value)


def _extend_evidence_dedup(existing: list[EvidenceEntry], incoming: list[EvidenceEntry]) -> None:
    known = {(e.field, e.value) for e in existing if e.value is not None}
    for entry in incoming:
        if entry.value is not None and (entry.field, entry.value) not in known:
            existing.append(entry)
            known.add((entry.field, entry.value))


def _route_after_orchestrator(state: AgentState) -> str:
    decision = state.orchestrator_decision
    if decision and decision.action == "crawl_page" and state.pending_urls:
        return "crawl_page"
    if decision and decision.action == "run_field_extractors" and state.current_field_task:
        return "field_extractor"
    return "finalize_profile"


def _record_field_attempt(state: AgentState, page_url: str, task: str) -> None:
    attempts = state.field_task_attempts.setdefault(page_url, [])
    if task not in attempts:
        attempts.append(task)


def _pop_next_field_task(state: AgentState) -> str | None:
    while state.field_tasks:
        task = state.field_tasks.pop(0)
        if state.current_page and task in state.field_task_attempts.get(state.current_page.url, []):
            continue
        return task
    return None


def _determine_field_tasks(state: AgentState) -> list[str]:
    if state.profile is None:
        return []
    tasks = ["address", "phone", "email", "legal", "registration_id", "industry"]
    return [
        task
        for task in tasks
        if not _field_task_satisfied(task, state.profile)
        and state.current_page is not None
        and task not in state.field_task_attempts.get(state.current_page.url, [])
    ]


def _field_task_satisfied(task: str, profile: OrganizationProfile) -> bool:
    if task == "address":
        return bool(profile.address)
    if task == "phone":
        return bool(profile.phone)
    if task == "email":
        return bool(profile.email)
    if task == "legal":
        return bool(profile.legal_form)
    if task == "registration_id":
        return bool(profile.registration_id)
    if task == "industry":
        return bool(profile.industry and profile.description)
    return False


def _missing_fields(profile: OrganizationProfile) -> list[str]:
    missing = []
    for field in (
        "website",
        "registration_id",
        "legal_form",
        "industry",
        "description",
        "address",
        "phone",
        "email",
        "country",
        "region",
    ):
        if not getattr(profile, field):
            missing.append(field)
    return missing


def _normalize_requested_url(url: str | None, links: list[WebsiteLink]) -> str | None:
    if not url:
        return None
    available = {without_fragment(link.url): link.url for link in links}
    normalized = without_fragment(url)
    if normalized in available:
        return available[normalized]
    if normalized.startswith("/"):
        return next((link.url for link in links if urlparse(link.url).path == normalized), None)
    return None


def _queue_selected_url(
    state: AgentState,
    url: str | None,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> bool:
    selected_url = _normalize_requested_url(url, state.candidate_links)
    if selected_url is None:
        report(progress, "route", f"Rejected URL request because it is not in filtered links: {url}")
        return False
    if state.current_crawl_depth >= settings.crawl_max_depth:
        report(progress, "route", f"Rejected URL request at max crawl depth: {selected_url}")
        return False
    if selected_url in state.queued_urls or selected_url in state.visited_urls:
        report(progress, "route", f"Rejected URL request already queued/visited: {selected_url}")
        return False

    link = next((link for link in state.candidate_links if link.url == selected_url), None)
    parent = _crawl_node(state, state.current_crawl_node_id)
    _queue_url(
        state,
        selected_url,
        label=link.text if link else selected_url,
        depth=state.current_crawl_depth + 1,
        parent_id=parent.id,
    )
    report(progress, "route", f"Queued crawl URL: {selected_url}")
    return True


def _queue_url(
    state: AgentState,
    url: str,
    label: str,
    depth: int,
    parent_id: int | None,
) -> None:
    state.crawl_nodes.append(
        CrawlNode(
            id=state.next_crawl_node_id,
            parent_id=parent_id,
            requested_url=url,
            label=label,
            depth=depth,
        )
    )
    state.pending_urls.append(
        CrawlTarget(
            url=url,
            depth=depth,
            node_id=state.next_crawl_node_id,
        )
    )
    state.queued_urls.add(url)
    state.next_crawl_node_id += 1


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
    for url in selected_urls:
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
