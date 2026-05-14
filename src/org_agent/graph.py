from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph

from org_agent.llm import build_chat_model
from org_agent.models import (
    AgentState,
    AppConfig,
    EvidenceEntry,
    LookupInput,
    OrganizationProfile,
    SearchResult,
)
from org_agent.progress import ProgressCallback, report
from org_agent.registry import query_registries
from org_agent.search import build_search_provider
from org_agent.settings import Settings
from org_agent.website import crawl_website


def build_graph(
    settings: Settings,
    app_config: AppConfig,
    progress: ProgressCallback | None = None,
):
    llm = build_chat_model(settings)
    search_provider = build_search_provider(settings)
    parser = PydanticOutputParser(pydantic_object=OrganizationProfile)

    async def initialize(state: AgentState) -> AgentState:
        report(progress, "input", f"Looking up: {state.input.name}")
        state.website = str(state.input.website) if state.input.website else None
        if state.website:
            report(progress, "input", f"Using provided website: {state.website}")
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
            state.website = _choose_website(state.input.name, state.search_results)
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

    async def crawl_site(state: AgentState) -> AgentState:
        if not state.website:
            return state
        try:
            report(progress, "website", "Checking website links...")
            state.website_pages = await crawl_website(
                state.website,
                max_pages=settings.crawl_max_pages,
                max_depth=settings.crawl_max_depth,
                headless=settings.playwright_headless,
                slow_mo=settings.playwright_slow_mo,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            state.errors.append(f"Website crawl failed: {exc}")
            report(progress, "website", f"Website crawl failed: {exc}")
        return state

    async def extract_profile(state: AgentState) -> AgentState:
        report(progress, "extract", "Asking LLM to extract structured organization profile.")
        profile = await _extract_profile(llm, state, parser, progress)
        profile.name = state.input.name
        if state.website and not profile.website:
            profile.website = state.website
        state.profile = profile
        report(progress, "extract", "Extraction complete.")
        return state

    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize)
    graph.add_node("discover_website", discover_website)
    graph.add_node("call_registries", call_registries)
    graph.add_node("crawl_site", crawl_site)
    graph.add_node("extract_profile", extract_profile)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "discover_website")
    graph.add_edge("discover_website", "call_registries")
    graph.add_edge("call_registries", "crawl_site")
    graph.add_edge("crawl_site", "extract_profile")
    graph.add_edge("extract_profile", END)
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
    for error in state.errors:
        state.profile.evidence.append(
            EvidenceEntry(
                field="general",
                value=None,
                source="agent",
                reasoning=error,
            )
        )
    return state.profile


def _choose_website(name: str, search_results: list[SearchResult]) -> str | None:
    del name
    for result in search_results:
        if result.url.startswith(("http://", "https://")):
            return result.url
    return None


def _build_extraction_prompt(state: AgentState, parser: PydanticOutputParser) -> str:
    context = {
        "input_name": state.input.name,
        "candidate_website": state.website,
        "search_results": [result.model_dump() for result in state.search_results],
        "registry_results": [result.model_dump() for result in state.registry_results],
        "website_pages": [page.model_dump() for page in state.website_pages],
        "errors": state.errors,
    }
    return (
        "Extract an organization profile from the following evidence.\n"
        "Use null for unknown fields. Keep description short, structural, and factual.\n"
        "The evidence list must explain sources and decisions for important fields.\n\n"
        f"{parser.get_format_instructions()}\n\n"
        f"Evidence JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


async def _extract_profile(
    llm: BaseChatModel,
    state: AgentState,
    parser: PydanticOutputParser,
    progress: ProgressCallback | None,
) -> OrganizationProfile:
    messages = _build_extraction_messages(state, parser)
    try:
        structured_llm = llm.with_structured_output(OrganizationProfile)
        result = await structured_llm.ainvoke(messages)
        if isinstance(result, OrganizationProfile):
            report(progress, "extract", "Received schema-constrained LLM response.")
            return result
        report(progress, "extract", "Received schema-constrained mapping from LLM.")
        return OrganizationProfile.model_validate(result)
    except Exception as exc:  # noqa: BLE001 - not all local models support structured output
        report(progress, "extract", f"Structured output failed; retrying JSON prompt. {exc}")

    response = await llm.ainvoke(messages)
    try:
        return parser.parse(str(response.content))
    except OutputParserException:
        report(progress, "extract", "Model returned non-JSON text; asking it to repair the output.")

    repair_response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "Convert the previous answer into valid JSON only. Do not include Markdown, "
                    "commentary, explanations, or code fences."
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


def _build_extraction_messages(
    state: AgentState,
    parser: PydanticOutputParser,
) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(
            content=(
                "You extract factual organization profiles. Avoid marketing language. "
                "Prefer evidence from official websites and registry responses. "
                "Return exactly one organization profile for the input organization. "
                "Return structured JSON only when not using native structured output."
            )
        ),
        HumanMessage(content=_build_extraction_prompt(state, parser)),
    ]
