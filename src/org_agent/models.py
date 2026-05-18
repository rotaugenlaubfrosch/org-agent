from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class EvidenceEntry(BaseModel):
    field: str = Field(description="The output field this evidence supports, or 'general'.")
    value: str | None = Field(default=None, description="The derived value, if applicable.")
    source: str | None = Field(default=None, description="URL, registry name, or process step.")
    reasoning: str = Field(description="Brief factual explanation of how this was derived.")


class OrganizationProfile(BaseModel):
    name: str
    website: str | None = None
    registration_id: str | None = None
    legal_form: str | None = None
    industry: str | None = None
    description: str | None = Field(
        default=None,
        description="Short factual structural description without advertising language.",
    )
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    country: str | None = None
    region: str | None = None
    evidence: list[EvidenceEntry] = Field(default_factory=list)


class OrganizationProfilePatch(BaseModel):
    name: str | None = None
    website: str | None = None
    registration_id: str | None = None
    legal_form: str | None = None
    industry: str | None = None
    description: str | None = None
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    country: str | None = None
    region: str | None = None


class LookupInput(BaseModel):
    name: str
    website: HttpUrl | None = None


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str | None = None


class RegistryEndpointConfig(BaseModel):
    name: str
    base_url: str
    method: Literal["GET", "POST"] = "GET"
    query_param: str = "q"
    api_key_env: str | None = None
    api_key_header: str | None = None
    api_key_prefix: str = ""
    enabled: bool = True
    extra_params: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    registries: list[RegistryEndpointConfig] = Field(default_factory=list)


class RegistryResult(BaseModel):
    registry: str
    url: str
    status_code: int
    content: str


class WebsitePage(BaseModel):
    url: str
    title: str | None = None
    text: str


class WebsiteLink(BaseModel):
    url: str
    text: str
    area: str = "body"


class CrawlDecision(BaseModel):
    is_complete: bool = Field(
        description="Whether enough evidence has been collected to extract the organization profile."
    )
    selected_urls: list[str] = Field(
        default_factory=list,
        description="URLs from the available links that should be visited next.",
    )
    reasoning: str = Field(description="Brief explanation of the crawl decision.")


class PageExtraction(BaseModel):
    profile_patch: OrganizationProfilePatch = Field(
        default_factory=OrganizationProfilePatch,
        description="Fields that can be filled or updated from the current page.",
    )
    evidence: list[EvidenceEntry] = Field(default_factory=list)
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Profile fields still missing after this extraction.",
    )
    reasoning: str = Field(
        description="Brief explanation of what was extracted and from where on the page.",
    )


class PageAnalysis(BaseModel):
    profile_patch: OrganizationProfilePatch = Field(
        default_factory=OrganizationProfilePatch,
        description="Fields that can be filled or updated from the current evidence.",
    )
    evidence: list[EvidenceEntry] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    selected_urls: list[str] = Field(
        default_factory=list,
        description="URLs from candidate_links that should be visited next.",
    )
    is_complete: bool = False
    reasoning: str


class CrawlTarget(BaseModel):
    url: str
    depth: int = 0
    node_id: int = 0


class CrawlNode(BaseModel):
    id: int
    parent_id: int | None = None
    requested_url: str
    label: str
    depth: int = 0
    final_url: str | None = None
    status: str = "queued"
    reason: str | None = None
    char_count: int | None = None


class AgentState(BaseModel):
    input: LookupInput
    website: str | None = None
    search_results: list[SearchResult] = Field(default_factory=list)
    registry_results: list[RegistryResult] = Field(default_factory=list)
    website_pages: list[WebsitePage] = Field(default_factory=list)
    current_page: WebsitePage | None = None
    current_crawl_node_id: int | None = None
    current_crawl_depth: int = 0
    raw_links: list[WebsiteLink] = Field(default_factory=list)
    candidate_links: list[WebsiteLink] = Field(default_factory=list)
    pending_urls: list[CrawlTarget] = Field(default_factory=list)
    visited_urls: set[str] = Field(default_factory=set)
    queued_urls: set[str] = Field(default_factory=set)
    crawl_nodes: list[CrawlNode] = Field(default_factory=list)
    next_crawl_node_id: int = 1
    page_analysis: PageAnalysis | None = None
    profile: OrganizationProfile | None = None
    should_continue_crawl: bool = False
    errors: list[str] = Field(default_factory=list)
