from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class DerivationEntry(BaseModel):
    field: str = Field(description="The output field this evidence supports, or 'general'.")
    value: str | None = Field(default=None, description="The derived value, if applicable.")
    source: str | None = Field(default=None, description="URL, registry name, or process step.")
    reasoning: str = Field(description="Brief factual explanation of how this was derived.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


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
    derivation: list[DerivationEntry] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


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


class AgentState(BaseModel):
    input: LookupInput
    website: str | None = None
    search_results: list[SearchResult] = Field(default_factory=list)
    registry_results: list[RegistryResult] = Field(default_factory=list)
    website_pages: list[WebsitePage] = Field(default_factory=list)
    profile: OrganizationProfile | None = None
    errors: list[str] = Field(default_factory=list)
