from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from org_agent.models import SearchResult
from org_agent.settings import Settings


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str) -> list[SearchResult]:
        raise NotImplementedError


class NoSearchProvider(SearchProvider):
    async def search(self, query: str) -> list[SearchResult]:
        return []


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str, timeout: float) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": 5},
            )
            response.raise_for_status()
        data = response.json()
        return [
            SearchResult(
                title=item.get("title") or item.get("url") or "Untitled",
                url=item.get("url", ""),
                snippet=item.get("content"),
            )
            for item in data.get("results", [])
            if item.get("url")
        ]


class BraveSearchProvider(SearchProvider):
    def __init__(self, api_key: str, timeout: float) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str) -> list[SearchResult]:
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {"q": query, "count": 5}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
        data: dict[str, Any] = response.json()
        results = data.get("web", {}).get("results", [])
        return [
            SearchResult(
                title=item.get("title") or item.get("url") or "Untitled",
                url=item.get("url", ""),
                snippet=item.get("description"),
            )
            for item in results
            if item.get("url")
        ]


def build_search_provider(settings: Settings) -> SearchProvider:
    provider = (settings.search_provider or "").lower().strip()

    if provider in {"", "none", "disabled"}:
        return NoSearchProvider()

    if not settings.search_api_key:
        raise ValueError("ORG_AGENT_SEARCH_API_KEY is required for configured search provider.")

    if provider == "tavily":
        return TavilySearchProvider(settings.search_api_key, settings.request_timeout)

    if provider == "brave":
        return BraveSearchProvider(settings.search_api_key, settings.request_timeout)

    raise ValueError("Unsupported ORG_AGENT_SEARCH_PROVIDER. Supported providers: tavily, brave, none.")
