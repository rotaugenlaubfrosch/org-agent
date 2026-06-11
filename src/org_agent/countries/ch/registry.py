from __future__ import annotations

import json
import os
import re

import httpx

from org_agent.models import EvidenceEntry, OrganizationProfilePatch, RegistryResult


BASE_URL = "https://www.zefix.admin.ch/ZefixPublicREST"
USERNAME_ENV = "ORG_AGENT_REGISTRY_CH_USERNAME"
PASSWORD_ENV = "ORG_AGENT_REGISTRY_CH_PASSWORD"


def has_credentials() -> bool:
    return bool((os.getenv(USERNAME_ENV) or "").strip() and (os.getenv(PASSWORD_ENV) or "").strip())


async def query_registry(
    client: httpx.AsyncClient,
    name: str,
    context_text: str | None = None,
) -> RegistryResult:
    del context_text
    username = os.getenv(USERNAME_ENV)
    password = os.getenv(PASSWORD_ENV)
    if not username or not password:
        raise ValueError(f"Missing Swiss registry credentials: {USERNAME_ENV} and {PASSWORD_ENV}")

    auth = (username, password)
    search_url = f"{BASE_URL}/api/v1/company/search"
    search_payload = {"name": name, "activeOnly": True}

    search_response = await client.post(search_url, json=search_payload, auth=auth)
    search_response.raise_for_status()
    candidates = search_response.json()
    if isinstance(candidates, list) and not candidates:
        fallback_name = _fallback_search_name(name)
        if fallback_name and fallback_name.lower() != name.strip().lower():
            search_response = await client.post(
                search_url,
                json={"name": fallback_name, "activeOnly": True},
                auth=auth,
            )
            search_response.raise_for_status()
            candidates = search_response.json()

    if not isinstance(candidates, list) or not candidates:
        return RegistryResult(
            registry="ch",
            url=search_url,
            status_code=search_response.status_code,
            content=search_response.text[:12000],
        )

    best = _pick_exact_match(name, candidates) or candidates[0]
    ehraid = best.get("ehraid")
    if ehraid is None:
        return RegistryResult(
            registry="ch",
            url=search_url,
            status_code=search_response.status_code,
            content=search_response.text[:12000],
        )

    detail_url = f"{BASE_URL}/api/v1/company/ehraid/{ehraid}"
    detail_response = await client.get(detail_url, auth=auth)
    detail_response.raise_for_status()
    detail = detail_response.json()

    patch = _build_profile_patch(detail, selected_name=best.get("name"))
    evidence = _build_evidence(patch, detail_url)
    content = json.dumps(
        {
            "search": candidates,
            "selected": best,
            "detail": detail,
        },
        ensure_ascii=False,
    )

    return RegistryResult(
        registry="ch",
        url=detail_url,
        status_code=detail_response.status_code,
        content=content[:12000],
        profile_patch=patch,
        evidence=evidence,
    )


def _pick_exact_match(name: str, candidates: list[dict]) -> dict | None:
    target = name.strip().lower()
    for item in candidates:
        candidate_name = str(item.get("name") or "").strip().lower()
        if candidate_name == target:
            return item
    return None


def _fallback_search_name(name: str) -> str:
    tokens = [token for token in _tokens(name) if token not in _legal_suffix_tokens()]
    return " ".join(tokens)


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", value.lower()) if token]


def _legal_suffix_tokens() -> set[str]:
    return {"ag", "sa", "ltd", "gmbh", "sagl", "holding"}


def _build_profile_patch(detail: dict, selected_name: str | None = None) -> OrganizationProfilePatch:
    legal_structure = _pick_legal_structure(detail.get("legalForm") or {})
    address = detail.get("address") or {}
    legal_address = _format_legal_address(detail.get("address") or {})
    return OrganizationProfilePatch(
        official_company_name=detail.get("name") or selected_name or address.get("organisation"),
        registration_id=detail.get("uid"),
        legal_structure=legal_structure,
        purpose=detail.get("purpose"),
        legal_address=legal_address,
        country="Switzerland",
        region=detail.get("canton") or detail.get("legalSeat"),
    )


def _pick_legal_structure(legal_structure: dict) -> str | None:
    short_name = legal_structure.get("shortName") or {}
    full_name = legal_structure.get("name") or {}
    for key in ("en", "de", "fr", "it"):
        value = short_name.get(key) or full_name.get(key)
        if value:
            return value
    return None


def _format_legal_address(address: dict) -> str | None:
    line = " ".join(part for part in [address.get("street"), address.get("houseNumber")] if part)
    locality = " ".join(part for part in [address.get("swissZipCode"), address.get("city")] if part)
    extras = [address.get("organisation"), address.get("careOf"), address.get("addon"), address.get("poBox")]
    parts = [part for part in extras if part]
    if line:
        parts.append(line)
    if locality:
        parts.append(locality)
    return ", ".join(parts) if parts else None


def _build_evidence(patch: OrganizationProfilePatch, source_url: str) -> list[EvidenceEntry]:
    evidence: list[EvidenceEntry] = []
    mappings = {
        "official_company_name": patch.official_company_name,
        "registration_id": patch.registration_id,
        "purpose": patch.purpose,
        "legal_address": patch.legal_address,
        "legal_structure": patch.legal_structure,
    }
    for field, value in mappings.items():
        if value:
            evidence.append(
                EvidenceEntry(
                    field=field,
                    value=value,
                    source=source_url,
                    reasoning="Extracted from the Swiss company register (Zefix).",
                )
            )
    return evidence
