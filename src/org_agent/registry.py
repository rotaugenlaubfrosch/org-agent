from __future__ import annotations

import json
import os

import httpx

from org_agent.models import (
    AppConfig,
    EvidenceEntry,
    OrganizationProfilePatch,
    RegistryEndpointConfig,
    RegistryResult,
)
from org_agent.progress import ProgressCallback, report


async def query_registries(
    name: str,
    app_config: AppConfig,
    timeout: float,
    progress: ProgressCallback | None = None,
    scope: str = "registry",
) -> list[RegistryResult]:
    results: list[RegistryResult] = []
    enabled_registries = [registry for registry in app_config.registries if registry.enabled]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for registry in enabled_registries:
            try:
                report(progress, scope, f"Querying {registry.name}: {registry.base_url}")
                provider = (registry.provider or "").lower().strip()
                if provider == "zefix" or registry.name.lower().strip() == "zefix":
                    result = await _query_zefix(client, name)
                else:
                    result = await _query_generic_registry(client, registry, name)
                report(progress, scope, f"{registry.name} responded with HTTP {result.status_code}.")
                results.append(result)
            except Exception as exc:  # noqa: BLE001 - registry failures should not abort lookup
                report(progress, scope, f"{registry.name} failed: {exc}")
                results.append(
                    RegistryResult(
                        registry=registry.name,
                        url=registry.base_url,
                        status_code=0,
                        content=f"Registry query failed: {exc}",
                    )
                )

    return results


async def _query_generic_registry(
    client: httpx.AsyncClient,
    registry: RegistryEndpointConfig,
    name: str,
) -> RegistryResult:
    headers: dict[str, str] = {}
    params = dict(registry.extra_params)
    params[registry.query_param] = name

    if registry.api_key_env and registry.api_key_header:
        api_key = os.getenv(registry.api_key_env)
        if api_key:
            headers[registry.api_key_header] = f"{registry.api_key_prefix}{api_key}"

    if registry.method == "POST":
        response = await client.post(registry.base_url, headers=headers, json=params)
    else:
        response = await client.get(registry.base_url, headers=headers, params=params)

    response.raise_for_status()
    return RegistryResult(
        registry=registry.name,
        url=str(response.request.url),
        status_code=response.status_code,
        content=response.text[:12000],
    )


async def _query_zefix(client: httpx.AsyncClient, name: str) -> RegistryResult:
    username = os.getenv("ORG_AGENT_ZEFIX_USERNAME")
    password = os.getenv("ORG_AGENT_ZEFIX_PASSWORD")
    if not username or not password:
        raise ValueError("Missing Zefix credentials: ORG_AGENT_ZEFIX_USERNAME and ORG_AGENT_ZEFIX_PASSWORD")

    base_url = "https://www.zefix.admin.ch/ZefixPublicREST"
    auth = (username, password)
    search_url = f"{base_url}/api/v1/company/search"
    search_payload = {"name": name, "activeOnly": True}

    search_response = await client.post(search_url, json=search_payload, auth=auth)
    search_response.raise_for_status()
    candidates = search_response.json()
    if not isinstance(candidates, list) or not candidates:
        return RegistryResult(
            registry="zefix",
            url=search_url,
            status_code=search_response.status_code,
            content=search_response.text[:12000],
        )

    best = _pick_best_zefix_match(name, candidates)
    ehraid = best.get("ehraid")
    if ehraid is None:
        return RegistryResult(
            registry="zefix",
            url=search_url,
            status_code=search_response.status_code,
            content=search_response.text[:12000],
        )

    detail_url = f"{base_url}/api/v1/company/ehraid/{ehraid}"
    detail_response = await client.get(detail_url, auth=auth)
    detail_response.raise_for_status()
    detail = detail_response.json()

    patch = _build_zefix_profile_patch(detail, selected_name=best.get("name"))
    evidence = _build_zefix_evidence(patch, detail_url)
    content = json.dumps(
        {
            "search": candidates,
            "selected": best,
            "detail": detail,
        },
        ensure_ascii=False,
    )

    return RegistryResult(
        registry="zefix",
        url=detail_url,
        status_code=detail_response.status_code,
        content=content[:12000],
        profile_patch=patch,
        evidence=evidence,
    )


def _pick_best_zefix_match(name: str, candidates: list[dict]) -> dict:
    target = name.strip().lower()
    for item in candidates:
        candidate_name = str(item.get("name") or "").strip().lower()
        if candidate_name == target:
            return item
    return candidates[0]


def _build_zefix_profile_patch(detail: dict, selected_name: str | None = None) -> OrganizationProfilePatch:
    legal_form = _pick_legal_form(detail.get("legalForm") or {})
    address = detail.get("address") or {}
    legal_address = _format_legal_address(detail.get("address") or {})
    return OrganizationProfilePatch(
        official_company_name=detail.get("name") or selected_name or address.get("organisation"),
        registration_id=detail.get("uid"),
        legal_form=legal_form,
        purpose=detail.get("purpose"),
        legal_address=legal_address,
        country="Switzerland",
        region=detail.get("canton") or detail.get("legalSeat"),
    )


def _pick_legal_form(legal_form: dict) -> str | None:
    short_name = legal_form.get("shortName") or {}
    full_name = legal_form.get("name") or {}
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


def _build_zefix_evidence(patch: OrganizationProfilePatch, source_url: str) -> list[EvidenceEntry]:
    evidence: list[EvidenceEntry] = []
    mappings = {
        "official_company_name": patch.official_company_name,
        "registration_id": patch.registration_id,
        "purpose": patch.purpose,
        "legal_address": patch.legal_address,
        "legal_form": patch.legal_form,
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
