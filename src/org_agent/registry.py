from __future__ import annotations

import importlib
import importlib.util

import httpx
import pycountry

from org_agent.models import RegistryResult
from org_agent.progress import ProgressCallback, report


async def query_country_registry(
    country: str | None,
    name: str,
    timeout: float,
    context_text: str | None = None,
    progress: ProgressCallback | None = None,
    scope: str = "registry",
) -> list[RegistryResult]:
    normalized_country = normalize_country_code(country)
    if normalized_country is None:
        report(progress, scope, "Skipped registry lookup because no country was selected.")
        return []

    registry_module = _registry_module(normalized_country)
    registry_country = normalized_country.upper()
    if registry_module is None:
        report(
            progress,
            scope,
            f"Skipped {registry_country} registry lookup because no registry integration is available.",
        )
        return []

    if not registry_module.has_credentials():
        report(
            progress,
            scope,
            f"Skipped {registry_country} registry lookup because registry credentials are missing.",
        )
        return []

    try:
        base_url = getattr(registry_module, "BASE_URL", f"org_agent.countries.{normalized_country}.registry")
        report(progress, scope, f"Querying {registry_country} registry: {base_url}")
        async with httpx.AsyncClient(timeout=timeout) as client:
            result = await registry_module.query_registry(client, name, context_text=context_text)
        report(progress, scope, f"{registry_country} registry responded with HTTP {result.status_code}.")
        return [result]
    except Exception as exc:  # noqa: BLE001 - registry failures should not abort lookup
        report(progress, scope, f"{registry_country} registry failed: {exc}")
        return [
            RegistryResult(
                registry=normalized_country,
                url=getattr(registry_module, "BASE_URL", f"org_agent.countries.{normalized_country}.registry"),
                status_code=0,
                content=f"Registry query failed: {exc}",
            )
        ]


def normalize_country_code(country: str | None) -> str | None:
    normalized = (country or "").strip().lower()
    if not normalized:
        return None
    if len(normalized) != 2 or not normalized.isalpha():
        raise ValueError(_invalid_country_message(country or ""))
    if pycountry.countries.get(alpha_2=normalized.upper()) is None:
        raise ValueError(_invalid_country_message(country or ""))
    return normalized


def validate_country_code(country: str | None) -> None:
    normalize_country_code(country)


def registry_module_available(country: str | None) -> bool:
    normalized_country = normalize_country_code(country)
    return normalized_country is not None and _registry_module(normalized_country) is not None


def _registry_module(country_code: str):
    module_name = f"org_agent.countries.{country_code}.registry"
    if importlib.util.find_spec(module_name) is None:
        return None
    return importlib.import_module(module_name)


def _invalid_country_message(country: str) -> str:
    return (
        f"Invalid --country value: {country}. "
        "Expected a two-letter ISO 3166-1 alpha-2 country code, for example CH."
    )
