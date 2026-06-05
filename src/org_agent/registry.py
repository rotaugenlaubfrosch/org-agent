from __future__ import annotations

import httpx

from org_agent.models import RegistryResult
from org_agent.progress import ProgressCallback, report


SUPPORTED_COUNTRIES = {"ch"}


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

    if normalized_country == "ch":
        from org_agent.countries.ch import registry as ch_registry

        if not ch_registry.has_credentials():
            report(
                progress,
                scope,
                "Skipped CH registry lookup because registry credentials are missing.",
            )
            return []

        try:
            report(progress, scope, f"Querying CH registry: {ch_registry.BASE_URL}")
            async with httpx.AsyncClient(timeout=timeout) as client:
                result = await ch_registry.query_registry(client, name, context_text=context_text)
            report(progress, scope, f"CH registry responded with HTTP {result.status_code}.")
            return [result]
        except Exception as exc:  # noqa: BLE001 - registry failures should not abort lookup
            report(progress, scope, f"CH registry failed: {exc}")
            return [
                RegistryResult(
                    registry="ch",
                    url=ch_registry.BASE_URL,
                    status_code=0,
                    content=f"Registry query failed: {exc}",
                )
            ]

    raise ValueError(_unsupported_country_message(normalized_country))


def normalize_country_code(country: str | None) -> str | None:
    normalized = (country or "").strip().lower()
    return normalized or None


def validate_country_code(country: str | None) -> None:
    normalized = normalize_country_code(country)
    if normalized is not None and normalized not in SUPPORTED_COUNTRIES:
        raise ValueError(_unsupported_country_message(normalized))


def _unsupported_country_message(country: str) -> str:
    supported = ", ".join(sorted(SUPPORTED_COUNTRIES))
    return f"Unsupported --country value: {country}. Supported values: {supported}."
