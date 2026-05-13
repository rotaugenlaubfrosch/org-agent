from __future__ import annotations

import os

import httpx

from org_agent.models import AppConfig, RegistryEndpointConfig, RegistryResult
from org_agent.progress import ProgressCallback, report


async def query_registries(
    name: str,
    app_config: AppConfig,
    timeout: float,
    progress: ProgressCallback | None = None,
) -> list[RegistryResult]:
    results: list[RegistryResult] = []
    enabled_registries = [registry for registry in app_config.registries if registry.enabled]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for registry in enabled_registries:
            try:
                report(progress, "registry", f"Querying {registry.name}: {registry.base_url}")
                response = await _query_registry(client, registry, name)
                report(progress, "registry", f"{registry.name} responded with HTTP {response.status_code}.")
                results.append(
                    RegistryResult(
                        registry=registry.name,
                        url=str(response.request.url),
                        status_code=response.status_code,
                        content=response.text[:12000],
                    )
                )
            except Exception as exc:  # noqa: BLE001 - registry failures should not abort lookup
                report(progress, "registry", f"{registry.name} failed: {exc}")
                results.append(
                    RegistryResult(
                        registry=registry.name,
                        url=registry.base_url,
                        status_code=0,
                        content=f"Registry query failed: {exc}",
                    )
                )

    return results


async def _query_registry(
    client: httpx.AsyncClient,
    registry: RegistryEndpointConfig,
    name: str,
) -> httpx.Response:
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
    return response
