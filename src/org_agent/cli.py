from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from org_agent.api import lookup_organization
from org_agent.models import LookupResult, OrganizationProfile, profile_display_field_groups

HELP_TEXT = """Enrich organization profiles from a name and required website.

Required environment variables:
ORG_AGENT_LLM_PROVIDER=openai|anthropic|ollama
ORG_AGENT_LLM_MODEL=<model name>

Required for OpenAI/Anthropic:
ORG_AGENT_API_KEY=<provider API key>

Required for Ollama:
ORG_AGENT_OLLAMA_BASE_URL=<Ollama base URL>

Optional runtime environment variables:
ORG_AGENT_REQUEST_TIMEOUT=<seconds, default 20>
ORG_AGENT_CRAWL_MAX_PAGES=<pages, default 6>
ORG_AGENT_CRAWL_MAX_DEPTH=<link depth, default 2>
ORG_AGENT_CRAWL_LOG_ENABLED=<true|false, default true>
ORG_AGENT_CRAWL_LOG_DIR=<directory for per-run page text logs, default project logs/>
ORG_AGENT_PLAYWRIGHT_HEADLESS=<true|false, default true>
ORG_AGENT_PLAYWRIGHT_SLOW_MO=<milliseconds, default 0>

Optional country registry credentials:
ORG_AGENT_REGISTRY_CH_USERNAME=<Swiss registry username>
ORG_AGENT_REGISTRY_CH_PASSWORD=<Swiss registry password>

Common lookup options:
  --website <url>  Required official website. Bare domains like example.com are accepted.
  --country <code> Two-letter ISO country code for country-specific behavior (e.g. ch)
  --json           Print raw JSON output
  --quiet          Suppress progress output

Examples:
  org-agent "Example Ltd" --website https://example.com
  org-agent "Example Ltd" --website example.com --quiet
"""

app = typer.Typer(help=HELP_TEXT, add_completion=False)
console = Console()
err_console = Console(stderr=True)


@app.command()
def main(
    ctx: typer.Context,
    name: str | None = typer.Argument(
        None,
        help="Organization name, assumed to be the correct name.",
    ),
    website: str | None = typer.Option(
        None,
        "--website",
        "-w",
        help="Required official website. Bare domains like example.com are accepted.",
    ),
    country: str | None = typer.Option(
        None,
        "--country",
        help="Two-letter ISO country code for country-specific behavior, e.g. ch.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON output."),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress the live trace of registry, website crawl, and extraction steps.",
    ),
) -> None:
    """Enrich organization profiles from a name and required website."""
    if name is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    if website is None:
        err_console.print("[red]Lookup failed:[/red] --website is required.")
        raise typer.Exit(code=1)

    try:
        if quiet:
            result = lookup_organization(
                name=name,
                website=website,
                country=country,
            )
        else:
            err_console.rule("[bold cyan]org-agent trace")
            result = lookup_organization(
                name=name,
                website=website,
                country=country,
                progress=_make_progress_logger(),
            )
            err_console.rule("[bold green]done")
    except Exception as exc:  # noqa: BLE001 - CLI should display concise failures
        err_console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return

    _print_lookup_result(result)


def _print_lookup_result(result: LookupResult) -> None:
    website_fields, registry_fields = profile_display_field_groups()
    _print_profile_table("Website Profile", result.website_profile, website_fields)
    if result.registry_profile is not None:
        _print_profile_table("Registry Profile", result.registry_profile, registry_fields)
    else:
        _print_registry_status_table(
            result.registry_message or "Registry lookup did not produce a registry profile."
        )

    _print_evidence_panel("Website Evidence", result.website_profile)
    if result.registry_profile is not None:
        _print_evidence_panel("Registry Evidence", result.registry_profile)


def _print_profile_table(title: str, profile: OrganizationProfile, fields: tuple[str, ...]) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    for field in fields:
        if field == "status":
            table.add_section()
        value = getattr(profile, field)
        table.add_row(field, "" if value is None else str(value))
        if field == "queried_country":
            table.add_section()
        if field == "address":
            for address_field, address_value in profile.address_fields.items():
                table.add_row(f"  {address_field}", address_value)

    console.print(table)


def _print_registry_status_table(message: str) -> None:
    table = Table(title="Registry Profile", show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("status", message)
    console.print(table)


def _print_evidence_panel(title: str, profile: OrganizationProfile) -> None:
    lines = []
    for entry in profile.evidence:
        parts = []
        if entry.source:
            parts.append(f"Source: {entry.source}.")
        parts.append(entry.reasoning)
        lines.append(f"[bold]{entry.field}[/bold]: {' '.join(parts)}")
    console.print(Panel("\n".join(lines) or "No evidence entries returned.", title=title))


def _make_progress_logger():
    styles = {
        "config": "magenta",
        "input": "cyan",
        "registry": "yellow",
        "website": "green",
        "extract": "bright_magenta",
        "initialize": "color(51)",
        "call_registries": "green",
        "seed_crawl": "color(45)",
        "crawl_page": "blue",
        "filter_links": "color(118)",
        "analyze_page": "color(201)",
        "validate_profile": "color(51)",
        "finalize_profile": "yellow",
    }

    def log(step: str, message: str) -> None:
        style = styles.get(step, "white")
        if _is_error_progress_message(message):
            err_console.print(f"[bold {style}]{step:>8}[/]  [red]{message}[/]", highlight=False)
        else:
            err_console.print(f"[bold {style}]{step:>8}[/]  {message}")

    return log


def _is_error_progress_message(message: str) -> bool:
    lowered = message.lower()
    return any(pattern in lowered for pattern in _ERROR_PROGRESS_PATTERNS)


_ERROR_PROGRESS_PATTERNS = (
    "llm timed out",
    "structured extraction failed",
    "structured contact extraction failed",
    "structured company facts extraction failed",
    "structured link selection failed",
    "repairing invalid json response",
    "response did not match list elements",
    "could not be parsed",
    "registry lookup failed",
    "registry failed",
    "first crawl page appears blocked",
    "credentials are missing",
    "no registry integration is available",
    "page text limit reached",
    "reduced page text after timeout",
    "removed email not found",
    "page could not be loaded",
    "could not be resolved",
    "no address fields config exists",
)


if __name__ == "__main__":
    app()
