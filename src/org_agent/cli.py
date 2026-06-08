from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from org_agent.api import lookup_organization
from org_agent.models import OrganizationProfile, profile_display_field_groups

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

Common lookup options:
  --website <url>  Required official website. Bare domains like example.com are accepted.
  --registry <id>  Enable optional registry provider (e.g. zefix)
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
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional registry YAML config with endpoints to query before extraction.",
    ),
    registries: list[str] = typer.Option(
        [],
        "--registry",
        help="Enable optional registry provider(s), e.g. zefix. Repeatable.",
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
            profile = lookup_organization(
                name=name,
                website=website,
                config=config,
                registries=registries,
            )
        else:
            err_console.rule("[bold cyan]org-agent trace")
            profile = lookup_organization(
                name=name,
                website=website,
                config=config,
                registries=registries,
                progress=_make_progress_logger(),
            )
            err_console.rule("[bold green]done")
    except Exception as exc:  # noqa: BLE001 - CLI should display concise failures
        err_console.print(f"[red]Lookup failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print(json.dumps(profile.model_dump(), ensure_ascii=False, indent=2))
        return

    _print_profile(profile)


def _print_profile(profile: OrganizationProfile) -> None:
    table = Table(title="Organization Profile", show_header=True, header_style="bold")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")

    normal_fields, registry_fields = profile_display_field_groups()
    for field in normal_fields:
        value = getattr(profile, field)
        table.add_row(field, "" if value is None else str(value))
    table.add_section()
    for field in registry_fields:
        value = getattr(profile, field)
        table.add_row(field, "" if value is None else str(value))

    console.print(table)

    lines = []
    for entry in profile.evidence:
        parts = []
        if entry.source:
            parts.append(f"Source: {entry.source}.")
        parts.append(entry.reasoning)
        lines.append(f"[bold]{entry.field}[/bold]: {' '.join(parts)}")
    console.print(Panel("\n".join(lines) or "No evidence entries returned.", title="Evidence"))


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
        err_console.print(f"[bold {style}]{step:>8}[/]  {message}")

    return log


if __name__ == "__main__":
    app()
