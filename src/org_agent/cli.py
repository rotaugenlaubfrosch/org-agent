from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from org_agent.api import lookup_organization
from org_agent.models import OrganizationProfile

HELP_TEXT = """Enrich organization profiles from a name and optional website.

Required environment variables:
ORG_AGENT_LLM_PROVIDER=openai|anthropic|ollama
ORG_AGENT_LLM_MODEL=<model name>

Required for OpenAI/Anthropic:
ORG_AGENT_API_KEY=<provider API key>

Required for Ollama:
ORG_AGENT_OLLAMA_BASE_URL=<Ollama base URL>

Optional search environment variables:
ORG_AGENT_SEARCH_PROVIDER=tavily|brave|none
ORG_AGENT_SEARCH_API_KEY=<search API key for tavily/brave>

Optional runtime environment variables:
ORG_AGENT_REQUEST_TIMEOUT=<seconds, default 20>
ORG_AGENT_CRAWL_MAX_PAGES=<pages, default 6>
ORG_AGENT_CRAWL_MAX_DEPTH=<link depth, default 2>
ORG_AGENT_PLAYWRIGHT_HEADLESS=<true|false, default true>
ORG_AGENT_PLAYWRIGHT_SLOW_MO=<milliseconds, default 0>

Common lookup options:
  --website <url>  Use a known official website
  --json           Print raw JSON output
  --quiet          Suppress progress output

Examples:
  org-agent lookup "Example Ltd"
  org-agent lookup "Example Ltd" --website https://example.com
  org-agent lookup "Example Ltd" --website https://example.com --quiet
"""

app = typer.Typer(help=HELP_TEXT, add_completion=False)
console = Console()
err_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Enrich organization profiles from a name and optional website."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def lookup(
    name: str = typer.Argument(..., help="Organization name, assumed to be the correct name."),
    website: str | None = typer.Option(
        None,
        "--website",
        "-w",
        help="Known official website. If omitted, configure a search provider or registry config.",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional registry YAML config with endpoints to query before extraction.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON output."),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress the live trace of search, registry, website crawl, and extraction steps.",
    ),
) -> None:
    """Lookup and enrich an organization profile.

    Required environment configuration:
    ORG_AGENT_LLM_PROVIDER=openai|anthropic|ollama
    ORG_AGENT_LLM_MODEL=<model name>

    Required for OpenAI/Anthropic:
    ORG_AGENT_API_KEY=<provider API key>

    Required for Ollama:
    ORG_AGENT_OLLAMA_BASE_URL=<Ollama base URL>

    Optional search configuration:
    ORG_AGENT_SEARCH_PROVIDER=tavily|brave|none
    ORG_AGENT_SEARCH_API_KEY=<search API key for tavily/brave>

    Optional runtime configuration:
    ORG_AGENT_REQUEST_TIMEOUT=<seconds, default 20>
    ORG_AGENT_CRAWL_MAX_PAGES=<pages, default 6>
    ORG_AGENT_CRAWL_MAX_DEPTH=<link depth, default 2>
    ORG_AGENT_PLAYWRIGHT_HEADLESS=<true|false, default true>
    ORG_AGENT_PLAYWRIGHT_SLOW_MO=<milliseconds, default 0>

    Examples:
    org-agent lookup "Example Ltd"
    org-agent lookup "Example Ltd" --website https://example.com
    org-agent lookup "Example Ltd" --website https://example.com --quiet
    """
    try:
        if quiet:
            profile = lookup_organization(name=name, website=website, config=config)
        else:
            err_console.rule("[bold cyan]org-agent trace")
            profile = lookup_organization(
                name=name,
                website=website,
                config=config,
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

    for field in (
        "name",
        "website",
        "registration_id",
        "legal_form",
        "industry",
        "description",
        "address",
        "phone",
        "email",
        "country",
        "region",
    ):
        value = getattr(profile, field)
        table.add_row(field, "" if value is None else str(value))

    console.print(table)

    lines = []
    for entry in profile.evidence:
        source = f" Source: {entry.source}." if entry.source else ""
        value = f" Value: {entry.value}." if entry.value else ""
        lines.append(f"[bold]{entry.field}[/bold]:{value}{source} {entry.reasoning}")
    console.print(Panel("\n".join(lines) or "No evidence entries returned.", title="Evidence"))


def _make_progress_logger():
    styles = {
        "config": "magenta",
        "input": "cyan",
        "search": "blue",
        "registry": "yellow",
        "website": "green",
        "extract": "bright_magenta",
        "orchestrator": "bright_blue",
        "llm": "bright_black",
        "route": "bright_cyan",
    }

    def log(step: str, message: str) -> None:
        style = styles.get(step, "white")
        err_console.print(f"[bold {style}]{step:>8}[/]  {message}")

    return log


if __name__ == "__main__":
    app()
