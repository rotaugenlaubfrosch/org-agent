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
"""

app = typer.Typer(help=HELP_TEXT)
console = Console()
err_console = Console(stderr=True)


@app.callback()
def main() -> None:
    """Enrich organization profiles from a name and optional website.

    Use `org-agent lookup --help` to show lookup arguments.
    Typer's `--install-completion` installs shell autocompletion.
    Typer's `--show-completion` prints the completion script instead of installing it.
    """


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
        "confidence",
    ):
        value = getattr(profile, field)
        table.add_row(field, "" if value is None else str(value))

    console.print(table)

    if profile.derivation:
        lines = []
        for entry in profile.derivation:
            source = f" Source: {entry.source}." if entry.source else ""
            value = f" Value: {entry.value}." if entry.value else ""
            lines.append(
                f"[bold]{entry.field}[/bold] ({entry.confidence:.2f}):"
                f"{value}{source} {entry.reasoning}"
            )
        console.print(Panel("\n".join(lines), title="Derivation"))


def _make_progress_logger():
    styles = {
        "config": "magenta",
        "input": "cyan",
        "search": "blue",
        "registry": "yellow",
        "website": "green",
        "extract": "bright_magenta",
    }

    def log(step: str, message: str) -> None:
        style = styles.get(step, "white")
        err_console.print(f"[bold {style}]{step:>8}[/]  {message}")

    return log


if __name__ == "__main__":
    app()
