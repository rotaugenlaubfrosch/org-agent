import json

from typer.testing import CliRunner

from org_agent import cli
from org_agent.models import OrganizationProfile


runner = CliRunner()


def test_cli_without_arguments_shows_help() -> None:
    result = runner.invoke(cli.app, [])

    assert result.exit_code == 0
    assert "Enrich organization profiles" in result.output
    assert "--website" in result.output


def test_cli_runs_lookup_from_root_command(monkeypatch) -> None:
    captured = {}

    def fake_lookup_organization(
        name: str,
        website: str | None = None,
        config: str | None = None,
        registries: list[str] | None = None,
        progress=None,
    ) -> OrganizationProfile:
        captured["name"] = name
        captured["website"] = website
        captured["config"] = config
        captured["registries"] = registries
        captured["progress"] = progress
        return OrganizationProfile(queried_name=name, website="https://example.com/")

    monkeypatch.setattr(cli, "lookup_organization", fake_lookup_organization)

    result = runner.invoke(
        cli.app,
        [
            "Example Ltd",
            "--website",
            "example.com",
            "--config",
            "registries.yml",
            "--registry",
            "zefix",
            "--json",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "name": "Example Ltd",
        "website": "example.com",
        "config": "registries.yml",
        "registries": ["zefix"],
        "progress": None,
    }
    assert json.loads(result.output)["queried_name"] == "Example Ltd"


def test_cli_requires_website_for_root_lookup() -> None:
    result = runner.invoke(cli.app, ["Example Ltd"])

    assert result.exit_code == 1
    assert "--website is required" in result.output


def test_cli_lookup_subcommand_is_not_available() -> None:
    result = runner.invoke(cli.app, ["lookup", "Example Ltd", "--website", "example.com"])

    assert result.exit_code == 2
