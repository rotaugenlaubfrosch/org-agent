from io import StringIO

from rich.console import Console
import json

from typer.testing import CliRunner

from org_agent import cli
from org_agent.models import OrganizationProfile


def test_print_profile_table_indents_address_fields(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False, width=100))
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        address="Example Street 1, 8000 Zürich",
        address_fields={"address_street": "Example Street", "address_city": "Zürich"},
    )

    cli._print_profile_table("Website Profile", profile, ("address",))

    rendered = output.getvalue()
    assert "address" in rendered
    assert "  address_street" in rendered
    assert "Example Street" in rendered
    assert "  address_city" in rendered
    assert "Zürich" in rendered
