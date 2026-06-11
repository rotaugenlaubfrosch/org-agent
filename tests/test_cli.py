from io import StringIO

from rich.console import Console

from org_agent import cli
from org_agent.models import LookupResult, OrganizationProfile


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


def test_print_lookup_result_shows_registry_status_without_registry_profile(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False, width=100))
    result = LookupResult(
        website_profile=OrganizationProfile(queried_name="Example Ltd"),
        registry_message="Registry lookup was not called because no country registry was selected.",
    )

    cli._print_lookup_result(result)

    rendered = output.getvalue()
    assert "Registry Profile" in rendered
    assert "status" in rendered
    assert "Registry lookup was not called because no country registry was selected." in rendered


def test_print_lookup_result_shows_sector_in_website_profile(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False, width=100))
    result = LookupResult(
        website_profile=OrganizationProfile(
            queried_name="Example Ltd",
            sector="Professional Services (tertiary)",
        ),
        registry_message="Registry lookup was not called because no country registry was selected.",
    )

    cli._print_lookup_result(result)

    rendered = output.getvalue()
    assert "sector" in rendered
    assert "Professional Services (tertiary)" in rendered


def test_print_lookup_result_shows_company_size_in_website_profile(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False, width=100))
    result = LookupResult(
        website_profile=OrganizationProfile(
            queried_name="Example Ltd",
            company_size=100,
        ),
        registry_message="Registry lookup was not called because no country registry was selected.",
    )

    cli._print_lookup_result(result)

    rendered = output.getvalue()
    assert "company_size" in rendered
    assert "100" in rendered


def test_print_lookup_result_shows_company_type_in_website_profile(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False, width=100))
    result = LookupResult(
        website_profile=OrganizationProfile(
            queried_name="Example Ltd",
            company_type="Research Institution",
        ),
        registry_message="Registry lookup was not called because no country registry was selected.",
    )

    cli._print_lookup_result(result)

    rendered = output.getvalue()
    assert "company_type" in rendered
    assert "Research Institution" in rendered


def test_print_lookup_result_shows_legal_structure_in_website_profile(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False, width=100))
    result = LookupResult(
        website_profile=OrganizationProfile(
            queried_name="Example Ltd",
            legal_structure="Limited Liability Company (GmbH / Sàrl)",
        ),
        registry_message="Registry lookup was not called because no country registry was selected.",
    )

    cli._print_lookup_result(result)

    rendered = output.getvalue()
    assert "legal_structure" in rendered
    assert "Limited Liability Company" in rendered
