import asyncio

from org_agent.countries.ch.registry import (
    _build_profile_patch,
    _fallback_search_name,
    _format_legal_address,
    _pick_exact_match,
)
from org_agent.registry import (
    normalize_country_code,
    query_country_registry,
    registry_module_available,
    validate_country_code,
)


def test_normalize_country_code_accepts_iso_alpha_2_codes() -> None:
    assert normalize_country_code(" CH ") == "ch"
    assert normalize_country_code("li") == "li"
    assert normalize_country_code("DE") == "de"
    assert normalize_country_code(None) is None
    assert normalize_country_code(" ") is None


def test_validate_country_code_rejects_non_alpha_2_values() -> None:
    for value in ["Switzerland", "CHE", "xx", "c"]:
        try:
            validate_country_code(value)
        except ValueError as exc:
            assert "Expected a two-letter ISO 3166-1 alpha-2 country code" in str(exc)
        else:  # pragma: no cover - explicit failure branch for readability
            raise AssertionError(f"Expected {value!r} to be rejected")


def test_validate_country_code_accepts_valid_country_without_registry() -> None:
    validate_country_code("li")


def test_query_country_registry_skips_country_without_registry_integration() -> None:
    reports = []

    result = asyncio.run(
        query_country_registry(
            "li",
            "Example AG",
            20,
            progress=lambda scope, message: reports.append((scope, message)),
            scope="call_registries",
        )
    )

    assert result == []
    assert reports == [
        (
            "call_registries",
            "Skipped LI registry lookup because no registry integration is available.",
        )
    ]


def test_registry_module_available_returns_false_for_missing_country_package() -> None:
    assert registry_module_available("de") is False


def test_build_zefix_profile_patch_maps_expected_fields() -> None:
    detail = {
        "name": "Zweifel Chips & Snacks AG",
        "uid": "CHE-107.721.785",
        "purpose": "Production and sale of snacks.",
        "canton": "ZH",
        "legalSeat": "Spreitenbach",
        "legalForm": {
            "shortName": {"de": "AG"},
            "name": {"de": "Aktiengesellschaft"},
        },
        "address": {
            "organisation": "Zweifel Chips & Snacks AG",
            "street": "Bahnstrasse",
            "houseNumber": "40",
            "swissZipCode": "8957",
            "city": "Spreitenbach",
        },
    }

    patch = _build_profile_patch(detail)

    assert patch.registration_id == "CHE-107.721.785"
    assert patch.official_company_name == "Zweifel Chips & Snacks AG"
    assert patch.purpose == "Production and sale of snacks."
    assert patch.legal_structure == "AG"
    assert patch.country == "Switzerland"
    assert patch.region == "ZH"
    assert patch.legal_address == "Zweifel Chips & Snacks AG, Bahnstrasse 40, 8957 Spreitenbach"


def test_format_legal_address_handles_sparse_fields() -> None:
    address = {"poBox": "PO Box 12", "city": "Bern"}

    value = _format_legal_address(address)

    assert value == "PO Box 12, Bern"


def test_pick_exact_match_returns_exact_name() -> None:
    candidates = [{"name": "Zweifel AG Wil"}, {"name": "Zweifel AG"}]

    assert _pick_exact_match("Zweifel AG", candidates) == {"name": "Zweifel AG"}


def test_pick_exact_match_returns_none_without_exact_name() -> None:
    candidates = [{"name": "Zweifel AG Wil"}]

    assert _pick_exact_match("Zweifel AG", candidates) is None


def test_fallback_search_name_removes_legal_suffix() -> None:
    assert _fallback_search_name("Zweifel Chips AG") == "zweifel chips"
    assert _fallback_search_name("Example GmbH") == "example"
