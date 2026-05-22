from org_agent.registry import _build_zefix_profile_patch, _format_legal_address


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

    patch = _build_zefix_profile_patch(detail)

    assert patch.registration_id == "CHE-107.721.785"
    assert patch.official_company_name == "Zweifel Chips & Snacks AG"
    assert patch.purpose == "Production and sale of snacks."
    assert patch.legal_form == "AG"
    assert patch.country == "Switzerland"
    assert patch.region == "ZH"
    assert patch.legal_address == "Zweifel Chips & Snacks AG, Bahnstrasse 40, 8957 Spreitenbach"


def test_format_legal_address_handles_sparse_fields() -> None:
    address = {"poBox": "PO Box 12", "city": "Bern"}

    value = _format_legal_address(address)

    assert value == "PO Box 12, Bern"
