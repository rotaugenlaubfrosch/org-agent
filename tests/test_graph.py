from org_agent.graph import _keep_requested_extraction_fields, _missing_profile_fields
from org_agent.models import EvidenceEntry, OrganizationProfile, OrganizationProfilePatch, PageExtraction


def test_missing_profile_fields_returns_only_empty_extractable_fields() -> None:
    profile = OrganizationProfile(
        name="Example Ltd",
        website="https://example.com",
        industry="Software",
        address="Example Street 1",
    )

    assert _missing_profile_fields(profile) == [
        "registration_id",
        "legal_form",
        "description",
        "phone",
        "email",
        "country",
        "region",
    ]


def test_keep_requested_extraction_fields_removes_unrequested_values() -> None:
    extraction = PageExtraction(
        profile_patch=OrganizationProfilePatch(
            industry="Software",
            address="Example Street 1",
            phone="+1 555 0100",
        ),
        evidence=[
            EvidenceEntry(field="industry", value="Software", reasoning="Found on page."),
            EvidenceEntry(field="address", value="Example Street 1", reasoning="Found on page."),
            EvidenceEntry(field="phone", value="+1 555 0100", reasoning="Found on page."),
        ],
        missing_fields=["address", "phone", "email"],
        reasoning="Extracted fields.",
    )

    _keep_requested_extraction_fields(extraction, ["address", "email"])

    assert extraction.profile_patch.industry is None
    assert extraction.profile_patch.address == "Example Street 1"
    assert extraction.profile_patch.phone is None
    assert [entry.field for entry in extraction.evidence] == ["address"]
    assert extraction.missing_fields == ["address", "email"]
