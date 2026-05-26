from org_agent.models import (
    PROFILE_DISPLAY_FIELDS,
    REGISTRY_ONLY_PROFILE_FIELDS,
    EvidenceEntry,
    OrganizationProfile,
    profile_display_field_groups,
)


def test_organization_profile_accepts_expected_fields() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        official_company_name="Example Ltd Official",
        website="https://example.com",
        registration_id="12345",
        legal_form="Limited company",
        industry="Software",
        description="The organization develops software products.",
        purpose="Develop and distribute software.",
        address="Example Street 1",
        legal_address="Registry Street 10",
        phone="+1 555 0100",
        email="info@example.com",
        country="United States",
        region="California",
        evidence=[
            EvidenceEntry(
                field="website",
                value="https://example.com",
                source="search",
                reasoning="Selected as likely official website.",
            )
        ],
    )

    assert profile.queried_name == "Example Ltd"
    assert profile.official_company_name == "Example Ltd Official"
    assert profile.purpose == "Develop and distribute software."
    assert profile.legal_address == "Registry Street 10"
    assert profile.evidence[0].field == "website"


def test_profile_display_fields_match_profile_scalar_fields() -> None:
    assert PROFILE_DISPLAY_FIELDS == (
        "queried_name",
        "official_company_name",
        "website",
        "registration_id",
        "legal_form",
        "industry",
        "description",
        "purpose",
        "address",
        "legal_address",
        "phone",
        "email",
        "country",
        "region",
    )
    assert "evidence" not in PROFILE_DISPLAY_FIELDS
    assert set(PROFILE_DISPLAY_FIELDS).issubset(OrganizationProfile.model_fields)


def test_profile_display_field_groups_separate_normal_and_registry_fields() -> None:
    normal_fields, registry_fields = profile_display_field_groups()

    assert normal_fields == (
        "queried_name",
        "website",
        "legal_form",
        "industry",
        "description",
        "address",
        "phone",
        "email",
        "country",
    )
    assert registry_fields == REGISTRY_ONLY_PROFILE_FIELDS
    assert registry_fields == (
        "official_company_name",
        "registration_id",
        "purpose",
        "legal_address",
        "region",
    )
    assert set(normal_fields).isdisjoint(registry_fields)
    assert normal_fields + registry_fields != PROFILE_DISPLAY_FIELDS
    assert set(normal_fields + registry_fields) == set(PROFILE_DISPLAY_FIELDS)
