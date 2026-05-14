from org_agent.models import EvidenceEntry, OrganizationProfile


def test_organization_profile_accepts_expected_fields() -> None:
    profile = OrganizationProfile(
        name="Example Ltd",
        website="https://example.com",
        registration_id="12345",
        legal_form="Limited company",
        industry="Software",
        description="The organization develops software products.",
        address="Example Street 1",
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

    assert profile.name == "Example Ltd"
    assert profile.evidence[0].field == "website"
