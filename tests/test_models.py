from org_agent.models import EvidenceEntry, OrganizationProfile


def test_organization_profile_accepts_expected_fields() -> None:
    profile = OrganizationProfile(
        name="Example Ltd",
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

    assert profile.name == "Example Ltd"
    assert profile.official_company_name == "Example Ltd Official"
    assert profile.purpose == "Develop and distribute software."
    assert profile.legal_address == "Registry Street 10"
    assert profile.evidence[0].field == "website"
