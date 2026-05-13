from org_agent.models import DerivationEntry, OrganizationProfile


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
        derivation=[
            DerivationEntry(
                field="website",
                value="https://example.com",
                source="search",
                reasoning="Selected as likely official website.",
                confidence=0.8,
            )
        ],
        confidence=0.7,
    )

    assert profile.name == "Example Ltd"
    assert profile.derivation[0].field == "website"
