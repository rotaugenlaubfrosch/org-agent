from org_agent.graph import (
    NO_REGISTRY_LEGAL_ADDRESS_MESSAGE,
    _fill_registry_only_field_messages,
    _keep_requested_extraction_fields,
    _missing_profile_fields,
)
from org_agent.models import (
    AppConfig,
    EvidenceEntry,
    OrganizationProfile,
    PageExtraction,
    RegistryEndpointConfig,
    WebsiteOrganizationProfilePatch,
)


def test_missing_profile_fields_returns_only_empty_extractable_fields() -> None:
    profile = OrganizationProfile(
        name="Example Ltd",
        website="https://example.com",
        industry="Software",
        address="Example Street 1",
    )

    assert _missing_profile_fields(profile) == [
        "official_company_name",
        "registration_id",
        "legal_form",
        "description",
        "phone",
        "email",
        "country",
        "region",
    ]
    assert "legal_address" not in _missing_profile_fields(profile)


def test_page_extraction_schema_does_not_prompt_for_legal_address() -> None:
    schema = PageExtraction.model_json_schema()

    assert "legal_address" not in str(schema)


def test_keep_requested_extraction_fields_removes_unrequested_values() -> None:
    extraction = PageExtraction(
        profile_patch=WebsiteOrganizationProfilePatch(
            official_company_name="Example Ltd Official",
            industry="Software",
            address="Example Street 1",
            phone="+1 555 0100",
        ),
        evidence=[
            EvidenceEntry(
                field="official_company_name",
                value="Example Ltd Official",
                reasoning="Found on page.",
            ),
            EvidenceEntry(field="industry", value="Software", reasoning="Found on page."),
            EvidenceEntry(field="address", value="Example Street 1", reasoning="Found on page."),
            EvidenceEntry(field="phone", value="+1 555 0100", reasoning="Found on page."),
        ],
        missing_fields=["address", "phone", "email"],
        reasoning="Extracted fields.",
    )

    _keep_requested_extraction_fields(extraction, ["address", "email"])

    assert extraction.profile_patch.industry is None
    assert extraction.profile_patch.official_company_name is None
    assert extraction.profile_patch.address == "Example Street 1"
    assert extraction.profile_patch.phone is None
    assert [entry.field for entry in extraction.evidence] == ["address"]
    assert extraction.missing_fields == ["address", "email"]


def test_fill_registry_only_field_messages_sets_no_registry_legal_address_message() -> None:
    profile = OrganizationProfile(name="Example Ltd")

    _fill_registry_only_field_messages(profile, AppConfig())

    assert profile.legal_address == NO_REGISTRY_LEGAL_ADDRESS_MESSAGE
    assert profile.evidence[-1].field == "legal_address"
    assert profile.evidence[-1].source == "agent"


def test_fill_registry_only_field_messages_skips_when_registry_enabled() -> None:
    profile = OrganizationProfile(name="Example Ltd")
    config = AppConfig(
        registries=[
            RegistryEndpointConfig(
                name="zefix",
                provider="zefix",
                base_url="https://www.zefix.admin.ch/ZefixPublicREST",
                enabled=True,
            )
        ]
    )

    _fill_registry_only_field_messages(profile, config)

    assert profile.legal_address is None
