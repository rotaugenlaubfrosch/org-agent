from langchain_core.output_parsers import PydanticOutputParser

from org_agent.graph import (
    NO_REGISTRY_LEGAL_ADDRESS_MESSAGE,
    NO_REGISTRY_OFFICIAL_COMPANY_NAME_MESSAGE,
    NO_REGISTRY_PURPOSE_MESSAGE,
    NO_REGISTRY_REGION_MESSAGE,
    NO_REGISTRY_REGISTRATION_ID_MESSAGE,
    _fill_registry_only_field_messages,
    _keep_requested_extraction_fields,
    _load_industries,
    _missing_crawl_fields,
    _merge_profile_patch,
    _missing_profile_fields,
    _parse_crawl_decision_result,
    _parse_industry_selection_result,
    _parse_structured_result,
    _truncate_progress_value,
    _validate_selected_industries,
)
from org_agent.models import (
    AppConfig,
    CrawlDecision,
    EvidenceEntry,
    IndustrySelection,
    OrganizationProfile,
    OrganizationProfilePatch,
    PageExtraction,
    RegistryEndpointConfig,
    WebsiteOrganizationProfilePatch,
)


def test_load_industries_reads_comma_separated_file(tmp_path) -> None:
    path = tmp_path / "industries.csv"
    path.write_text("AgroTech,Metallurgy\nAgroTech,Microactuators", encoding="utf-8")

    assert _load_industries(path) == ["AgroTech", "Metallurgy", "Microactuators"]


def test_validate_selected_industries_keeps_only_canonical_values() -> None:
    assert _validate_selected_industries(
        ["AgroTech", "Invented", "Metallurgy", "AgroTech"],
        ["AgroTech", "Metallurgy"],
        2,
    ) == ["AgroTech", "Metallurgy"]


def test_parse_industry_selection_accepts_schema_like_properties_wrapper() -> None:
    parser = PydanticOutputParser(pydantic_object=IndustrySelection)

    selection = _parse_industry_selection_result(
        {
            "properties": {
                "industries": ["Food Processing", "Consumer Goods"],
                "reasoning": "Selected based on snacks production.",
            },
            "required": ["reasoning"],
        },
        parser,
    )

    assert selection.industries == ["Food Processing", "Consumer Goods"]
    assert selection.reasoning == "Selected based on snacks production."


def test_parse_crawl_decision_accepts_schema_like_properties_wrapper() -> None:
    parser = PydanticOutputParser(pydantic_object=CrawlDecision)

    decision = _parse_crawl_decision_result(
        {
            "properties": {
                "is_complete": False,
                "selected_urls": ["https://example.com/about"],
                "reasoning": "The about page is likely useful.",
            },
        },
        parser,
    )

    assert decision.is_complete is False
    assert decision.selected_urls == ["https://example.com/about"]
    assert decision.reasoning == "The about page is likely useful."


def test_parse_page_extraction_accepts_schema_like_properties_wrapper() -> None:
    parser = PydanticOutputParser(pydantic_object=PageExtraction)

    extraction = _parse_structured_result(
        {
            "properties": {
                "profile_patch": {"address": "Example Street 1"},
                "evidence": [
                    {
                        "field": "address",
                        "value": "Example Street 1",
                        "reasoning": "Found on the contact page.",
                    }
                ],
                "missing_fields": ["email"],
                "reasoning": "Extracted address from page content.",
            },
        },
        PageExtraction,
        parser,
    )

    assert extraction.profile_patch.address == "Example Street 1"
    assert [entry.field for entry in extraction.evidence] == ["address"]
    assert extraction.missing_fields == ["email"]
    assert extraction.reasoning == "Extracted address from page content."


def test_missing_profile_fields_returns_only_empty_extractable_fields() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        website="https://example.com",
        industry="Software",
        address="Example Street 1",
    )

    assert _missing_profile_fields(profile) == [
        "legal_form",
        "phone",
        "email",
        "country",
    ]
    assert _missing_crawl_fields(profile) == [
        "legal_form",
        "phone",
        "description",
        "email",
        "country",
    ]
    assert "legal_address" not in _missing_profile_fields(profile)
    assert "official_company_name" not in _missing_profile_fields(profile)
    assert "registration_id" not in _missing_profile_fields(profile)
    assert "region" not in _missing_profile_fields(profile)


def test_page_extraction_schema_does_not_prompt_for_legal_address() -> None:
    schema = PageExtraction.model_json_schema()
    patch_properties = schema["$defs"]["WebsiteOrganizationProfilePatch"]["properties"]

    assert "description" not in patch_properties
    assert "industry" not in patch_properties
    assert "legal_address" not in str(schema)
    assert "official_company_name" not in str(schema)
    assert "queried_name" not in str(schema)
    assert "purpose" not in str(schema)
    assert "registration_id" not in str(schema)
    assert "region" not in str(schema)


def test_keep_requested_extraction_fields_removes_unrequested_values() -> None:
    extraction = PageExtraction(
        profile_patch=WebsiteOrganizationProfilePatch(
            legal_form="Limited company",
            address="Example Street 1",
            phone="+1 555 0100",
        ),
        evidence=[
            EvidenceEntry(
                field="official_company_name",
                value="Example Ltd Official",
                reasoning="Found on page.",
            ),
            EvidenceEntry(field="legal_form", value="Limited company", reasoning="Found on page."),
            EvidenceEntry(field="address", value="Example Street 1", reasoning="Found on page."),
            EvidenceEntry(field="phone", value="+1 555 0100", reasoning="Found on page."),
        ],
        missing_fields=["address", "phone", "email"],
        reasoning="Extracted fields.",
    )

    _keep_requested_extraction_fields(extraction, ["address", "email"])

    assert extraction.profile_patch.legal_form is None
    assert extraction.profile_patch.address == "Example Street 1"
    assert extraction.profile_patch.phone is None
    assert [entry.field for entry in extraction.evidence] == ["address"]
    assert extraction.missing_fields == ["address", "email"]


def test_fill_registry_only_field_messages_sets_no_registry_messages() -> None:
    profile = OrganizationProfile(queried_name="Example Ltd")

    _fill_registry_only_field_messages(profile, AppConfig())

    assert profile.legal_address == NO_REGISTRY_LEGAL_ADDRESS_MESSAGE
    assert profile.official_company_name == NO_REGISTRY_OFFICIAL_COMPANY_NAME_MESSAGE
    assert profile.purpose == NO_REGISTRY_PURPOSE_MESSAGE
    assert profile.registration_id == NO_REGISTRY_REGISTRATION_ID_MESSAGE
    assert profile.region == NO_REGISTRY_REGION_MESSAGE
    assert [entry.field for entry in profile.evidence] == [
        "official_company_name",
        "registration_id",
        "legal_address",
        "purpose",
        "region",
    ]
    assert all(entry.source == "agent" for entry in profile.evidence)


def test_fill_registry_only_field_messages_skips_when_registry_enabled() -> None:
    profile = OrganizationProfile(queried_name="Example Ltd")
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
    assert profile.official_company_name is None
    assert profile.purpose is None
    assert profile.registration_id is None
    assert profile.region is None


def test_merge_profile_patch_returns_only_newly_filled_fields() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        website="https://example.com",
        email="old@example.com",
    )
    patch = OrganizationProfilePatch(
        email="new@example.com",
        phone="+1 555 0100",
    )

    filled_fields = _merge_profile_patch(profile, patch)

    assert profile.email == "new@example.com"
    assert profile.phone == "+1 555 0100"
    assert filled_fields == [("phone", "+1 555 0100")]


def test_truncate_progress_value_appends_ellipsis_at_limit() -> None:
    value = "x" * 61

    assert _truncate_progress_value(value) == f"{'x' * 60}..."
