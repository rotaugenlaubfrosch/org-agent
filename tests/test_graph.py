import asyncio

from langchain_core.output_parsers import PydanticOutputParser

from org_agent.graph import (
    NO_REGISTRY_LEGAL_ADDRESS_MESSAGE,
    NO_REGISTRY_OFFICIAL_COMPANY_NAME_MESSAGE,
    NO_REGISTRY_PURPOSE_MESSAGE,
    NO_REGISTRY_REGION_MESSAGE,
    NO_REGISTRY_REGISTRATION_ID_MESSAGE,
    _fill_registry_only_field_messages,
    _extract_page_info,
    _keep_requested_extraction_fields,
    _load_industries,
    _missing_crawl_fields,
    _build_registry_profile,
    _address_fields_config_path,
    _fragment_profile_address,
    _load_address_fields_config,
    _merge_profile_patch,
    _merge_profile_patch_missing_only,
    _missing_profile_fields,
    _normalize_country,
    _normalize_profile_country,
    _parse_crawl_decision_result,
    _parse_industry_selection_result,
    _parse_structured_result,
    _queried_country_value,
    _missing_registry_credentials_message,
    _registry_result_message,
    _should_continue_crawl,
    _truncate_progress_value,
    _validate_profile_email,
    _validate_selected_industries,
    _validated_address_fields,
)
from org_agent.models import (
    AgentState,
    CrawlDecision,
    CrawlTarget,
    EvidenceEntry,
    IndustrySelection,
    LookupInput,
    OrganizationProfile,
    OrganizationProfilePatch,
    PageAnalysis,
    PageExtraction,
    RegistryResult,
    WebsiteOrganizationProfilePatch,
    WebsitePage,
)


class _CrawlSettings:
    crawl_max_pages = 6


class _CapturingStructuredLLM:
    def __init__(self) -> None:
        self.messages = None

    def with_structured_output(self, _model_type):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return PageExtraction(reasoning="No values on page.")


class _AddressFragmentationLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages

        class _Response:
            def __init__(self, content: str) -> None:
                self.content = content

        return _Response(self.content)


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


def test_load_address_fields_config_requires_prompt_and_validation(tmp_path) -> None:
    path = tmp_path / "address_fields.json"
    path.write_text(
        '{"city": {"prompt": "Extract the city.", "validation": true}}',
        encoding="utf-8",
    )

    assert _load_address_fields_config(path) == {
        "city": {"prompt": "Extract the city.", "validation": True}
    }


def test_load_address_fields_config_rejects_extra_keys(tmp_path) -> None:
    path = tmp_path / "address_fields.json"
    path.write_text(
        '{"city": {"prompt": "Extract the city.", "validation": true, "example": "Zürich"}}',
        encoding="utf-8",
    )

    try:
        _load_address_fields_config(path)
    except ValueError as exc:
        assert "exactly prompt and validation" in str(exc)
    else:
        raise AssertionError("Expected malformed address fields config to be rejected.")


def test_address_fields_config_path_resolves_explicit_country_and_profile_country() -> None:
    assert _address_fields_config_path("ch", None) is not None
    assert _address_fields_config_path(None, "Switzerland") is not None


def test_validated_address_fields_prefixes_and_filters_values() -> None:
    config = {
        "street": {"prompt": "Extract street.", "validation": True},
        "city": {"prompt": "Extract city.", "validation": True},
        "note": {"prompt": "Extract note.", "validation": False},
    }
    extracted = {
        "street": "Rämistrasse",
        "address_city": "Invented City",
        "address_note": "Generated note",
        "unknown": "Ignored",
    }

    assert _validated_address_fields(
        "ETH Zürich, Rämistrasse 101, CH-8092 Zürich",
        extracted,
        config,
    ) == {
        "address_street": "Rämistrasse",
        "address_note": "Generated note",
    }


def test_fragment_profile_address_keeps_address_and_reports_llm_output() -> None:
    profile = OrganizationProfile(
        queried_name="ETH Zürich",
        address="ETH Zürich, Rämistrasse 101, CH-8092 Zürich",
    )
    llm = _AddressFragmentationLLM(
        '{"address_organization": "ETH Zürich", "address_street": "Rämistrasse", '
        '"address_number": "101", "address_postal_code": "8092", "address_city": "Zürich"}'
    )
    reports = []

    asyncio.run(
        _fragment_profile_address(
            llm,
            profile,
            "ch",
            lambda step, message: reports.append((step, message)),
            "validate_profile",
        )
    )

    assert profile.address == "ETH Zürich, Rämistrasse 101, CH-8092 Zürich"
    assert profile.address_fields == {
        "address_organization": "ETH Zürich",
        "address_street": "Rämistrasse",
        "address_number": "101",
        "address_postal_code": "8092",
        "address_city": "Zürich",
    }
    assert llm.messages is not None
    assert "address_city" in llm.messages[-1].content
    assert reports[0] == (
        "validate_profile",
        "Address fields country source: explicit --country=ch -> ch.",
    )
    assert reports[1][0] == "validate_profile"
    assert "Therefore, using address fields config:" in reports[1][1]
    assert "Address fragmentation LLM output:" in reports[-1][1]


def test_fragment_profile_address_reports_profile_country_fallback() -> None:
    profile = OrganizationProfile(
        queried_name="ETH Zürich",
        address="ETH Zürich, Rämistrasse 101, CH-8092 Zürich",
        country="Switzerland",
    )
    llm = _AddressFragmentationLLM('{"address_city": "Zürich"}')
    reports = []

    asyncio.run(
        _fragment_profile_address(
            llm,
            profile,
            None,
            lambda step, message: reports.append((step, message)),
            "validate_profile",
        )
    )

    assert reports[0] == (
        "validate_profile",
        "Address fields country source: extracted profile.country=Switzerland -> ch.",
    )
    assert "Therefore, using address fields config:" in reports[1][1]
    assert profile.address_fields == {"address_city": "Zürich"}


def test_fragment_profile_address_reports_no_country_skip() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        address="Example Street 1",
    )
    llm = _AddressFragmentationLLM('{"address_city": "Zürich"}')
    reports = []

    asyncio.run(
        _fragment_profile_address(
            llm,
            profile,
            None,
            lambda step, message: reports.append((step, message)),
            "validate_profile",
        )
    )

    assert reports == [
        (
            "validate_profile",
            "Skipped address fragmentation because no country was provided or extracted.",
        )
    ]
    assert llm.messages is None
    assert profile.address_fields == {}


def test_fragment_profile_address_reports_no_config_skip() -> None:
    profile = OrganizationProfile(
        queried_name="Example GmbH",
        address="Example Street 1, 10115 Berlin",
    )
    llm = _AddressFragmentationLLM('{"address_city": "Berlin"}')
    reports = []

    asyncio.run(
        _fragment_profile_address(
            llm,
            profile,
            "de",
            lambda step, message: reports.append((step, message)),
            "validate_profile",
        )
    )

    assert reports == [
        (
            "validate_profile",
            "Skipped address fragmentation because explicit --country=de resolved to de, "
            "but no address fields config exists for country: de.",
        )
    ]
    assert llm.messages is None
    assert profile.address_fields == {}


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


def test_extract_page_info_prompt_excludes_registry_results() -> None:
    llm = _CapturingStructuredLLM()
    parser = PydanticOutputParser(pydantic_object=PageExtraction)

    asyncio.run(
        _extract_page_info(
            llm,
            parser,
            "Example Ltd",
            "https://example.com",
            ["address", "phone"],
            WebsitePage(url="https://example.com", text="Contact page text."),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[-1].content

    assert "Current page:" in prompt
    assert "Registry results:" not in prompt


def test_missing_profile_fields_returns_only_empty_extractable_fields() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        queried_website="https://example.com",
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

    _fill_registry_only_field_messages(profile)

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


def test_fill_registry_only_field_messages_skips_when_registry_results_exist() -> None:
    profile = OrganizationProfile(queried_name="Example Ltd")

    _fill_registry_only_field_messages(profile, has_registry_results=True)

    assert profile.legal_address is None
    assert profile.official_company_name is None
    assert profile.purpose is None
    assert profile.registration_id is None
    assert profile.region is None


def test_merge_profile_patch_returns_only_newly_filled_fields() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        queried_website="https://example.com",
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


def test_merge_profile_patch_missing_only_does_not_overwrite_website_fields() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        queried_website="https://example.com",
        address="Website Street 1",
        phone="+1 555 0100",
        email="info@example.com",
        country="Website Country",
    )
    patch = OrganizationProfilePatch(
        official_company_name="Example Ltd Official",
        registration_id="CHE-123",
        legal_form="AG",
        address="Registry Street 2",
        phone="+1 555 9999",
        email="registry@example.com",
        country="Switzerland",
        legal_address="Registry Street 2, 8000 Zurich",
    )

    filled_fields = _merge_profile_patch_missing_only(profile, patch)

    assert profile.official_company_name == "Example Ltd Official"
    assert profile.registration_id == "CHE-123"
    assert profile.legal_form == "AG"
    assert profile.legal_address == "Registry Street 2, 8000 Zurich"
    assert profile.address == "Website Street 1"
    assert profile.phone == "+1 555 0100"
    assert profile.email == "info@example.com"
    assert profile.country == "Website Country"
    assert filled_fields == [
        ("official_company_name", "Example Ltd Official"),
        ("registration_id", "CHE-123"),
        ("legal_form", "AG"),
        ("legal_address", "Registry Street 2, 8000 Zurich"),
    ]


def test_build_registry_profile_keeps_registry_fields_separate() -> None:
    registry_result = RegistryResult(
        registry="ch",
        url="https://registry.example/detail",
        status_code=200,
        content="{}",
        profile_patch=OrganizationProfilePatch(
            official_company_name="Example Ltd Official",
            phone="+1 555 9999",
            country="Switzerland",
        ),
        evidence=[
            EvidenceEntry(
                field="official_company_name",
                value="Example Ltd Official",
                source="https://registry.example/detail",
                reasoning="Extracted from registry.",
            ),
            EvidenceEntry(
                field="phone",
                value="+1 555 9999",
                source="https://registry.example/detail",
                reasoning="Extracted from registry.",
            ),
        ],
    )

    profile = _build_registry_profile("Example Ltd", "ch", [registry_result])

    assert profile is not None
    assert profile.queried_name == "Example Ltd"
    assert profile.queried_country == "CH"
    assert profile.official_company_name == "Example Ltd Official"
    assert profile.country == "Switzerland"
    assert profile.phone == "+1 555 9999"
    assert [entry.field for entry in profile.evidence] == [
        "official_company_name",
        "phone",
    ]


def test_build_registry_profile_returns_none_without_registry_values() -> None:
    registry_result = RegistryResult(
        registry="ch",
        url="https://registry.example/detail",
        status_code=0,
        content="Registry query failed.",
    )

    assert _build_registry_profile("Example Ltd", None, [registry_result]) is None


def test_queried_country_value_uses_uppercase_code_or_not_specified() -> None:
    assert _queried_country_value("ch") == "CH"
    assert _queried_country_value(" CH ") == "CH"
    assert _queried_country_value("CHE") == "CH"
    assert _queried_country_value("Switzerland") == "CH"
    assert _queried_country_value(None) == "not specified"
    assert _queried_country_value(" ") == "not specified"


def test_registry_result_message_explains_no_country_skip() -> None:
    assert (
        _registry_result_message(None, [], None)
        == "Registry lookup was not called because no country registry was selected."
    )


def test_missing_registry_credentials_message_uses_uppercase_country_code() -> None:
    assert (
        _missing_registry_credentials_message("ch")
        == "Registry lookup was not called because CH registry credentials are missing."
    )


def test_registry_result_message_explains_no_profile_fields() -> None:
    registry_result = RegistryResult(
        registry="ch",
        url="https://registry.example/detail",
        status_code=200,
        content="{}",
    )

    assert (
        _registry_result_message("ch", [registry_result], None)
        == "Registry lookup completed but did not produce registry profile fields."
    )


def test_registry_result_message_explains_registry_failure() -> None:
    registry_result = RegistryResult(
        registry="ch",
        url="https://registry.example/detail",
        status_code=0,
        content="Registry query failed.",
    )

    assert (
        _registry_result_message("ch", [registry_result], None)
        == "Registry lookup failed; see registry trace output."
    )


def test_registry_result_message_is_empty_when_profile_exists() -> None:
    assert (
        _registry_result_message(
            "ch",
            [],
            OrganizationProfile(queried_name="Example Ltd", queried_country="CH"),
        )
        is None
    )


def test_normalize_country_resolves_iso_codes_and_names() -> None:
    assert _normalize_country(" CH ") == "Switzerland"
    assert _normalize_country("CHE") == "Switzerland"
    assert _normalize_country("switzerland") == "Switzerland"
    assert _normalize_country("Swiss") == "Switzerland"


def test_normalize_country_keeps_unknown_values() -> None:
    assert _normalize_country("Atlantis") == "Atlantis"
    assert _normalize_country(" ") is None
    assert _normalize_country(None) is None


def test_normalize_profile_country_updates_profile_and_evidence() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        country="CHE",
        evidence=[
            EvidenceEntry(field="country", value="CH", reasoning="Found on page."),
            EvidenceEntry(field="email", value="info@example.com", reasoning="Found on page."),
        ],
    )

    _normalize_profile_country(profile)

    assert profile.country == "Switzerland"
    assert profile.evidence[0].value == "Switzerland"
    assert profile.evidence[1].value == "info@example.com"


def test_validate_profile_email_keeps_email_present_in_website_pages() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        email=" info@example.com ",
        evidence=[EvidenceEntry(field="email", value="info@example.com", reasoning="Found on page.")],
    )
    pages = [WebsitePage(url="https://example.com", text="Contact us at info@example.com.")]

    _validate_profile_email(profile, pages)

    assert profile.email == "info@example.com"
    assert [entry.field for entry in profile.evidence] == ["email"]


def test_validate_profile_email_matches_case_insensitively() -> None:
    profile = OrganizationProfile(queried_name="Example Ltd", email="Info@Example.com")
    pages = [WebsitePage(url="https://example.com", text="Contact: info@example.com")]

    _validate_profile_email(profile, pages)

    assert profile.email == "Info@Example.com"


def test_validate_profile_email_removes_email_absent_from_website_pages() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        email="info@example.com",
        evidence=[
            EvidenceEntry(field="email", value="info@example.com", reasoning="Found on page."),
            EvidenceEntry(field="country", value="Switzerland", reasoning="Found on page."),
        ],
    )
    pages = [WebsitePage(url="https://example.com", text="No email address here.")]

    _validate_profile_email(profile, pages)

    assert profile.email is None
    assert [entry.field for entry in profile.evidence] == ["country"]


def test_validate_profile_email_ignores_empty_website_pages() -> None:
    profile = OrganizationProfile(queried_name="Example Ltd", email="info@example.com")

    _validate_profile_email(profile, [])

    assert profile.email == "info@example.com"


def test_should_continue_crawl_visits_newly_queued_selected_link() -> None:
    state = AgentState(
        input=LookupInput(name="Example Ltd", website="https://example.com"),
        profile=OrganizationProfile(
            queried_name="Example Ltd",
            queried_website="https://example.com",
            description="Example Ltd does things.",
            industry="Food",
            phone="+1 555 0100",
            evidence=[EvidenceEntry(field="phone", value="+1 555 0100", reasoning="Found.")],
        ),
        website_pages=[WebsitePage(url="https://example.com", text="Page text.")],
        pending_urls=[CrawlTarget(url="https://example.com/contact", depth=1, node_id=1)],
        page_analysis=PageAnalysis(
            reasoning="The profile is complete, but a selected link was queued.",
            is_complete=True,
        ),
    )

    assert _should_continue_crawl(state, _CrawlSettings()) is False
    assert _should_continue_crawl(state, _CrawlSettings(), queued_selected_link=True) is True


def test_truncate_progress_value_appends_ellipsis_at_limit() -> None:
    value = "x" * 61

    assert _truncate_progress_value(value) == f"{'x' * 60}..."
