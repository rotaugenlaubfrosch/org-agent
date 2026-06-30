import asyncio

from langchain_core.output_parsers import PydanticOutputParser

from org_agent.graph import (
    LLM_LIST_SELECTION_MISMATCH,
    NO_REGISTRY_LEGAL_ADDRESS_MESSAGE,
    NO_REGISTRY_OFFICIAL_COMPANY_NAME_MESSAGE,
    NO_REGISTRY_PURPOSE_MESSAGE,
    NO_REGISTRY_REGION_MESSAGE,
    NO_REGISTRY_REGISTRATION_ID_MESSAGE,
    _fill_registry_only_field_messages,
    _country_focus,
    _extract_company_type,
    _extract_company_facts,
    _extract_contact_info,
    _extract_description,
    _extract_industries,
    _country_from_address,
    _derive_profile_country_from_address,
    _extract_legal_structure,
    _extract_page_info,
    _extract_sector,
    _first_page_blocked_reason,
    _keep_requested_extraction_fields,
    _load_industries,
    _missing_crawl_fields,
    _build_registry_profile,
    _address_fields_config_path,
    _fragment_profile_address,
    _has_minimum_profile,
    _load_address_fields_config,
    _merge_profile_patch,
    _merge_profile_patch_missing_only,
    _missing_profile_fields,
    _normalize_country,
    _normalize_profile_country,
    _parse_crawl_decision_result,
    _parse_multiple_candidate_response,
    _parse_single_candidate_response,
    _parse_structured_result,
    _queried_country_value,
    _reduced_timeout_retry_text,
    run_lookup,
    _missing_registry_credentials_message,
    _missing_registry_integration_message,
    _registry_result_message,
    _select_multiple_candidates,
    _select_next_links,
    _select_single_candidate,
    _should_continue_after_skipped_page,
    _should_continue_crawl,
    _truncate_progress_value,
    _validate_profile_email,
    _validate_selected_company_type,
    _validate_selected_industries,
    _validate_selected_legal_structure,
    _validate_selected_sector,
    _validated_address_fields,
)
from org_agent.models import (
    AgentState,
    CompanyFactsExtraction,
    CrawlDecision,
    CrawlTarget,
    EvidenceEntry,
    LookupInput,
    OrganizationProfile,
    OrganizationProfilePatch,
    PageAnalysis,
    ContactPageExtraction,
    PageExtraction,
    RegistryResult,
    WebsiteLink,
    WebsiteOrganizationProfilePatch,
    WebsitePage,
)
from org_agent.settings import Settings


class _CrawlSettings:
    crawl_max_pages = 6
    crawl_max_depth = 2


class _CapturingStructuredLLM:
    def __init__(self) -> None:
        self.messages = None
        self.model_type = PageExtraction

    def with_structured_output(self, model_type):
        self.model_type = model_type
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.model_type(reasoning="No values on page.")


class _CapturingCrawlDecisionLLM:
    def __init__(self) -> None:
        self.messages = None

    def with_structured_output(self, model_type):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return CrawlDecision(is_complete=False, reasoning="Selected no links.")


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


class _SequentialTextLLM:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        content = self.contents.pop(0)

        class _Response:
            def __init__(self, content: str) -> None:
                self.content = content

        return _Response(content)


class _SlowLLM:
    async def ainvoke(self, messages):
        await asyncio.sleep(1)
        return messages


class _TimeoutThenTextLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        if len(self.messages) == 1:
            await asyncio.sleep(1)

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


def test_validate_selected_sector_keeps_only_canonical_value() -> None:
    assert _validate_selected_sector(
        "Manufacturer (secondary)",
        ["Producer (primary)", "Manufacturer (secondary)"],
    ) == "Manufacturer (secondary)"
    assert _validate_selected_sector(
        "Invented",
        ["Producer (primary)", "Manufacturer (secondary)"],
    ) is None


def test_validate_selected_company_type_keeps_only_canonical_value() -> None:
    assert _validate_selected_company_type(
        "Academic Institution",
        ["Commercial Enterprise", "Academic Institution"],
    ) == "Academic Institution"
    assert _validate_selected_company_type(
        "Invented",
        ["Commercial Enterprise", "Academic Institution"],
    ) is None


def test_validate_selected_legal_structure_keeps_only_canonical_value() -> None:
    assert _validate_selected_legal_structure(
        "Limited Liability Company (GmbH / Sàrl)",
        ["Company Limited by Shares (AG / SA)", "Limited Liability Company (GmbH / Sàrl)"],
    ) == "Limited Liability Company (GmbH / Sàrl)"
    assert _validate_selected_legal_structure(
        "Invented",
        ["Company Limited by Shares (AG / SA)", "Limited Liability Company (GmbH / Sàrl)"],
    ) is None


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
    assert _address_fields_config_path("de", None) is not None


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
        address_country="Switzerland",
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
        "Address fields country source: extracted profile.address_country=Switzerland -> ch.",
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


def test_fragment_profile_address_uses_default_config_when_country_config_missing() -> None:
    profile = OrganizationProfile(
        queried_name="Example GmbH",
        address="Example Street 1, 10115 Berlin",
    )
    llm = _AddressFragmentationLLM(
        '{"address_street": "Example Street", "address_number": "1", '
        '"address_postal_code": "10115", "address_city": "Berlin"}'
    )
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

    assert reports[0] == (
        "validate_profile",
        "Address fields country source: explicit --country=de -> de; using default address "
        "fields config because no country-specific config exists: "
        "src/org_agent/countries/_DEFAULT/address_fields.json.",
    )
    assert "Therefore, using address fields config:" in reports[1][1]
    assert "_DEFAULT/address_fields.json" in reports[1][1]
    assert profile.address_fields == {
        "address_street": "Example Street",
        "address_number": "1",
        "address_postal_code": "10115",
        "address_city": "Berlin",
    }


def test_parse_single_candidate_response_accepts_exact_value_or_none() -> None:
    candidates = ["Company Limited by Shares (AG / SA)", "Limited Liability Company (GmbH / Sàrl)"]

    assert (
        _parse_single_candidate_response("Company Limited by Shares (AG / SA)", candidates)
        == "Company Limited by Shares (AG / SA)"
    )
    assert _parse_single_candidate_response("NONE", candidates) is None
    assert _parse_single_candidate_response('{"legal_structure": "AG"}', candidates) == (
        LLM_LIST_SELECTION_MISMATCH
    )


def test_parse_multiple_candidate_response_accepts_exact_lines_or_none() -> None:
    candidates = ["Food Processing", "Consumer Goods"]

    assert _parse_multiple_candidate_response(
        "Food Processing\nConsumer Goods",
        candidates,
        2,
    ) == ["Food Processing", "Consumer Goods"]
    assert _parse_multiple_candidate_response("NONE", candidates, 2) == []
    assert _parse_multiple_candidate_response("Food Processing\nInvented", candidates, 2) == (
        LLM_LIST_SELECTION_MISMATCH
    )


def test_reduced_timeout_retry_text_keeps_first_and_last_quarters() -> None:
    text = "a" * 50 + "b" * 100 + "c" * 50

    reduced = _reduced_timeout_retry_text(text)

    assert reduced.startswith("a" * 50)
    assert reduced.endswith("c" * 50)
    assert "middle 50% removed after LLM timeout" in reduced
    assert "b" not in reduced


def test_reduced_timeout_retry_text_keeps_short_text_unchanged() -> None:
    assert _reduced_timeout_retry_text("short text") == "short text"


def test_extract_description_timeout_returns_empty_string(monkeypatch) -> None:
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr("org_agent.graph.LLM_CALL_TIMEOUT_SECONDS", 0.01)

    description = asyncio.run(
        _extract_description(
            _SlowLLM(),
            "Describe [Account Name].",
            "Example Ltd",
            WebsitePage(url="https://example.com", text="About page text."),
            lambda scope, message: messages.append((scope, message)),
            "analyze_page",
        )
    )

    assert description == ""
    assert messages[-1] == ("analyze_page", "LLM timed out after 20 seconds: description.")


def test_extract_description_retries_timeout_with_reduced_page_text(monkeypatch) -> None:
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr("org_agent.graph.LLM_CALL_TIMEOUT_SECONDS", 0.01)
    llm = _TimeoutThenTextLLM("Reduced description")
    page_text = "a" * 100 + "MIDDLE" * 100 + "z" * 100

    description = asyncio.run(
        _extract_description(
            llm,
            "Describe [Account Name].",
            "Example Ltd",
            WebsitePage(url="https://example.com", text=page_text),
            lambda scope, message: messages.append((scope, message)),
            "analyze_page",
        )
    )

    retry_text = llm.messages[1][-1].content
    assert description == "Reduced description"
    assert len(llm.messages) == 2
    assert retry_text.startswith("a")
    assert retry_text.endswith("z")
    assert "middle 50% removed after LLM timeout" in retry_text
    assert (
        "analyze_page",
        "Retrying description with reduced page text after timeout: kept first 25% and last 25%.",
    ) in messages


def test_select_single_candidate_timeout_returns_none(monkeypatch) -> None:
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr("org_agent.graph.LLM_CALL_TIMEOUT_SECONDS", 0.01)

    selected = asyncio.run(
        _select_single_candidate(
            _SlowLLM(),
            "Choose one.",
            ["Commercial Enterprise"],
            "company_type",
            lambda scope, message: messages.append((scope, message)),
            "analyze_page",
        )
    )

    assert selected is None
    assert messages[-1] == (
        "analyze_page",
        "LLM timed out after 20 seconds: company_type selection.",
    )


def test_select_single_candidate_retries_same_prompt_and_reports_mismatch() -> None:
    llm = _SequentialTextLLM(
        '{"candidate_legal_structures": ["Company Limited by Shares (AG / SA)"]}',
        "Company Limited by Shares (AG / SA)",
    )
    reports = []

    selected = asyncio.run(
        _select_single_candidate(
            llm,
            "Prompt text",
            ["Company Limited by Shares (AG / SA)"],
            "legal_structure",
            lambda step, message: reports.append((step, message)),
            "analyze_page",
        )
    )

    assert selected == "Company Limited by Shares (AG / SA)"
    assert len(llm.messages) == 2
    assert llm.messages[0][-1].content == llm.messages[1][-1].content == "Prompt text"
    assert reports == [
        (
            "analyze_page",
            "LLM legal_structure response did not match list elements; retrying once. "
            "Response: {\"candidate_legal_structures\": [\"Company Limited by Shares (AG / SA)\"]}",
        )
    ]


def test_select_multiple_candidates_returns_message_after_retry_failure() -> None:
    llm = _SequentialTextLLM("Invented", "Still invented")
    reports = []

    selected = asyncio.run(
        _select_multiple_candidates(
            llm,
            "Prompt text",
            ["Food Processing", "Consumer Goods"],
            2,
            "industry",
            lambda step, message: reports.append((step, message)),
            "analyze_page",
        )
    )

    assert selected == [LLM_LIST_SELECTION_MISMATCH]
    assert len(llm.messages) == 2
    assert reports[-1] == (
        "analyze_page",
        "LLM industry response did not match list elements. Response: Still invented",
    )


def test_extract_company_type_prompt_excludes_organization_description(tmp_path) -> None:
    company_types_path = tmp_path / "company_types.csv"
    company_types_path.write_text("Commercial Enterprise,Academic Institution", encoding="utf-8")
    llm = _SequentialTextLLM("Commercial Enterprise")

    selected = asyncio.run(
        _extract_company_type(
            llm,
            "This description should not be included.",
            "Current page text with company context.",
            str(company_types_path),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[0][-1].content
    assert selected == "Commercial Enterprise"
    assert "Organization description:" not in prompt
    assert "This description should not be included." not in prompt
    assert "Current page text:" in prompt
    assert "Allowed answers:" in prompt
    assert prompt.index("Allowed answers:") < prompt.index("Current page text:")
    assert "If the organization sells products or services commercially" in prompt
    assert "Do not return any sentence from the page." in prompt
    assert "If none fits the organization description" not in prompt


def test_extract_legal_structure_prompt_uses_allowed_answers_before_page_text(tmp_path) -> None:
    legal_structures_path = tmp_path / "legal_structures.csv"
    legal_structures_path.write_text(
        "Company Limited by Shares (AG / SA),Limited Liability Company (GmbH / Sàrl)",
        encoding="utf-8",
    )
    llm = _SequentialTextLLM("Company Limited by Shares (AG / SA)")

    selected = asyncio.run(
        _extract_legal_structure(
            llm,
            "Zweifel Chips & Snacks AG page text.",
            str(legal_structures_path),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[0][-1].content
    assert selected == "Company Limited by Shares (AG / SA)"
    assert "Classify the organization into exactly one legal structure." in prompt
    assert prompt.index("Allowed answers:") < prompt.index("Current page text:")
    assert "AG or SA" in prompt
    assert "Do not return any sentence from the page." in prompt
    assert "Do not return JSON." in prompt


def test_extract_sector_prompt_uses_allowed_answers_before_context(tmp_path) -> None:
    sectors_path = tmp_path / "sectors.csv"
    sectors_path.write_text("Producer (primary),Manufacturer (secondary)", encoding="utf-8")
    llm = _SequentialTextLLM("Manufacturer (secondary)")

    selected = asyncio.run(
        _extract_sector(
            llm,
            "The organization makes snacks.",
            "Page text about production.",
            str(sectors_path),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[0][-1].content
    assert selected == "Manufacturer (secondary)"
    assert "Classify the organization into exactly one economic sector." in prompt
    assert prompt.index("Allowed answers:") < prompt.index("Organization description:")
    assert prompt.index("Allowed answers:") < prompt.index("Current page text:")
    assert "Base the choice on what the organization does." in prompt
    assert "Do not return any sentence from the page." in prompt
    assert "If none fits the organization description" not in prompt


def test_extract_industries_prompt_uses_allowed_answers_before_description(tmp_path) -> None:
    industries_path = tmp_path / "industries.csv"
    industries_path.write_text("Food Processing,Consumer Goods", encoding="utf-8")
    llm = _SequentialTextLLM("Food Processing")

    selected = asyncio.run(
        _extract_industries(
            llm,
            "The organization makes snacks.",
            str(industries_path),
            2,
            25,
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[0][-1].content
    assert selected == ["Food Processing"]
    assert "Classify the organization into at most 2 industries." in prompt
    assert prompt.index("Allowed answers:") < prompt.index("Organization description:")
    assert "Return one allowed answer per line." in prompt
    assert "Do not quote text from the page or description." in prompt
    assert "Do not return JSON." in prompt


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
    assert "Do not summarize the page." in prompt
    assert "No requested profile fields found." in prompt


def test_extract_page_info_prompt_includes_employees_guidance() -> None:
    llm = _CapturingStructuredLLM()
    parser = PydanticOutputParser(pydantic_object=PageExtraction)

    asyncio.run(
        _extract_page_info(
            llm,
            parser,
            "Example Ltd",
            "https://example.com",
            ["employees"],
            WebsitePage(url="https://example.com", text="About page text."),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[-1].content

    assert "For employees" in prompt
    assert "estimate the number of employees as an integer" in prompt


def test_extract_contact_info_prompt_uses_contact_schema_only() -> None:
    llm = _CapturingStructuredLLM()
    parser = PydanticOutputParser(pydantic_object=ContactPageExtraction)

    asyncio.run(
        _extract_contact_info(
            llm,
            parser,
            "Example Ltd",
            "https://example.com",
            ["address", "phone", "email"],
            WebsitePage(url="https://example.com/contact", text="Contact page text."),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[-1].content
    assert "Extract only these missing contact fields" in prompt
    assert "- main address" in prompt
    assert "- address" not in prompt
    assert "- phone" in prompt
    assert "- email" in prompt
    assert "Do not infer contact details." in prompt
    assert "address" in ContactPageExtraction.model_json_schema()["$defs"][
        "ContactProfilePatch"
    ]["properties"]
    assert "employees" not in str(ContactPageExtraction.model_json_schema())
    assert "country" not in str(ContactPageExtraction.model_json_schema())


def test_extract_contact_info_prompt_omits_country_focus_by_default() -> None:
    llm = _CapturingStructuredLLM()
    parser = PydanticOutputParser(pydantic_object=ContactPageExtraction)

    asyncio.run(
        _extract_contact_info(
            llm,
            parser,
            "Example Ltd",
            "https://example.com",
            ["address"],
            WebsitePage(url="https://example.com/contact", text="Contact page text."),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[-1].content
    assert "Prioritize information for" not in prompt


def test_extract_contact_info_prompt_includes_country_focus_name() -> None:
    llm = _CapturingStructuredLLM()
    parser = PydanticOutputParser(pydantic_object=ContactPageExtraction)

    asyncio.run(
        _extract_contact_info(
            llm,
            parser,
            "Example Ltd",
            "https://example.com",
            ["address"],
            WebsitePage(url="https://example.com/contact", text="Contact page text."),
            None,
            "analyze_page",
            country_focus_name="Switzerland",
        )
    )

    prompt = llm.messages[-1].content
    assert "Prioritize information for Switzerland." in prompt


def test_extract_company_facts_prompt_extracts_address_country_and_employees() -> None:
    llm = _CapturingStructuredLLM()
    parser = PydanticOutputParser(pydantic_object=CompanyFactsExtraction)

    asyncio.run(
        _extract_company_facts(
            llm,
            parser,
            "Example Ltd",
            "https://example.com",
            ["address_country", "employees"],
            WebsitePage(url="https://example.com/about", text="About page text."),
            None,
            "analyze_page",
        )
    )

    prompt = llm.messages[-1].content
    assert "Extract only these missing company fact fields" in prompt
    assert "- address_country" in prompt
    assert "- employees" in prompt
    assert "For address_country, return the country of the organization's headquarters." in prompt
    assert "only when the page states employee count" in prompt
    assert "Use null for employees if no employee count is stated." in prompt
    fact_properties = CompanyFactsExtraction.model_json_schema()["$defs"][
        "CompanyFactsProfilePatch"
    ]["properties"]
    assert "address" not in fact_properties
    assert "phone" not in fact_properties
    assert "country" not in fact_properties
    assert "address_country" in fact_properties


def test_extract_company_facts_prompt_includes_country_focus_name() -> None:
    llm = _CapturingStructuredLLM()
    parser = PydanticOutputParser(pydantic_object=CompanyFactsExtraction)

    asyncio.run(
        _extract_company_facts(
            llm,
            parser,
            "Example Ltd",
            "https://example.com",
            ["address_country", "employees"],
            WebsitePage(url="https://example.com/about", text="About page text."),
            None,
            "analyze_page",
            country_focus_name="Liechtenstein",
        )
    )

    prompt = llm.messages[-1].content
    assert "Prioritize information for Liechtenstein." in prompt


def test_select_next_links_prompt_includes_country_focus_name() -> None:
    llm = _CapturingCrawlDecisionLLM()
    parser = PydanticOutputParser(pydantic_object=CrawlDecision)

    asyncio.run(
        _select_next_links(
            llm,
            parser,
            "Example Ltd",
            ["address"],
            [
                WebsiteLink(
                    url="https://example.com/ch/contact",
                    text="Contact",
                    area="navigation",
                )
            ],
            None,
            "analyze_page",
            country_focus_name="Switzerland",
        )
    )

    prompt = llm.messages[-1].content
    assert "Prioritize links that appear to belong to Switzerland." in prompt


def test_country_focus_resolves_country_name_from_code() -> None:
    assert _country_focus("ch") == ("CH", "Switzerland")
    assert _country_focus("li") == ("LI", "Liechtenstein")
    assert _country_focus(None) is None


def test_first_page_blocked_reason_detects_short_blocked_root_page() -> None:
    page = WebsitePage(
        url="https://example.com",
        text="Access denied\nYour request was blocked.",
    )

    reason = _first_page_blocked_reason(page, depth=0)

    assert reason is not None
    assert "blocked" in reason


def test_first_page_blocked_reason_ignores_long_pages() -> None:
    page = WebsitePage(
        url="https://example.com",
        text="\n".join(["This page says blocked but has content."] * 26),
    )

    assert _first_page_blocked_reason(page, depth=0) is None


def test_first_page_blocked_reason_ignores_non_root_pages() -> None:
    page = WebsitePage(url="https://example.com/contact", text="Access denied")

    assert _first_page_blocked_reason(page, depth=1) is None


def test_run_lookup_sets_failed_status_for_blocked_first_page(monkeypatch) -> None:
    async def fake_fetch_page_with_playwright(*args, **kwargs):
        return (
            WebsitePage(
                url="https://example.com",
                title="Access Denied",
                text="Access denied\nYour request was blocked.",
            ),
            [],
            None,
        )

    monkeypatch.setattr("org_agent.graph.build_chat_model", lambda settings: _CapturingStructuredLLM())
    monkeypatch.setattr("org_agent.graph.fetch_page_with_playwright", fake_fetch_page_with_playwright)

    result = asyncio.run(
        run_lookup(
            LookupInput(name="Example Ltd", website="https://example.com"),
            Settings(
                llm_provider="ollama",
                llm_model="test-model",
                ollama_base_url="http://localhost:11434",
                crawl_log_enabled=False,
            ),
        )
    )

    assert result.website_profile.status == "FAILED"
    assert result.website_profile.evidence == []


def test_missing_profile_fields_returns_only_empty_extractable_fields() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        queried_website="https://example.com",
        industry="Software",
        sector="Knowledge & Information (quaternary)",
        company_type="Commercial Enterprise",
        employees=42,
        address="Example Street 1",
    )

    assert _missing_profile_fields(profile) == [
        "phone",
        "email",
        "address_country",
    ]
    assert _missing_crawl_fields(profile) == [
        "legal_structure",
        "phone",
        "description",
        "email",
        "address_country",
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
    assert "legal_structure" not in patch_properties
    assert "sector" not in patch_properties
    assert "company_type" not in patch_properties
    assert "employees" in patch_properties
    assert "country" not in patch_properties
    assert "legal_address" not in str(schema)
    assert "official_company_name" not in str(schema)
    assert "queried_name" not in str(schema)
    assert "purpose" not in str(schema)
    assert "registration_id" not in str(schema)
    assert "region" not in str(schema)


def test_keep_requested_extraction_fields_removes_unrequested_values() -> None:
    extraction = PageExtraction(
        profile_patch=WebsiteOrganizationProfilePatch(
            address="Example Street 1",
            phone="+1 555 0100",
        ),
        evidence=[
            EvidenceEntry(
                field="official_company_name",
                value="Example Ltd Official",
                reasoning="Found on page.",
            ),
            EvidenceEntry(field="address", value="Example Street 1", reasoning="Found on page."),
            EvidenceEntry(field="phone", value="+1 555 0100", reasoning="Found on page."),
        ],
        missing_fields=["address", "phone", "email"],
        reasoning="Extracted fields.",
    )

    _keep_requested_extraction_fields(extraction, ["address", "email"])

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
        legal_structure="AG",
        address="Registry Street 2",
        phone="+1 555 9999",
        email="registry@example.com",
        country="Switzerland",
        legal_address="Registry Street 2, 8000 Zurich",
    )

    filled_fields = _merge_profile_patch_missing_only(profile, patch)

    assert profile.official_company_name == "Example Ltd Official"
    assert profile.registration_id == "CHE-123"
    assert profile.legal_structure == "AG"
    assert profile.legal_address == "Registry Street 2, 8000 Zurich"
    assert profile.address == "Website Street 1"
    assert profile.phone == "+1 555 0100"
    assert profile.email == "info@example.com"
    assert profile.country == "Website Country"
    assert filled_fields == [
        ("official_company_name", "Example Ltd Official"),
        ("registration_id", "CHE-123"),
        ("legal_structure", "AG"),
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
    assert _queried_country_value("li") == "LI"
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


def test_missing_registry_integration_message_uses_uppercase_country_code() -> None:
    assert (
        _missing_registry_integration_message("li")
        == "Registry lookup was not called because no registry integration is available for LI."
    )


def test_registry_result_message_explains_missing_registry_integration() -> None:
    assert (
        _registry_result_message("li", [], None)
        == "Registry lookup was not called because no registry integration is available for LI."
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


def test_country_from_address_uses_explicit_country_name() -> None:
    assert (
        _country_from_address("Feldkircher Strasse 100, Postfach 333, 9494 Schaan, Liechtenstein")
        == "Liechtenstein"
    )


def test_country_from_address_uses_postal_country_prefix() -> None:
    assert _country_from_address("Feldkircher Strasse 100, LI-9494 Schaan") == "Liechtenstein"
    assert _country_from_address("Regensdorferstrasse 20, CH-8049 Zürich-Höngg") == "Switzerland"


def test_country_from_address_does_not_guess_from_city() -> None:
    assert _country_from_address("Regensdorferstrasse 20, 8049 Zürich-Höngg") is None


def test_derive_profile_country_from_address_sets_address_country() -> None:
    profile = OrganizationProfile(
        queried_name="Hilti Corporation",
        address="Feldkircher Strasse 100, Postfach 333, 9494 Schaan, Liechtenstein",
    )

    derived_country = _derive_profile_country_from_address(profile)

    assert derived_country == "Liechtenstein"
    assert profile.address_country == "Liechtenstein"
    assert profile.country is None
    assert profile.evidence[-1].field == "address_country"
    assert profile.evidence[-1].value == "Liechtenstein"
    assert profile.evidence[-1].source == "agent"


def test_derive_profile_country_from_address_overrides_draft_address_country() -> None:
    profile = OrganizationProfile(
        queried_name="Hilti Corporation",
        address_country="Switzerland",
        address="Feldkircher Strasse 100, Postfach 333, 9494 Schaan, Liechtenstein",
        evidence=[
            EvidenceEntry(
                field="address_country",
                value="Switzerland",
                reasoning="Drafted from company facts.",
            )
        ],
    )

    derived_country = _derive_profile_country_from_address(profile)

    assert derived_country == "Liechtenstein"
    assert profile.address_country == "Liechtenstein"
    assert [entry.value for entry in profile.evidence if entry.field == "address_country"] == [
        "Liechtenstein"
    ]


def test_derive_profile_country_from_address_keeps_draft_without_country_signal() -> None:
    profile = OrganizationProfile(
        queried_name="Zweifel AG",
        address_country="Switzerland",
        address="Regensdorferstrasse 20, 8049 Zürich-Höngg",
    )

    derived_country = _derive_profile_country_from_address(profile)

    assert derived_country is None
    assert profile.address_country == "Switzerland"


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


def test_validate_profile_email_normalizes_obfuscated_email_present_in_website_pages() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        email="info <at> example (dot) com",
        evidence=[
            EvidenceEntry(
                field="email",
                value="info <at> example (dot) com",
                reasoning="Found on page.",
            )
        ],
    )
    pages = [
        WebsitePage(url="https://example.com", text="Contact: info <at> example (dot) com")
    ]

    _validate_profile_email(profile, pages)

    assert profile.email == "info@example.com"
    assert [entry.field for entry in profile.evidence] == ["email"]


def test_validate_profile_email_normalizes_obfuscated_email_with_spaced_markers() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        email="info [at] example [dot] com",
    )
    pages = [
        WebsitePage(url="https://example.com", text="Contact: info [at] example [dot] com")
    ]

    _validate_profile_email(profile, pages)

    assert profile.email == "info@example.com"


def test_validate_profile_email_allows_normalized_email_presence_fallback() -> None:
    profile = OrganizationProfile(queried_name="Example Ltd", email="info (at) example <dot> com")
    pages = [WebsitePage(url="https://example.com", text="Contact: info@example.com")]

    _validate_profile_email(profile, pages)

    assert profile.email == "info@example.com"


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


def test_validate_profile_email_removes_invalid_email_before_website_text_check() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        email="Contact form",
        evidence=[
            EvidenceEntry(field="email", value="Contact form", reasoning="Found on page."),
            EvidenceEntry(field="country", value="Switzerland", reasoning="Found on page."),
        ],
    )
    pages = [WebsitePage(url="https://example.com", text="Contact: Contact form")]

    _validate_profile_email(profile, pages)

    assert profile.email is None
    assert [entry.field for entry in profile.evidence] == ["country"]


def test_validate_profile_email_removes_invalid_email_without_website_pages() -> None:
    profile = OrganizationProfile(queried_name="Example Ltd", email="info at example dot com")

    _validate_profile_email(profile, [])

    assert profile.email is None


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
            legal_structure="Limited Liability Company (GmbH / Sàrl)",
            description="Example Ltd does things.",
            industry="Food",
            sector="Manufacturer (secondary)",
            company_type="Commercial Enterprise",
            employees=42,
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


def test_should_continue_crawl_reports_control_state_without_internal_flag() -> None:
    messages: list[tuple[str, str]] = []
    state = AgentState(
        input=LookupInput(name="Example Ltd", website="https://example.com"),
        profile=OrganizationProfile(queried_name="Example Ltd", queried_website="https://example.com"),
        website_pages=[WebsitePage(url="https://example.com", text="Page text.")],
        pending_urls=[CrawlTarget(url="https://example.com/facts", depth=1, node_id=1)],
    )

    should_continue = _should_continue_crawl(
        state,
        _CrawlSettings(),
        queued_selected_link=True,
        progress=lambda scope, message: messages.append((scope, message)),
        scope="analyze_page",
    )

    rendered = "\n".join(message for _, message in messages)
    assert should_continue is True
    assert "depth=0/2" in rendered
    assert "links_in_queue=1" in rendered
    assert "minimum_profile=false" in rendered
    assert "queued_selected_link" not in rendered
    assert "Continuing crawl: selected links were added to the queue." in rendered


def test_should_continue_after_skipped_page_continues_when_links_remain() -> None:
    messages: list[tuple[str, str]] = []
    state = AgentState(
        input=LookupInput(name="Example Ltd", website="https://example.com"),
        profile=OrganizationProfile(queried_name="Example Ltd", queried_website="https://example.com"),
        website_pages=[WebsitePage(url="https://example.com", text="Page text.")],
        pending_urls=[CrawlTarget(url="https://example.com/impressum", depth=1, node_id=2)],
    )

    should_continue = _should_continue_after_skipped_page(
        state,
        _CrawlSettings(),
        progress=lambda scope, message: messages.append((scope, message)),
        scope="crawl_page",
    )

    rendered = "\n".join(message for _, message in messages)
    assert should_continue is True
    assert "depth=0/2" in rendered
    assert "links_in_queue=1" in rendered
    assert "Continuing crawl: skipped page, links remain in queue." in rendered


def test_should_continue_after_skipped_page_stops_without_links() -> None:
    messages: list[tuple[str, str]] = []
    state = AgentState(
        input=LookupInput(name="Example Ltd", website="https://example.com"),
        profile=OrganizationProfile(queried_name="Example Ltd", queried_website="https://example.com"),
        website_pages=[WebsitePage(url="https://example.com", text="Page text.")],
    )

    should_continue = _should_continue_after_skipped_page(
        state,
        _CrawlSettings(),
        progress=lambda scope, message: messages.append((scope, message)),
        scope="crawl_page",
    )

    rendered = "\n".join(message for _, message in messages)
    assert should_continue is False
    assert "depth=0/2" in rendered
    assert "links_in_queue=0" in rendered
    assert "Stopping crawl: no links in queue." in rendered


def test_has_minimum_profile_requires_employees() -> None:
    profile = OrganizationProfile(
        queried_name="Example Ltd",
        queried_website="https://example.com",
        legal_structure="Limited Liability Company (GmbH / Sàrl)",
        description="Example Ltd does things.",
        industry="Food",
        sector="Manufacturer (secondary)",
        company_type="Commercial Enterprise",
        phone="+1 555 0100",
        evidence=[EvidenceEntry(field="phone", value="+1 555 0100", reasoning="Found.")],
    )

    assert _has_minimum_profile(profile) is False

    profile.employees = 42

    assert _has_minimum_profile(profile) is True


def test_truncate_progress_value_appends_ellipsis_at_limit() -> None:
    value = "x" * 61

    assert _truncate_progress_value(value) == f"{'x' * 60}..."
