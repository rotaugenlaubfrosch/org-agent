from org_agent.models import WebsiteLink
from org_agent.website import (
    MAX_CRAWLED_TEXT_WORD_LENGTH,
    PAGE_TEXT_CHAR_LIMIT,
    _limit_page_text,
    _remove_long_words_from_text,
    filter_candidate_links,
)


def test_filter_candidate_links_removes_shop_and_keeps_info_links() -> None:
    links = [
        WebsiteLink(url="https://www.zweifel.ch/ch_de/shop/", text="PRODUKTE", area="navigation"),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/shop/shop-cart/",
            text="(no_link_text)",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/wettbewerbe/",
            text="WETTBEWERBE",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/kontakt/",
            text="KONTAKT",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/impressum/",
            text="Impressum",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/datenschutzerklaerung/",
            text="Datenschutzerklärung",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/unternehmen/",
            text="UNTERNEHMEN",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/team/",
            text="Our people",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/facts/",
            text="Numbers",
            area="navigation",
        ),
        WebsiteLink(
            url="https://www.zweifel.ch/ch_de/portrait/",
            text="Profile",
            area="navigation",
        ),
    ]

    candidates = filter_candidate_links(
        links,
        root_url="https://www.zweifel.ch/ch_de/",
        current_url="https://www.zweifel.ch/ch_de/",
    )
    candidate_urls = {link.url for link in candidates}

    assert "https://www.zweifel.ch/ch_de/shop/" not in candidate_urls
    assert "https://www.zweifel.ch/ch_de/shop/shop-cart/" not in candidate_urls
    assert "https://www.zweifel.ch/ch_de/wettbewerbe/" not in candidate_urls
    assert "https://www.zweifel.ch/ch_de/kontakt/" in candidate_urls
    assert "https://www.zweifel.ch/ch_de/impressum/" in candidate_urls
    assert "https://www.zweifel.ch/ch_de/datenschutzerklaerung/" in candidate_urls
    assert "https://www.zweifel.ch/ch_de/unternehmen/" in candidate_urls
    assert "https://www.zweifel.ch/ch_de/team/" in candidate_urls
    assert "https://www.zweifel.ch/ch_de/facts/" in candidate_urls
    assert "https://www.zweifel.ch/ch_de/portrait/" in candidate_urls


def test_filter_candidate_links_orders_text_keyword_matches_first() -> None:
    links = [
        WebsiteLink(
            url="https://example.com/contact",
            text="Reach us",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.com/team",
            text="Contact",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.com/company",
            text="About us",
            area="navigation",
        ),
    ]

    candidates = filter_candidate_links(
        links,
        root_url="https://example.com",
        current_url="https://example.com",
    )

    assert [link.text for link in candidates] == ["Contact", "About us", "Reach us"]


def test_filter_candidate_links_removes_blacklisted_domains() -> None:
    links = [
        WebsiteLink(
            url="https://www.google.com/search?q=example+contact",
            text="Contact",
            area="navigation",
        ),
        WebsiteLink(
            url="https://mailchimp.com/legal/privacy/",
            text="Privacy",
            area="navigation",
        ),
        WebsiteLink(
            url="https://x.com/example/about",
            text="About",
            area="navigation",
        ),
        WebsiteLink(
            url="https://twitter.com/example/contact",
            text="Contact",
            area="navigation",
        ),
        WebsiteLink(
            url="https://instagram.com/example/contact",
            text="Contact",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.com/contact",
            text="Contact",
            area="navigation",
        ),
    ]

    candidates = filter_candidate_links(
        links,
        root_url="https://example.com",
        current_url="https://example.com",
    )

    assert [link.url for link in candidates] == ["https://example.com/contact"]


def test_filter_candidate_links_removes_pdfs() -> None:
    links = [
        WebsiteLink(
            url="https://example.com/legal/privacy-policy.pdf",
            text="Privacy policy",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.com/legal/privacy",
            text="Privacy policy",
            area="navigation",
        ),
    ]

    candidates = filter_candidate_links(
        links,
        root_url="https://example.com",
        current_url="https://example.com",
    )

    assert [link.url for link in candidates] == ["https://example.com/legal/privacy"]


def test_limit_page_text_reports_when_truncated() -> None:
    messages: list[tuple[str, str]] = []
    text = "x" * (PAGE_TEXT_CHAR_LIMIT + 1)

    limited = _limit_page_text(
        text,
        lambda step, message: messages.append((step, message)),
        "crawl_page",
        "https://example.com",
    )

    assert limited == text[:PAGE_TEXT_CHAR_LIMIT]
    assert messages == [
        (
            "crawl_page",
            "Page text limit reached for https://example.com: truncated extracted text "
            f"from {PAGE_TEXT_CHAR_LIMIT + 1} to {PAGE_TEXT_CHAR_LIMIT} chars.",
        )
    ]


def test_limit_page_text_does_not_report_when_within_limit() -> None:
    messages: list[tuple[str, str]] = []
    text = "x" * PAGE_TEXT_CHAR_LIMIT

    limited = _limit_page_text(
        text,
        lambda step, message: messages.append((step, message)),
        "crawl_page",
        "https://example.com",
    )

    assert limited == text
    assert messages == []


def test_remove_long_words_from_text_removes_words_longer_than_limit() -> None:
    long_word = "x" * (MAX_CRAWLED_TEXT_WORD_LENGTH + 1)

    cleaned = _remove_long_words_from_text(f"Keep {long_word} this")

    assert cleaned == "Keep this"


def test_remove_long_words_from_text_keeps_words_at_limit() -> None:
    boundary_word = "x" * MAX_CRAWLED_TEXT_WORD_LENGTH

    cleaned = _remove_long_words_from_text(f"Keep {boundary_word}")

    assert cleaned == f"Keep {boundary_word}"


def test_remove_long_words_from_text_preserves_lines_with_remaining_words() -> None:
    long_word = "x" * (MAX_CRAWLED_TEXT_WORD_LENGTH + 1)

    cleaned = _remove_long_words_from_text(f"First line\n{long_word}\nSecond line")

    assert cleaned == "First line\nSecond line"


def test_filter_candidate_links_prioritizes_country_branch_before_keyword_matches() -> None:
    links = [
        WebsiteLink(
            url="https://example.com/contact",
            text="Contact",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.com/ch/about",
            text="Read more",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.ch/privacy",
            text="Privacy",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.com/switzerland/company",
            text="Company",
            area="navigation",
        ),
    ]

    candidates = filter_candidate_links(
        links,
        root_url="https://example.com",
        current_url="https://example.com",
        country_focus_code="CH",
        country_focus_name="Switzerland",
    )

    assert [link.url for link in candidates] == [
        "https://example.ch/privacy",
        "https://example.com/switzerland/company",
        "https://example.com/ch/about",
        "https://example.com/contact",
    ]


def test_filter_candidate_links_without_country_keeps_existing_ordering() -> None:
    links = [
        WebsiteLink(
            url="https://example.com/contact",
            text="Contact",
            area="navigation",
        ),
        WebsiteLink(
            url="https://example.com/ch/about",
            text="Read more",
            area="navigation",
        ),
    ]

    candidates = filter_candidate_links(
        links,
        root_url="https://example.com",
        current_url="https://example.com",
    )

    assert [link.url for link in candidates] == [
        "https://example.com/contact",
        "https://example.com/ch/about",
    ]
