from org_agent.models import WebsiteLink
from org_agent.website import filter_candidate_links


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
