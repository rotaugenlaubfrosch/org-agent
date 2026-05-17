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
