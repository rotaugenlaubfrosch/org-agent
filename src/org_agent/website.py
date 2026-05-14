from __future__ import annotations

from urllib.parse import urldefrag, urlparse

from playwright.async_api import async_playwright
from rich.markup import escape

from org_agent.models import CrawlNode, WebsiteLink, WebsitePage
from org_agent.progress import ProgressCallback, report

NOISE_LINK_PARTS = (
    "/account",
    "/cart",
    "/checkout",
    "/login",
    "/password",
    "/profile",
    "/register",
    "/shop",
    "/shop/produkt",
    "/shop/product",
    "advocate",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "promotionid=",
    "returnurl=",
    "tiktok.com",
    "twitter.com",
    "youtube.com",
)
INFORMATION_LINK_SIGNALS = (
    "about",
    "agb",
    "allgemeine geschäftsbedingungen",
    "company",
    "consumer service",
    "contact",
    "datenschutz",
    "datenschutzerklärung",
    "impressum",
    "imprint",
    "kontakt",
    "legal",
    "legal notice",
    "privacy",
    "story",
    "unternehmen",
    "ueber",
    "uber",
    "über",
)
NOISE_EXTENSIONS = (
    ".avi",
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mp3",
    ".mp4",
    ".png",
    ".svg",
    ".webp",
    ".zip",
)


async def fetch_page_with_playwright(
    url: str,
    headless: bool = True,
    slow_mo: int = 0,
    progress: ProgressCallback | None = None,
) -> tuple[WebsitePage | None, list[WebsiteLink], str | None]:
    try:
        report(progress, "website", f"Checking: {url}")
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless, slow_mo=slow_mo)
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page()
            try:
                await page.goto(_normalize_url(url), wait_until="domcontentloaded", timeout=15000)
                await _settle_page(page)
                final_url = _without_fragment(page.url)
                title = await page.title()
                text = await page.locator("body").inner_text(timeout=5000)
                cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
                links = await _extract_links(page)
            finally:
                await context.close()
                await browser.close()
    except Exception as exc:  # noqa: BLE001 - caller stores this as crawl node reason
        return None, [], f"page could not be loaded: {exc}"

    if not cleaned:
        return None, links, "page body was empty"

    return WebsitePage(url=final_url, title=title or None, text=cleaned[:12000]), links, None


def filter_candidate_links(
    links: list[WebsiteLink],
    root_url: str,
    current_url: str,
) -> list[WebsiteLink]:
    candidates: dict[str, WebsiteLink] = {}
    for link in links:
        url = _without_fragment(link.url)
        haystack = f"{url.lower()} {link.text.lower()}"
        if not url.startswith(("http://", "https://")):
            continue
        if any(part in haystack for part in NOISE_LINK_PARTS):
            continue
        if urlparse(url).path.lower().endswith(NOISE_EXTENSIONS):
            continue
        if not _has_information_signal(link):
            continue
        if not (_same_site(root_url, url) or _same_site(current_url, url) or _looks_official_external(link)):
            continue
        candidates.setdefault(url, WebsiteLink(url=url, text=link.text, area=link.area))
    return list(candidates.values())


def normalize_url(url: str) -> str:
    return _normalize_url(url)


def without_fragment(url: str) -> str:
    return _without_fragment(url)


def report_crawl_tree(nodes: list[CrawlNode], progress: ProgressCallback | None) -> None:
    if progress is None:
        return

    children: dict[int | None, list[CrawlNode]] = {}
    for node in nodes:
        children.setdefault(node.parent_id, []).append(node)

    input_count = sum(1 for node in nodes if node.status == "captured")
    report(progress, "website", f"Crawl tree: {input_count} page(s) selected as LLM input")
    for root in children.get(None, []):
        _report_node(root, children, progress, prefix="", is_last=True)


async def _settle_page(page) -> None:
    try:
        await page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass
    await page.evaluate(
        """
        async () => {
            const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
            const steps = 5;
            for (let i = 1; i <= steps; i++) {
                window.scrollTo(0, document.body.scrollHeight * i / steps);
                await delay(250);
            }
            window.scrollTo(0, 0);
            await delay(150);
        }
        """
    )


async def _extract_links(page) -> list[WebsiteLink]:
    links = await page.locator("a[href]").evaluate_all(
        """
        anchors => anchors.map(anchor => ({
            href: anchor.href,
            text: (anchor.innerText || anchor.ariaLabel || anchor.title || '').trim(),
            area: [anchor.closest('header'), anchor.closest('nav'), anchor.closest('footer')]
                .map(Boolean).some(Boolean) ? 'navigation' : 'body'
        }))
        """
    )
    candidates: dict[str, WebsiteLink] = {}
    for link in links:
        href = _without_fragment(str(link.get("href", "")))
        text = " ".join(str(link.get("text", "")).split()) or "(no_link_text)"
        area = str(link.get("area", "body"))
        if href.startswith(("http://", "https://")):
            candidates.setdefault(href, WebsiteLink(url=href, text=text, area=area))
    return list(candidates.values())


def _looks_official_external(link: WebsiteLink) -> bool:
    return _has_information_signal(link)


def _has_information_signal(link: WebsiteLink) -> bool:
    haystack = f"{link.url.lower()} {link.text.lower()}"
    return any(keyword in haystack for keyword in INFORMATION_LINK_SIGNALS)


def _same_site(left: str, right: str) -> bool:
    left_host = urlparse(left).netloc.lower().removeprefix("www.")
    right_host = urlparse(right).netloc.lower().removeprefix("www.")
    return left_host == right_host


def _normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return f"https://{url}"
    return url


def _without_fragment(url: str) -> str:
    return urldefrag(url).url


def _report_node(
    node: CrawlNode,
    children: dict[int | None, list[CrawlNode]],
    progress: ProgressCallback | None,
    prefix: str,
    is_last: bool,
) -> None:
    connector = "`-- " if is_last else "|-- "
    report(progress, "website", f"{prefix}{connector}{_format_node(node)}")
    child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
    child_nodes = children.get(node.id, [])
    for index, child in enumerate(child_nodes):
        _report_node(
            child,
            children,
            progress,
            prefix=child_prefix,
            is_last=index == len(child_nodes) - 1,
        )


def _format_node(node: CrawlNode) -> str:
    destination = node.final_url or node.requested_url
    label = node.label if node.label != "root" else destination
    details = _node_details(node)
    style = "green" if node.status == "captured" else "bright_black"
    return f"[{style}]{escape(label)} -> {escape(destination)}[/{style}]  {escape(details)}"


def _node_details(node: CrawlNode) -> str:
    if node.status == "captured":
        return f"LLM input, {node.char_count} chars"
    if node.status == "queued":
        return "queued, not visited"
    reason = f", {node.reason}" if node.reason else ""
    return f"skipped{reason}"
