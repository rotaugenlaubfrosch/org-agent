from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urldefrag, urlparse

from playwright.async_api import async_playwright
from rich.markup import escape

from org_agent.models import WebsitePage
from org_agent.progress import ProgressCallback, report


LINK_KEYWORDS = (
    "about",
    "about-us",
    "contact",
    "contacts",
    "impressum",
    "imprint",
    "legal",
    "privacy",
    "datenschutz",
    "kontakt",
    "ueber-uns",
    "uber-uns",
)
HIGH_VALUE_KEYWORDS = (
    "contact",
    "contacts",
    "impressum",
    "imprint",
    "legal notice",
    "kontakt",
)
EXCLUDED_LINK_PARTS = (
    "/account",
    "/cart",
    "/checkout",
    "/login",
    "/profile",
    "returnurl=",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "youtube.com",
)


@dataclass(frozen=True)
class LinkCandidate:
    url: str
    text: str
    score: int


@dataclass(frozen=True)
class CrawlTarget:
    url: str
    depth: int
    node_id: int


@dataclass
class CrawlNode:
    id: int
    parent_id: int | None
    requested_url: str
    label: str
    depth: int
    score: int | None = None
    final_url: str | None = None
    status: str = "queued"
    reason: str | None = None
    char_count: int | None = None


async def crawl_website(
    website: str,
    max_pages: int = 6,
    max_depth: int = 2,
    headless: bool = True,
    slow_mo: int = 0,
    progress: ProgressCallback | None = None,
) -> list[WebsitePage]:
    normalized = _normalize_url(website)
    pages: list[WebsitePage] = []
    attempted: set[str] = set()
    captured: set[str] = set()
    queued: set[str] = {normalized}
    nodes: dict[int, CrawlNode] = {
        0: CrawlNode(id=0, parent_id=None, requested_url=normalized, label="root", depth=0)
    }
    candidates = [CrawlTarget(url=normalized, depth=0, node_id=0)]
    next_node_id = 1
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless, slow_mo=slow_mo)
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        index = 0
        while index < len(candidates):
            target = candidates[index]
            index += 1
            url = target.url
            node = nodes[target.node_id]

            if len(pages) >= max_pages:
                break
            if url in attempted:
                node.status = "skipped"
                node.reason = "already attempted"
                continue
            attempted.add(url)
            try:
                report(progress, "website", f"Checking: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await _settle_page(page)
                final_url = _without_fragment(page.url)
                node.final_url = final_url
                if final_url in captured:
                    node.status = "skipped"
                    node.reason = f"redirected to already captured {final_url}"
                    continue
                title = await page.title()
                text = await page.locator("body").inner_text(timeout=5000)
                cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
                if cleaned:
                    captured.add(final_url)
                    node.status = "captured"
                    node.char_count = len(cleaned)
                    pages.append(WebsitePage(url=final_url, title=title or None, text=cleaned[:12000]))
                    links = await _ranked_links(page, normalized, final_url)
                    if target.depth >= max_depth:
                        continue
                    queued_count = 0
                    for link in links:
                        if link.url in attempted or link.url in queued:
                            continue
                        nodes[next_node_id] = CrawlNode(
                            id=next_node_id,
                            parent_id=node.id,
                            requested_url=link.url,
                            label=link.text,
                            depth=target.depth + 1,
                            score=link.score,
                        )
                        candidates.append(
                            CrawlTarget(
                                url=link.url,
                                depth=target.depth + 1,
                                node_id=next_node_id,
                            )
                        )
                        queued.add(link.url)
                        queued_count += 1
                        next_node_id += 1
                else:
                    node.status = "skipped"
                    node.reason = "page body was empty"
            except Exception:
                node.status = "skipped"
                node.reason = "page could not be loaded"
                continue

        await context.close()
        await browser.close()

    _report_crawl_tree(nodes, progress)
    return pages


def _normalize_url(website: str) -> str:
    if not website.startswith(("http://", "https://")):
        return f"https://{website}"
    return website


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


async def _ranked_links(page, root_url: str, current_url: str) -> list[LinkCandidate]:
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
    candidates: dict[str, LinkCandidate] = {}
    for link in links:
        href = _without_fragment(str(link.get("href", "")))
        text = " ".join(str(link.get("text", "")).split()) or "(no_link_text)"
        area = str(link.get("area", "body"))
        if not href.startswith(("http://", "https://")):
            continue
        score = _score_link(href=href, text=text, area=area)
        if score <= 0:
            continue
        is_same_site = _same_site(root_url, href) or _same_site(current_url, href)
        is_relevant_external = score >= 6 and _has_high_value_signal(href, text)
        if not is_same_site and not is_relevant_external:
            continue
        current = candidates.get(href)
        candidate = LinkCandidate(url=href, text=text, score=score)
        if current is None or candidate.score > current.score:
            candidates[href] = candidate
    return sorted(candidates.values(), key=lambda candidate: candidate.score, reverse=True)


def _score_link(href: str, text: str, area: str = "body") -> int:
    haystack = f"{href.lower()} {text.lower()}"
    if any(part in haystack for part in EXCLUDED_LINK_PARTS):
        return 0

    score = 0
    for keyword in LINK_KEYWORDS:
        if keyword in haystack:
            score += 2
    for keyword in HIGH_VALUE_KEYWORDS:
        if keyword in haystack:
            score += 2
    if score == 0:
        return 0
    if area == "navigation":
        score += 1
    if urlparse(href).path in {"", "/"}:
        score -= 2
    return max(score, 0)


def _has_high_value_signal(href: str, text: str) -> bool:
    haystack = f"{href.lower()} {text.lower()}"
    return any(keyword in haystack for keyword in HIGH_VALUE_KEYWORDS)


def _same_site(left: str, right: str) -> bool:
    left_host = urlparse(left).netloc.lower().removeprefix("www.")
    right_host = urlparse(right).netloc.lower().removeprefix("www.")
    return left_host == right_host


def _without_fragment(url: str) -> str:
    return urldefrag(url).url


def _report_crawl_tree(nodes: dict[int, CrawlNode], progress: ProgressCallback | None) -> None:
    if progress is None:
        return

    children: dict[int | None, list[CrawlNode]] = {}
    for node in nodes.values():
        children.setdefault(node.parent_id, []).append(node)

    input_count = sum(1 for node in nodes.values() if node.status == "captured")
    report(progress, "website", f"Crawl tree: {input_count} page(s) selected as LLM input")
    for root in children.get(None, []):
        _report_node(root, children, progress, prefix="", is_last=True)


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
