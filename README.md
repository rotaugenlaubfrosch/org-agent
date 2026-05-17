# org-agent

`org-agent` enriches an organization profile from a company or organization name and, optionally, a known website.

It is a Python package and CLI built with LangGraph, Playwright, Typer, Rich, and `uv`.

## What It Does

- Looks up an organization by name.
- Uses a provided website, or discovers one through an optional search provider.
- Crawls the website with Playwright. 
- Follows useful links found on the website, such as contact, imprint, legal, privacy, and about pages.
- Optionally queries configured registry API endpoints.
- Sends gathered evidence to an LLM.
- Returns a structured organization profile with evidence entries.

## Workflow

![LangGraph workflow](docs/graph.png)

## Setup

```bash
uv sync --extra dev
uv run playwright install chromium
```

Create a `.env` file in the project root.


```env
# Required environment variables:
ORG_AGENT_LLM_PROVIDER=openai|anthropic|ollama
ORG_AGENT_LLM_MODEL=<model name>

# Required for OpenAI/Anthropic:
ORG_AGENT_API_KEY=<provider API key>

# Required for Ollama:
ORG_AGENT_OLLAMA_BASE_URL=<Ollama base URL>

# Optional: Search provider used to discover a website when --website is not provided. Use none to disable search.
ORG_AGENT_SEARCH_PROVIDER=tavily|brave|none
ORG_AGENT_SEARCH_API_KEY=<search API key for tavily/brave>

# Optional runtime environment variables:
ORG_AGENT_REQUEST_TIMEOUT=<seconds, default 20>
ORG_AGENT_CRAWL_MAX_PAGES=<pages, default 6>
ORG_AGENT_CRAWL_MAX_DEPTH=<link depth, default 2>
ORG_AGENT_PLAYWRIGHT_HEADLESS=<true|false, default true>
ORG_AGENT_PLAYWRIGHT_SLOW_MO=<milliseconds, default 0>
```


## CLI

Run `uv run org-agent` to show the help dashboard.

Common lookup options:

- `--website <url>`: use a known official website
- `--json`: print raw JSON output
- `--quiet`: suppress progress output

Run with a known website:

```bash
uv run org-agent lookup "Zweifel Chips & Snacks AG" --website https://zweifel.ch/
```

Print JSON:

```bash
uv run org-agent lookup "Example Ltd" --website https://example.com --json
```

Suppress progress output:

```bash
uv run org-agent lookup "Example Ltd" --website https://example.com --quiet
```

The `lookup` command supports `--quiet` to suppress the live trace and show only the result.

Use a registry config:

```bash
uv run org-agent lookup "Example Ltd" --config org-agent.yaml
```

Name-only lookup requires either a configured search provider or an enabled registry config. If neither is configured, provide `--website`.

## Search

Search is optional. Supported providers are:

- `none`
- `tavily`
- `brave`

Example:

```env
ORG_AGENT_SEARCH_PROVIDER=tavily
ORG_AGENT_SEARCH_API_KEY=your-search-key
```

If `ORG_AGENT_SEARCH_PROVIDER=none`, the agent will not try to discover a website from the name.

## Website Crawling
 
The crawler starts only from the provided or discovered website URL. It does not guess paths like `/contact` or `/impressum`.

The crawler:

- opens the website with Playwright
- waits briefly for the page to settle
- scrolls the page to trigger lazy-loaded content
- extracts visible body text
- extracts actual links from the page
- removes obvious junk links such as carts, login pages, product detail pages, campaigns, and social media
- asks the LLM to update the partial profile from the current page
- asks the LLM whether enough information has been collected
- asks the LLM which remaining candidate links should be visited next
- repeats page-by-page up to the configured crawl limits

The crawl uses a hybrid approach. Deterministic filtering removes obvious noise before the LLM sees the links, including shopping, account/login, product detail, recipe, campaign, social media, and static asset links. Candidate links must also contain an organization-information signal such as contact, imprint/legal, privacy, company/about, story, or terms. The LLM then receives the current page text, accumulated page evidence, registry evidence, the current partial profile, and the filtered candidate links. It returns a profile patch, evidence entries, missing fields, a stop/continue decision, and up to three actual discovered URLs to visit next. The crawler does not invent `/contact` or `/impressum` paths.

Default crawl limits:

```env
ORG_AGENT_CRAWL_MAX_PAGES=6
ORG_AGENT_CRAWL_MAX_DEPTH=2
```

To watch Playwright operate in a visible browser window, disable headless mode:

```env
ORG_AGENT_PLAYWRIGHT_HEADLESS=false
ORG_AGENT_PLAYWRIGHT_SLOW_MO=300
```

`ORG_AGENT_PLAYWRIGHT_SLOW_MO` adds a delay in milliseconds to Playwright actions, which makes navigation easier to observe.

The default trace shows concise `Checking:` lines while crawling, the exact links passed to the LLM for each page, the links the LLM selected, then a final crawl tree. In the tree:

- green links were selected as LLM input
- gray links were skipped or queued but not visited
- `(no_link_text)` means the link had no visible anchor text
- character counts show how much visible text was extracted from each LLM input page

Example tree shape:

```text
website Crawl tree: 6 page(s) selected as LLM input
website `-- https://www.example.com -> https://www.example.com  LLM input, 2400 chars
website     |-- Contact -> https://www.example.com/contact  LLM input, 900 chars
website     |-- Imprint -> https://www.example.com/imprint  LLM input, 1300 chars
website     `-- Privacy -> https://www.example.com/privacy  queued, not visited
```

## Registry Config

Registry APIs are optional. The config is intentionally generic because registry APIs differ by country and provider.

Example `org-agent.yaml`:

```yaml
registries:
  - name: example_registry
    base_url: https://api.example.com/search
    method: GET
    query_param: q
    api_key_env: EXAMPLE_REGISTRY_API_KEY
    api_key_header: Authorization
    api_key_prefix: "Bearer "
    enabled: true
```

Registry responses are collected and passed to the extraction step as evidence.

## Python API

```python
from org_agent import lookup_organization

profile = lookup_organization(
    "Example Ltd",
    website="https://example.com",
)

print(profile.model_dump())
```

Async API:

```python
from org_agent.api import lookup_organization_async

profile = await lookup_organization_async(
    "Example Ltd",
    website="https://example.com",
)
```

## Output Fields

The result is an `OrganizationProfile` with:

- `name`
- `website`
- `registration_id`
- `legal_form`
- `industry`
- `description`
- `address`
- `phone`
- `email`
- `country`
- `region`
- `evidence`

The `description` should be factual and non-promotional. The `evidence` entries explain sources and decisions.

## Development

Run checks:

```bash
uv run ruff check .
uv run pytest
```

Show CLI help:

```bash
uv run org-agent --help
uv run org-agent lookup --help
```
