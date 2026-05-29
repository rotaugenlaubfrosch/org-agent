# org-agent

`org-agent` enriches an organization profile from a company or organization name and, optionally, a known website.

It is a Python package and CLI built with LangGraph, Playwright, Typer, Rich, and `uv`.

## What It Does

- Looks up an organization by name.
- Uses a provided website, or discovers one through an optional search provider.
- Crawls the website with Playwright. 
- Follows useful links found on the website in breadth-first order, such as contact, imprint, legal, privacy, and about pages.
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

# Optional: Swiss company register (Zefix) credentials for --registry zefix
ORG_AGENT_ZEFIX_USERNAME=<zefix username>
ORG_AGENT_ZEFIX_PASSWORD=<zefix password>

# Optional runtime environment variables:
ORG_AGENT_REQUEST_TIMEOUT=<seconds, default 20>
ORG_AGENT_CRAWL_MAX_PAGES=<pages, default 6>
ORG_AGENT_CRAWL_MAX_DEPTH=<link depth, default 2>
ORG_AGENT_CRAWL_LOG_ENABLED=<true|false, default true>
ORG_AGENT_CRAWL_LOG_DIR=<directory for per-run page text logs, optional>
ORG_AGENT_PLAYWRIGHT_HEADLESS=<true|false, default true>
ORG_AGENT_PLAYWRIGHT_SLOW_MO=<milliseconds, default 0>
ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT=<system prompt for dedicated description extraction>
ORG_AGENT_INDUSTRIES_CSV=src/org_agent/data/industries.csv
ORG_AGENT_MAX_INDUSTRIES=<count, default 1>
ORG_AGENT_INDUSTRY_SHORTLIST_SIZE=<count, default 25>
```


## CLI

Run `uv run org-agent` to show the help dashboard.

Common lookup options:

- `--website <url>`: use a known official website
- `--registry <id>`: enable optional registry provider (currently `zefix`)
- `--json`: print raw JSON output
- `--quiet`: suppress progress output

Run with a known website:

```bash
uv run org-agent lookup "Example Ltd" --website https://example.com/
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

Use Zefix from CLI flags:

```bash
uv run org-agent lookup "Example Ltd" --registry zefix
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
- asks the LLM which remaining candidate links should be visited next, prioritizing links likely to contain fields that are still missing
- repeats page-by-page up to the configured crawl limits

The crawl uses a hybrid approach. Deterministic filtering removes obvious noise before the LLM sees the links, including shopping, account/login, product detail, campaign, social media, and static asset links. Candidate links must also contain an organization-information signal such as contact, imprint/legal, privacy, company/about, story, or terms. The LLM extracts profile data from the current page, then receives the filtered candidate links and the fields still missing from the profile. It selects up to three actual discovered URLs most likely to contain that missing company information. The crawler processes queued links in breadth-first order and does not invent `/contact` or `/impressum` paths.

The `description` and `industry` fields use dedicated prompts instead of the generic page extraction prompt. `description` is generated from the current crawled page text using `ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT`. After description generation, `industry` is selected from `ORG_AGENT_INDUSTRIES_CSV`. The industries CSV is a headerless comma-separated list, for example `Additive Manufacturing,Metal-Organic Frameworks (MOF),Advanced Manufacturing`. If the list contains more entries than `ORG_AGENT_INDUSTRY_SHORTLIST_SIZE`, the generated description and industry labels are embedded with `intfloat/multilingual-e5-small`, the closest configured industries are shortlisted, and the LLM chooses at most `ORG_AGENT_MAX_INDUSTRIES`. Returned industries are accepted only if they exactly match entries from the CSV.

Default crawl limits:

```env
ORG_AGENT_CRAWL_MAX_PAGES=6
ORG_AGENT_CRAWL_MAX_DEPTH=2
```

Set `ORG_AGENT_CRAWL_LOG_DIR=logs` to save captured page text. Each command execution creates a new timestamped subdirectory and writes one `.txt` file per captured web page. Set `ORG_AGENT_CRAWL_LOG_ENABLED=false` to disable this logging.

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

- `queried_name`
- `official_company_name`
- `website`
- `registration_id`
- `legal_form`
- `industry`
- `description`
- `purpose`
- `address`
- `legal_address`
- `phone`
- `email`
- `country`
- `region`
- `evidence`

The `queried_name` field is the original name passed to the lookup command or API. The CLI and experiment evaluator display the same ordered scalar fields. The `description` is generated by a dedicated description prompt configured with `ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT`. The `industry` field is selected from the configured industries CSV and may contain multiple comma-separated canonical entries when `ORG_AGENT_MAX_INDUSTRIES` is greater than 1. The `official_company_name`, `registration_id`, `purpose`, `legal_address`, and `region` fields are registry-only and are not crawled from websites; if no third-party registry is attached, they contain messages explaining that limitation. The `evidence` entries explain sources and decisions.

The experiment evaluator accepts the same registry inputs as normal lookups, for example `--registry zefix` or `--config registries.yml`.

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
