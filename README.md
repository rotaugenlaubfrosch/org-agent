> [!WARNING]
> **Work in Progress:** This repository is currently under development.


# org-agent 🤖

`org-agent` enriches an organization profile from a company or organization name and a known website.

It is a Python package and CLI tool built with LangGraph, Playwright, Typer, Rich, and uv.

## LangGraph Graph Structure

![LangGraph workflow](docs/schema.jpg)

## What It Does

- Looks up an organization by name.
- Uses the provided website as the crawl starting point.
- Crawls the website with Playwright. 
- Follows useful links found on the website in breadth-first order, such as contact, imprint, legal, privacy, and about pages.
- Optionally queries country registry APIs when an adapter exists.
- Optionally fragments website addresses into country-specific address fields.
- Sends gathered evidence to an LLM.
- Returns separate website and registry profiles with evidence entries.
- Reports website crawl `status` as `SUCCESS` or `FAILED` in CLI tables, JSON output, and evaluation tables.

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

O
# Required for OpenAI/Anthropic:
ORG_AGENT_API_KEY=<provider API key>

# Required for Ollama:
ORG_AGENT_OLLAMA_BASE_URL=<Ollama base URL>

# Optional: Swiss company register credentials for --country ch
ORG_AGENT_REGISTRY_CH_USERNAME=<Swiss registry username>
ORG_AGENT_REGISTRY_CH_PASSWORD=<Swiss registry password>

# Optional runtime environment variables:
ORG_AGENT_REQUEST_TIMEOUT=<seconds, default 20>
ORG_AGENT_CRAWL_MAX_PAGES=<pages, default 6>
ORG_AGENT_CRAWL_MAX_DEPTH=<link depth, default 2>
ORG_AGENT_CRAWL_LOG_ENABLED=<true|false, default true>
ORG_AGENT_CRAWL_LOG_DIR=<directory for per-run page text logs, default project logs/>
ORG_AGENT_PLAYWRIGHT_HEADLESS=<true|false, default true>
ORG_AGENT_PLAYWRIGHT_SLOW_MO=<milliseconds, default 0>
ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT=<system prompt for dedicated description extraction>
ORG_AGENT_LEGAL_STRUCTURES_CSV=src/org_agent/data/legal_structures.csv
ORG_AGENT_SECTORS_CSV=src/org_agent/data/sectors.csv
ORG_AGENT_COMPANY_TYPES_CSV=src/org_agent/data/company_types.csv
ORG_AGENT_INDUSTRIES_CSV=src/org_agent/data/industries.csv
ORG_AGENT_MAX_INDUSTRIES=<count, default 1>
ORG_AGENT_INDUSTRY_SHORTLIST_SIZE=<count, default 25>
```


## CLI

Run `uv run org-agent` to show the help dashboard.

Common lookup options:

- `--website <url>`: required official website. Bare domains like `example.com` are accepted and normalized to `https://example.com`.
- `--country <code>`: optional two-letter ISO country code. When set, website crawling and extraction prioritize that country branch, registry lookup is attempted if an adapter exists, and country-specific address field derivation is triggered.
- `--json`: print raw JSON output with separate `website_profile` and `registry_profile` objects
- `--quiet`: suppress progress output

Run with a known website:

```bash
uv run org-agent "Example Ltd" --website https://example.com/
```

Bare domains are accepted:

```bash
uv run org-agent "Example Ltd" --website example.com
```

Print JSON:

```bash
uv run org-agent "Example Ltd" --website https://example.com --json
```

Suppress progress output:

```bash
uv run org-agent "Example Ltd" --website https://example.com --quiet
```

The CLI supports `--quiet` to suppress the live trace and show only the result.

Use a country registry integration:

```bash
uv run org-agent "Example Ltd" --website example.com --country ch
```

`--website` is required even when country registry lookup is enabled. The `--country` option must be a two-letter ISO 3166-1 alpha-2 country code, for example `ch`, `li`, or `de`. When set, the country name is resolved dynamically with `pycountry`, website crawling prioritizes links that appear to belong to that country branch, and extraction prompts prioritize information for that country. If a registry adapter exists at `src/org_agent/countries/<code>/registry.py`, the registry lookup is attempted. If no adapter exists, or required registry credentials are missing, the registry lookup is skipped and the website crawl continues normally. Without `--country`, the generic website crawl and extraction behavior is unchanged.

## Website Crawling
 
The crawler starts only from the provided website URL. It does not search for a website and does not guess paths like `/contact` or `/impressum`.

The crawler:

- opens the website with Playwright
- marks the website profile `status` as `FAILED` and stops website crawling if the first crawled page has 25 or fewer non-empty text lines and contains `forbidden`, `blocked`, or `denied` case-insensitively
- waits briefly for the page to settle
- scrolls the page to trigger lazy-loaded content
- extracts visible body text, removes words longer than 25 characters, and caps the result at 20,000 characters per page
- extracts actual links from the page
- removes obvious junk links such as carts, login pages, product detail pages, campaigns, and social media
- keeps links with organization-information signals in the link text or URL, such as contact, imprint/legal, privacy, company/about, story, or terms
- orders filtered links so links with configured keywords in their visible link text are shown first, preserving page order within each group. When `--country` is set, country-branch-looking links are sorted before generic links, then the same keyword ordering is applied within each group
- sends at most the first 25 filtered and ordered candidate links to the LLM for link selection
- asks the LLM to update the partial profile from the current page
- asks the LLM whether enough information has been collected
- asks the LLM which remaining candidate links should be visited next, prioritizing links likely to contain fields that are still missing
- repeats page-by-page up to the configured crawl limits

The crawl uses a hybrid approach. The `filter_links` node deterministically filters and orders candidate links before the LLM sees them. Candidate links must contain an organization-information signal in the URL or visible link text, such as contact, imprint/legal, privacy, company/about, story, or terms. The later `analyze_page` node receives at most the first 25 filtered and ordered links plus the fields still missing from the profile, then asks the LLM which links should be crawled next. The crawler processes queued links in breadth-first order and does not invent `/contact` or `/impressum` paths.

If a page-text LLM call times out, the agent retries that prompt once with reduced page text. The retry keeps the first 25% and last 25% of the already-captured page text by character count and removes the middle 50%. The CLI prints this retry as an error-style red progress message. However, as I could observe so far, this does not mitigate the LLM timeout problem so far.

The website profile includes a final `status` field. It defaults to `SUCCESS`. It is set to `FAILED` only when the first crawled page looks like an access-block page by this deterministic rule: the extracted text has 25 or fewer non-empty lines and contains `forbidden`, `blocked`, or `denied` case-insensitively. In that case, website crawling stops before extraction. The field is visible in normal CLI output, raw `--json` output, and `experiments/evaluate_agent.py` output.

The `description`, `legal_structure`, `sector`, `company_type`, and `industry` fields use dedicated prompts instead of the generic page extraction prompt. `description` is generated from the current crawled page text using `ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT`. After description generation, `legal_structure` is selected from `ORG_AGENT_LEGAL_STRUCTURES_CSV` using only the current page text as context. The legal structures CSV is a headerless comma-separated list. By default it contains `Company Limited by Shares (AG / SA)`, `Limited Liability Company (GmbH / Sàrl)`, `Association (Verein)`, `Foundation (Stiftung)`, `Sole Proprietorship (Einzelunternehmen)`, `Partnership (Gesellschaft)`, `Cooperative (Genossenschaft)`, and `Public Law Institution (Öffentlich-rechtliche Anstalt)`. After legal structure selection, `sector` is selected from `ORG_AGENT_SECTORS_CSV`. The sectors CSV is a headerless comma-separated list. By default it contains `Producer (primary)`, `Manufacturer (secondary)`, `Professional Services (tertiary)`, `Knowledge & Information (quaternary)`, and `Governance (quinary)`. After sector selection, `company_type` is selected from `ORG_AGENT_COMPANY_TYPES_CSV`. The company types CSV is a headerless comma-separated list. By default it contains `Commercial Enterprise`, `Academic Institution`, `Research Institution`, `Government Organisation`, `Non-Government / Non-Profit Organisation (NGO / NPO)`, and `Innovation / Funding Agency`. After company type selection, `industry` is selected from `ORG_AGENT_INDUSTRIES_CSV`. The industries CSV is a headerless comma-separated list, for example `Additive Manufacturing,Metal-Organic Frameworks (MOF),Advanced Manufacturing`. If the list contains more entries than `ORG_AGENT_INDUSTRY_SHORTLIST_SIZE`, the generated description and industry labels are embedded with `intfloat/multilingual-e5-small`, the closest configured industries are shortlisted, and the LLM chooses at most `ORG_AGENT_MAX_INDUSTRIES`. Returned legal structures, sectors, company types, and industries are accepted only if they exactly match entries from the configured CSV files. Contact fields (`address`, `phone`, and `email`) are extracted by a dedicated contact prompt. The `employees` field is extracted by a separate company facts prompt and is filled only when a page explicitly states employee count, headcount, or number of employees. The website `country` field is not prompted from the LLM; it is derived from explicit country names or country-prefixed postal codes in the extracted address, such as `Liechtenstein`, `CH-8049`, or `LI-9494`.

The built-in default description prompt writes descriptions in English. Override `ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT` to change the language or wording.

The `address_country` field is drafted by the company facts prompt using the instruction: "For address_country, return the country of the organization's headquarters." During validation, explicit country evidence in the extracted `address`, such as `Liechtenstein`, `CH-8049`, or `LI-9494`, overrides that draft value.

The `email` field is validated after extraction. Validation trims whitespace, normalizes common obfuscations (`[at]`, `(at)`, `<at>` to `@` and `[dot]`, `(dot)`, `<dot>` to `.` including surrounding spaces), then discards blank values and values that do not match a conservative email regex. If crawled website pages exist, the original extracted email text must appear in the crawled text case-insensitively; the normalized email is also accepted as a fallback. Removed invalid or unconfirmed email values are shown as error-style CLI progress messages.

Default crawl limits:

```env
ORG_AGENT_CRAWL_MAX_PAGES=6
ORG_AGENT_CRAWL_MAX_DEPTH=2
```

Captured page text is saved by default under the project's `logs/` directory. Each command execution creates a new timestamped subdirectory and writes one `.txt` file per captured web page. Set `ORG_AGENT_CRAWL_LOG_DIR` to override the directory, or set `ORG_AGENT_CRAWL_LOG_ENABLED=false` to disable this logging.

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

## Country Registries

Country registry APIs are optional and selected by two-letter ISO country code. Registry support is discovered at runtime from `src/org_agent/countries/<code>/registry.py`; the repository currently includes `ch` for the Swiss company register.

Swiss registry credentials are optional. If either value is missing, `--country ch` skips the registry lookup and continues with the website crawl.

```env
ORG_AGENT_REGISTRY_CH_USERNAME=<Swiss registry username>
ORG_AGENT_REGISTRY_CH_PASSWORD=<Swiss registry password>
```

Country registry responses are collected separately from the website crawl. They are returned in `registry_profile` and do not influence website crawling or website field extraction.

## Customizable Country-Specific Address Fields

Website addresses can be fragmented into custom country-specific address fields after the website profile has been validated. The original `address` value remains unchanged. Derived address fragments are stored in the nested `address_fields` object.

Address field configs live at:

```text
src/org_agent/countries/<code>/address_fields.json
```

The default fallback config lives at:

```text
src/org_agent/countries/_DEFAULT/address_fields.json
```

For example, the Swiss config is:

```text
src/org_agent/countries/ch/address_fields.json
```

Each configured address field must have exactly two keys:

```json
{
  "city": {
    "prompt": "Extract the city or locality from the address.",
    "validation": true
  }
}
```

The configured field name is prefixed with `address_` in the output. For example, a config field named `city` is stored as `address_city` because these are fields derived by the `address` field.

Example output:

```json
{
  "address": "Example Organization, Example Street 12, 12345 Example City",
  "address_fields": {
    "address_organization": "Example Organization",
    "address_street": "Example Street",
    "address_number": "12",
    "address_postal_code": "12345",
    "address_city": "Example City"
  }
}
```

Address field config selection uses this priority:

- explicit `--country <code>`, using `src/org_agent/countries/<code>/address_fields.json` when it exists
- extracted `profile.address_country`, resolved to an ISO alpha-2 country code when possible, using `src/org_agent/countries/<code>/address_fields.json` when it exists
- `src/org_agent/countries/_DEFAULT/address_fields.json` when a country is known but no country-specific config exists
- skip address fragmentation if no country is available or no default config exists

Validation behavior:

- `validation: true` keeps an extracted value only if it appears in the original `address`, using case-insensitive whitespace-normalized comparison
- `validation: false` keeps any non-empty LLM output for that configured field
- fields not defined in the country JSON config are discarded

## Python API

```python
from org_agent import lookup_organization

result = lookup_organization(
    "Example Ltd",
    website="https://example.com",
)

print(result.website_profile.model_dump())
print(result.registry_profile.model_dump() if result.registry_profile else None)
```

Async API:

```python
from org_agent.api import lookup_organization_async

result = await lookup_organization_async(
    "Example Ltd",
    website="https://example.com",
)
```

## Output Fields

The result is a `LookupResult` with separate `website_profile` and `registry_profile` entries. If no registry profile is available, `registry_message` explains why the registry table is shown without registry values. Each profile is an `OrganizationProfile` with:

- `queried_name`
- `queried_website`
- `queried_country`
- `official_company_name`
- `registration_id`
- `legal_structure`
- `industry`
- `description`
- `sector`
- `company_type`
- `employees`
- `purpose`
- `address`
- `address_fields`
- `legal_address`
- `phone`
- `email`
- `address_country`
- `country`
- `region`
- `evidence`

The `queried_name` field is the original name passed to the CLI or API. The `queried_website` field is the normalized website passed with `--website` or `website=...`. The `queried_country` field is the explicit user-provided country as an uppercase two-letter code, or `not specified` if no country was provided. The `address_country` field is drafted by the company facts prompt and overridden when the extracted website address contains an explicit country signal. The `email` field is normalized from common `[at]`/`[dot]` style obfuscations, kept only when it has a valid email-like format and, when crawled pages exist, the original or normalized value appears in crawled website text. Registry profiles keep the separate `country` field from registry evidence. The CLI displays website and registry fields in separate tables. The `description` is generated by a dedicated description prompt configured with `ORG_AGENT_DESCRIPTION_SYSTEM_PROMPT`. The `legal_structure` field is selected from the configured legal structures CSV using current page text. The `sector` field is selected from the configured sectors CSV. The `company_type` field is selected from the configured company types CSV. The `employees` field is the extracted employee count as an integer when the website explicitly states a headcount. The `industry` field is selected from the configured industries CSV and may contain multiple comma-separated canonical entries when `ORG_AGENT_MAX_INDUSTRIES` is greater than 1. Fields such as `legal_structure` may appear in both `website_profile` and `registry_profile` because they come from independent sources. The `evidence` entries explain sources and decisions for each profile.

The experiment evaluator accepts the same country registry input as normal lookups, for example `--country ch`.

## Development

Run checks:

```bash
uv run ruff check .
uv run pytest
```

Show CLI help:

```bash
uv run org-agent --help
```
