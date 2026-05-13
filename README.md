# org-agent

`org-agent` is a LangGraph-powered CLI and importable Python package for enriching organization profiles from a name and, optionally, a website.

It can:

- Discover a likely website through a configured search provider.
- Query optional registry endpoints from a YAML config file.
- Use Playwright to inspect the provided/discovered website and useful links found on that site.
- Extract structured fields with an API-hosted or local LLM.
- Return JSON or a readable terminal table with derivation evidence.

## Setup

```bash
uv sync --extra dev
uv run playwright install chromium
```

Create a `.env` file:

```env
ORG_AGENT_LLM_PROVIDER=openai
ORG_AGENT_LLM_MODEL=gpt-4.1-mini

# Required for OpenAI or Anthropic
ORG_AGENT_API_KEY=your-provider-key

# Optional search provider. Supported: tavily, brave, none.
# Name-only lookup needs a search provider or registry config.
ORG_AGENT_SEARCH_PROVIDER=none
ORG_AGENT_SEARCH_API_KEY=your-search-provider-key

# Required for Ollama
ORG_AGENT_OLLAMA_BASE_URL=http://localhost:11434

# Optional, defaults to 20
ORG_AGENT_REQUEST_TIMEOUT=20
ORG_AGENT_CRAWL_MAX_PAGES=6
ORG_AGENT_CRAWL_MAX_DEPTH=2
```

For Ollama:

```env
ORG_AGENT_LLM_PROVIDER=ollama
ORG_AGENT_LLM_MODEL=llama3.1
ORG_AGENT_OLLAMA_BASE_URL=http://localhost:11434

# Optional, defaults to 20
ORG_AGENT_REQUEST_TIMEOUT=20
ORG_AGENT_CRAWL_MAX_PAGES=6
ORG_AGENT_CRAWL_MAX_DEPTH=2
```

There are no built-in defaults for LLM settings. Missing required LLM values produce an error before the agent runs. Runtime settings are optional: `ORG_AGENT_REQUEST_TIMEOUT` defaults to 20 seconds, `ORG_AGENT_CRAWL_MAX_PAGES` defaults to 6, and `ORG_AGENT_CRAWL_MAX_DEPTH` defaults to 2. `OLLAMA_BASE_URL` is also accepted as a fallback, but `ORG_AGENT_OLLAMA_BASE_URL` is the preferred project-specific name.

## CLI

```bash
uv run org-agent lookup "Example Ltd"
uv run org-agent lookup "Example Ltd" --website https://example.com
uv run org-agent lookup "Example Ltd" --website https://example.com --json
uv run org-agent lookup "Example Ltd" --website https://example.com --quiet
uv run org-agent lookup "Example Ltd" --config org-agent.yaml
```

The CLI is verbose by default and shows a live trace of configuration, search, registry, website crawl, and extraction steps. Use `--quiet` / `-q` to suppress the trace. Progress is written to stderr, so `--json` still keeps the JSON result on stdout.

The website crawler starts only from the provided/discovered website URL. It does not guess paths like `/contact` or `/impressum`. Instead, it loads the page with Playwright, scrolls it, extracts visible text and actual page links, scores useful links such as contact/legal/imprint/about/privacy links, then queues the best candidates. Directly linked external domains are allowed only when the link is highly relevant, such as an official contact or imprint link. Link scoring and filtering are deterministic crawler logic, not LLM decisions. The LLM receives the gathered page/registry/search evidence and extracts the final structured organization profile.

Default crawl limits are `max_pages=6` and `max_depth=2`. The default trace prints concise `Checking:` lines while pages are crawled, then the final crawl tree. The tree shows parent-child link relationships, which pages were selected as LLM input, how many visible characters were extracted from each selected page, and which queued pages were not visited because a crawl limit was reached. In the tree, only the `link text -> URL` part is colored: captured pages are green, while skipped or unvisited pages are gray. Links without visible anchor text are shown as `(no_link_text)`.

If no website is supplied, configure a search provider. If `ORG_AGENT_SEARCH_PROVIDER=none`, lookup requires `--website`.

## Registry Config

Registry APIs are optional and generic. Because registry response formats vary, raw responses are collected and passed into the extraction step.

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

## Python API

```python
from org_agent import lookup_organization

profile = lookup_organization("Example Ltd", website="https://example.com")
print(profile.model_dump())
```

## Output Fields

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
- `derivation`
- `confidence`
