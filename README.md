# langchain-jobo

LangChain integration for the [Jobo](https://jobo.world) jobs API: live job listings collected directly
from employer career sites and applicant tracking systems, as LangChain tools, a retriever and a document
loader.

| Component | What it does | Jobo endpoint | Billed |
| --- | --- | --- | --- |
| `JoboSearchTool` (`jobo_job_search`) | Search jobs with the full filter set | `POST /api/jobs/search` | Per job returned |
| `JoboJobDetailsTool` (`jobo_job_details`) | One listing as readable prose | `GET /api/jobs/{id}` | Free |
| `JoboCompanySearchTool` (`jobo_company_search`) | Company name or domain to company ids | `GET /api/connectors/companies/suggest` | Free |
| `JoboRetriever` | Query to job `Document`s | `POST /api/jobs/search` | Per job returned |
| `JoboFeedLoader` | Stream the job feed as `Document`s | `POST /api/jobs/feed` | Per new job |
| `get_jobo_tools()` | The three tools in one call | | |

Read [Cost](#cost) before pointing an agent at it.

## Install

```bash
pip install -U langchain-jobo
```

Requires Python 3.10+. The package depends on `langchain-core`, `httpx` and `pydantic` only.

## Authentication

Create an API key in the [Jobo dashboard](https://enterprise.jobo.world/api-keys?src=langchain) under
**Developers → API Keys** and export it:

```bash
export JOBO_API_KEY="jbe_live_..."
```

Every component reads `JOBO_API_KEY`, or takes `api_key=...`. The key is held as a pydantic `SecretStr`: it
is masked in `repr()` and model dumps, and is only ever written into the `X-Api-Key` request header.

A sandbox key (`jbe_test_...`) is free and returns fictional jobs in the real response shape. It serves
search and job lookups, so it works with `JoboSearchTool`, `JoboJobDetailsTool` and `JoboRetriever`. The
feed is not available to sandbox keys: `JoboFeedLoader` needs a live key. See the
[sandbox docs](https://jobo.world/docs/sandbox).

## Tools

### `JoboSearchTool`

```python
from langchain_jobo import JoboSearchTool

tool = JoboSearchTool()

print(
    tool.invoke(
        {
            "query": "backend engineer",
            "location": "Berlin, Germany",
            "work_model": ["remote", "hybrid"],
            "experience_level": ["senior"],
            "exclude_skills": ["PHP"],
            "min_salary_usd": 90000,
            "page_size": 3,
        }
    )
)
```

The model sees compact text it can cite: each job's id, title, company, location, work model, salary when
the posting states one, posted date and listing URL. Called with a `ToolCall`, the tool returns a
`ToolMessage` whose `artifact` holds the raw data for your own code:

```python
message = tool.invoke(
    {
        "type": "tool_call",
        "id": "call-1",
        "name": "jobo_job_search",
        "args": {"query": "nurse", "location": "Ohio"},
    }
)
message.content  # the compact text
message.artifact["jobs"]  # the raw job objects, descriptions included
message.artifact["total"]  # total matches, plus page, page_size, total_pages, warnings
message.artifact["usage"]  # e.g. {"credits_deducted": 15, "credits_balance": 4985}
```

Arguments: `query`, `location`, `sources`, `companies`, `work_model`, `employment_type`,
`experience_level`, `language`, `skills`, `industries`, `company_categories`, `job_categories`,
`job_subcategories`, `occupation_ids`, each include filter's `exclude_` twin (`exclude_companies`,
`exclude_skills`, `exclude_industries`, `exclude_company_categories`, `exclude_job_categories`,
`exclude_job_subcategories`, `exclude_occupation_ids`), `min_salary_usd`, `max_salary_usd`, `posted_after`,
`posted_before`, `discovered_after`, `discovered_before`, `search_description`, `sort`,
`description_format`, `page` and `page_size`.

- Enum filters take lowercase keys: `work_model` is `remote`, `hybrid` or `onsite`; `employment_type` is
  `full-time`, `part-time`, `contract`, `internship`, `freelance` or `temporary`; `experience_level` is
  `intern`, `entry`, `mid`, `senior`, `lead` or `executive`. An unknown value is rejected before any
  request is made, because the API would answer it with zero results rather than an error.
- List arguments take a list. A single comma-separated string is split for you.
- `page_size` defaults to 5. The tool never requests more than `max_page_size` jobs per call (default 25,
  at most 100, the API's own limit), whatever the model asks for:

```python
tool = JoboSearchTool(max_page_size=10)
```

### `JoboJobDetailsTool`

```python
from langchain_jobo import JoboJobDetailsTool

details = JoboJobDetailsTool()
print(details.invoke({"job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}))
```

Returns the listing as self-contained prose: company, location, compensation, responsibilities,
qualifications, benefits, the employer's description, and the listing and apply URLs. The raw job object is
the artifact. An unknown id comes back as a plain "no job exists with id ..." answer rather than an error.
Pass `include_description=False` for the structured fields only.

### `JoboCompanySearchTool`

```python
from langchain_jobo import JoboCompanySearchTool

companies = JoboCompanySearchTool()
print(companies.invoke({"query": "stripe"}))
```

The search filter `companies` matches a company *name* whole (`Glenmark` does not find `Glenmark
Pharmaceuticals`), and a name also matches every other company called that. This tool looks companies up by
the start of their name or website domain and returns their ids, which match exactly.

### All three at once

```python
from langchain_jobo import get_jobo_tools

tools = get_jobo_tools(max_page_size=10)
```

`get_jobo_tools(api_key=None, *, max_page_size=25, include_company_search=True, **kwargs)` passes any
other keyword (`base_url`, `timeout`, `max_retries`, `http_client`, `http_async_client`,
`handle_tool_error`) to every tool.

## Agent

Install LangChain and the integration package for your chat model, then:

```python
import os

from langchain.agents import create_agent
from langchain_jobo import get_jobo_tools

agent = create_agent(
    # A "provider:model" string for the chat model you use, or a chat model instance.
    model=os.environ["AGENT_MODEL"],
    tools=get_jobo_tools(max_page_size=10, handle_tool_error=True),
    system_prompt=(
        "You help people find jobs. Search with jobo_job_search, read a posting with "
        "jobo_job_details before describing it, and cite each job's listing URL."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Find three remote senior Python jobs paying over $150k.",
            }
        ]
    }
)
print(result["messages"][-1].content)
```

`handle_tool_error=True` hands an API error (a rate limit, an empty wallet) back to the model as the tool's
answer. Without it the tool raises a `ToolException` and the run stops, which is the safer default when a
failure should not be retried by the model.

## Retriever

```python
from langchain_jobo import JoboRetriever

retriever = JoboRetriever(
    k=5,
    filters={
        "location": "London",
        "work_model": ["remote", "hybrid"],
        "exclude_companies": ["Acme Staffing"],
    },
)

for doc in retriever.invoke("staff data engineer"):
    print(doc.metadata["title"], "-", doc.metadata["company"], doc.metadata["url"])
```

- `filters` takes the same filter names as the search tool (a dict or a `JoboSearchFilters`) and applies
  them to every query. Unknown names are rejected.
- `page_content` is the same prose `JoboJobDetailsTool` returns. `include_description=False` leaves the
  employer's description out.
- `metadata` holds `id`, `url`, `apply_url`, `title`, `company`, `company_id`, `location`,
  `workplace_type`, `employment_type`, `experience_level`, `salary`, `salary_min`, `salary_max`,
  `salary_currency`, `salary_period`, `date_posted`, `created_at`, `updated_at`, `valid_through` and
  `source` (the ATS the listing came from). A key is present only when the posting has the value; nothing
  is filled in.
- `Document.id` is the Jobo job id.
- `k` is 1-100 and can be overridden per call: `retriever.invoke("nurse", k=2)`. Each query bills up to
  `k` jobs.

## Document loader

```python
from langchain_jobo import JoboFeedLoader

loader = JoboFeedLoader(
    max_jobs=500,
    filters={
        "sources": ["greenhouse"],
        "work_models": ["remote"],
        "job_categories": {"include": ["Healthcare"]},
        "exclude_employment_types": ["internship"],
    },
)

for doc in loader.lazy_load():
    print(doc.metadata["title"])

if loader.has_more:
    print("resume from:", loader.last_cursor)
```

`max_jobs` is required and has no default, because the feed streams every matching job and bills each one
that is new to you. Pass a number, or `max_jobs=None` to opt in to reading the whole feed.

- **Whole batches only.** The loader never requests more jobs than `max_jobs` allows, so nothing is
  delivered and billed beyond the budget. It plans equal batches of at most `batch_size` (default and
  maximum 1,000) that fit under `max_jobs`; when `max_jobs` is not a whole number of batches it stops
  slightly short (`max_jobs=2500` loads 2,499 in three batches of 833). It never asks for a smaller last
  batch, because the feed keeps the smallest batch size a scan has used for every later page.
- **Resuming.** `loader.last_cursor` is the cursor to continue from, and `loader.has_more` says whether
  there is anything left. Persist the cursor and pass it back:

  ```python
  loader = JoboFeedLoader(max_jobs=500, cursor=saved_cursor)
  ```

  A cursor keeps the filters of the scan it came from, so `filters` are ignored when `cursor` is given. Use
  the same `max_jobs` and `batch_size` on every run of one scan to keep its batch size steady.
- **When the cursor moves.** `last_cursor` advances once every job of a batch has been yielded. If you stop
  iterating part-way through a batch, it still points at the start of that batch: resuming re-delivers the
  batch, and the feed does not bill again for jobs you received in the last 30 days. Dedupe on
  `Document.id`.
- **Filters** are the feed's own (`JoboFeedFilters`), not the search tool's: plural list names with
  `exclude_` twins (`work_models`, `exclude_work_models`, ...), structured `locations`
  (`{"country": ..., "region": ..., "city": ...}`), and `{"include": [...], "exclude": [...]}` objects for
  `companies`, `job_categories`, `job_subcategories`, `occupation_ids`, `industries`,
  `company_primary_industries`, `company_categories`, `company_sizes` and `company_revenues`, plus
  `company_is_agency`, `posted_after`, `updated_after`, `include_reactivated` and `description_format`.
  Unknown names are rejected: the feed itself ignores them silently, and a mistyped filter would stream an
  unfiltered feed.
- Also on the loader after a pass: `jobs_loaded`, `jobs_billed` (the API's `X-Jobs-Billed` count),
  `estimated_total` and `warnings`.
- `alazy_load()` and `aload()` are the async equivalents.

Use the retriever or the search tool to answer a question. Use the loader to copy jobs into your own index.

## Cost

Jobo bills per job, not per request. These figures are from the
[billing docs](https://jobo.world/docs/billing), which are the source of truth:

- **Search** (`JoboSearchTool`, `JoboRetriever`) bills every job a response returns, including a job an
  earlier search already returned. An empty result costs nothing.
- **Feed** (`JoboFeedLoader`) bills a job you have not received through Feed, Export or Outbound in the
  last 30 days. A job you already received is delivered again at no charge.
- A Job Search plan's included jobs are used first, then the prepaid wallet at the pay-as-you-go rate of
  $3.00 per 1,000 jobs. The Unlimited plan has no per-job charge on Search or Feed.
- `GET /api/jobs/{id}` (`JoboJobDetailsTool`) is free, and so is the company lookup. Both still count
  toward rate limits.
- Before a billed request runs, the API checks that included jobs plus the wallet cover a *full* page
  (`page_size` on search, `batch_size` on the feed) and answers `402` if they do not. A smaller page needs
  less cover.

What this package does about it:

- The search tool asks for 5 jobs by default and caps every call at `max_page_size`.
- The retriever returns `k` jobs (default 5).
- The feed loader will not run without an explicit `max_jobs`.
- `402` is never retried. Neither is a read timeout: a page may already have been delivered.
- Search tool artifacts carry the `usage` the API reported for the call.

An agent that searches in a loop spends per job returned on every call, so keep `max_page_size` low and
say in your prompt when to stop searching.

## Errors and retries

Requests that fail with `429`, `500` or `503`, or that cannot connect, are retried up to `max_retries`
times (default 2). `Retry-After` is honoured when the API sends it; otherwise the wait backs off
exponentially. A `Retry-After` longer than 60 seconds is not slept through: the error is raised with
`retry_after` set.

`402` (insufficient credits) and `409` (for example `feed_cursor_restart_required`) are never retried.

| Exception | Status | Notes |
| --- | --- | --- |
| `JoboValidationError` | 400 | Includes `invalid_cursor` on the feed |
| `JoboAuthenticationError` | 401 | Missing, invalid or expired key |
| `JoboInsufficientCreditsError` | 402 | `credits_required`, `credits_balance`, `top_up_url`, `upgrade_url` |
| `JoboPermissionError` | 403 | |
| `JoboNotFoundError` | 404 | Unknown id, or `not_available_in_sandbox` |
| `JoboConflictError` | 409 | |
| `JoboFeedCursorRestartRequiredError` | 409 | Discard the cursor and start a new scan; `replacement_request` |
| `JoboRateLimitError` | 429 | `retry_after` |
| `JoboServerError` | 5xx | |
| `JoboConnectionError` | | No HTTP response (DNS, TLS, socket, timeout) |

All derive from `JoboError`; the HTTP ones derive from `JoboAPIError` and carry `status_code`, the stable
`code` when the API sends one, `detail` and the parsed `body`. The retriever and the loader raise these
directly. The tools wrap them in `ToolException` (the original is the `__cause__`).

```python
from langchain_jobo import JoboInsufficientCreditsError, JoboRetriever

try:
    docs = JoboRetriever(k=5).invoke("data engineer")
except JoboInsufficientCreditsError as error:
    print("Top up:", error.top_up_url)
```

## Configuration

Every component accepts:

| Argument | Default | |
| --- | --- | --- |
| `api_key` | `JOBO_API_KEY` | |
| `base_url` | `https://connect.jobo.world` | |
| `timeout` | 30 seconds (120 on the feed loader) | Response timeout |
| `max_retries` | 2 | Retries after the first attempt |
| `http_client` | | An `httpx.Client` to send sync requests with |
| `http_async_client` | | An `httpx.AsyncClient` to send async requests with |

Every request carries `X-Jobo-Client: langchain` and a `langchain-jobo/<version>` user agent.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e . --group dev

.venv/bin/pytest          # no network, no API key: the HTTP layer is mocked
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
```

The test suite runs LangChain's standard tool and retriever tests (`langchain-tests`) against a mocked
API, and a drift test that compares the tool schema with the shared connector filter contract when this
package is checked out inside the Jobo connectors monorepo.

## License

MIT. See [LICENSE](LICENSE).
