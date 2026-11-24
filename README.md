# langchain-jobo

LangChain integration for the [Jobo](https://jobo.world) jobs API: live job listings collected directly
from employer career sites and applicant tracking systems, as LangChain tools, a retriever and a document
loader.

| Component | What it does | Jobo endpoint | Billed |
| --- | --- | --- | --- |
| `JoboSearchTool` (`jobo_job_search`) | Search jobs with the full filter set | `POST /api/jobs/search` | Per job returned |
| `JoboJobDetailsTool` (`jobo_job_details`) | One listing as readable prose | `GET /api/jobs/{id}` | Free |
| `JoboCompanySearchTool` (`jobo_company_search`) | Company name or domain to company ids | `GET /api/connectors/companies/suggest` | Free |
| `JoboFilterValuesTool` (`jobo_filter_values`) | The values a filter accepts, with job counts | `GET /api/connectors/filter-options/{facet}`, `GET /api/occupations` | Free |
| `JoboLocationSearchTool` (`jobo_location_search`) | A place name to the locations the search takes | `GET /api/connectors/locations/suggest` | Free |
| `JoboHiringSignalsTool` (`jobo_hiring_signals`) | Hiring signals: events at companies, such as a hiring surge or a first job in a country | `GET /api/signals`, `GET /api/companies/{id}/signals` | Per signal returned, at ten times a job |
| `JoboRetriever` | Query to job `Document`s | `POST /api/jobs/search` | Per job returned |
| `JoboFeedLoader` | Stream the job feed as `Document`s, and list closed jobs | `POST /api/jobs/feed`, `GET /api/jobs/expired` | Per new job; closed-job ids are free |
| `iter_expired_job_ids()` | The ids of closed jobs, without a loader | `GET /api/jobs/expired` | Free |
| `get_jobo_tools()` | The five job tools in one call, and the hiring signals tool when you ask for it | | |

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

Every component reads `JOBO_API_KEY`, or takes `api_key=...` as a plain `str` or a pydantic `SecretStr`.
Either way the key is held as a `SecretStr`: it is masked in `repr()` and model dumps, and is only ever
written into the `X-Api-Key` request header.

A missing or blank key is an error when the component is built, not on its first request:
`JoboConfigurationError`, with a message that says where to create a key.

### Sandbox keys

A sandbox key (`jbe_test_...`) is free and returns fictional jobs in the real response shape. The sandbox
serves job search and job lookups, and nothing else this package calls:

| Component | With a sandbox key |
| --- | --- |
| `JoboSearchTool`, `JoboJobDetailsTool`, `JoboRetriever` | Work |
| `JoboCompanySearchTool`, `JoboFilterValuesTool`, `JoboLocationSearchTool` | The sandbox answers `404 not_available_in_sandbox`. The tool does not raise: it tells the model the lookup is unavailable and what to do instead, so an agent run carries on |
| `JoboHiringSignalsTool` | The sandbox answers `404 not_available_in_sandbox`. The tool does not raise: it tells the model that hiring signals need a live key and not to call it again |
| `JoboFeedLoader`, `iter_expired_job_ids()` | Refused when it is built or called, with `JoboConfigurationError` |

So `get_jobo_tools()` is safe to use with a sandbox key. `get_jobo_tools(include_lookups=False)` leaves the
three lookups out. See the [sandbox docs](https://jobo.world/docs/sandbox).

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
from langchain_jobo import JoboSearchTool

tool = JoboSearchTool()
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

Arguments, in the order the model is shown them: `query`, `location`, `sources`, `companies`,
`exclude_companies`, `work_model`, `employment_type`, `experience_level`, `language`, `skills`,
`exclude_skills`, `industries`, `exclude_industries`, `company_categories`, `exclude_company_categories`,
`job_categories`, `exclude_job_categories`, `job_subcategories`, `exclude_job_subcategories`,
`occupation_ids`, `exclude_occupation_ids`, `min_salary_usd`, `max_salary_usd`, `posted_after`,
`posted_before`, `discovered_after`, `discovered_before`, `search_description`, `sort`, `page` and
`page_size`.

- **Fixed-value filters** take lowercase keys: `work_model` is `remote`, `hybrid` or `onsite`;
  `employment_type` is `full-time`, `part-time`, `contract`, `internship`, `freelance` or `temporary`;
  `experience_level` is `intern`, `entry`, `mid`, `senior`, `lead` or `executive`; `company_categories` is
  `b2b`, `b2c`, `b2b2c`, `b2g`, `saas`, `marketplace`, `e-commerce`, `service-provider`, `hardware`,
  `agency`, `nonprofit`, `government` or `education`. An unknown value is rejected before any request is
  made, because the API would answer it with zero results rather than an error.
- **Open-value filters** (`sources`, `skills`, `industries`, `job_categories`, `job_subcategories`,
  `occupation_ids`, `companies`) match only values the index holds. Their descriptions point the model at
  `jobo_filter_values` and `jobo_company_search`, which look the values up for free.
- **`location`** is one place or a list of places; a list matches jobs in any of them.
- **`language`** takes ISO 639-1 codes (`en`, `de`). Names such as `English` also work.
- **List arguments take a list.** A single string is also accepted, because models send one. For ids,
  codes, fixed values and skills it is split on commas. For names (`companies`, `industries`,
  `job_categories`, `job_subcategories` and their `exclude_` twins) it is one value, never split:
  `"FedEx, Inc."` is one company.
- **`page_size`** defaults to 5. The tool never requests more than `max_page_size` jobs per call (default
  25, at most 100, the API's own limit). The schema the model is given states that cap as `page_size`'s
  maximum, and a larger number is reduced to it, with a note in the answer, instead of failing the call:

```python
from langchain_jobo import JoboSearchTool

tool = JoboSearchTool(max_page_size=10)
```

`description_format` (`markdown`, `html`, `enhanced` or `enhanced_html`) is an option of the tool, not an
argument the model can set: it only changes the descriptions in the artifact, which the model never reads.

```python
from langchain_jobo import JoboSearchTool

tool = JoboSearchTool(description_format="html")
```

### `JoboJobDetailsTool`

```python
from langchain_jobo import JoboJobDetailsTool

details = JoboJobDetailsTool()
print(details.invoke({"job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}))
```

Returns the listing as self-contained prose: company, location, compensation, responsibilities,
qualifications, benefits, the employer's description, and the listing and apply URLs. The raw job object is
the artifact. Pass `include_description=False` for the structured fields only.

An id the API says does not exist, or one that is not a UUID at all, comes back as a plain "no job exists
with id ..." answer rather than an error. Any other `404` raises: it means the request did not reach the
job endpoint (a wrong `base_url`, for example), and must not be read as "that job is gone".

### `JoboCompanySearchTool`

```python
from langchain_jobo import JoboCompanySearchTool

companies = JoboCompanySearchTool()
print(companies.invoke({"query": "stripe"}))
```

The search filter `companies` matches a company *name* whole (`Glenmark` does not find `Glenmark
Pharmaceuticals`), and a name also matches every other company called that. This tool looks companies up by
the start of their name or website domain and returns their ids, which match exactly.

### `JoboFilterValuesTool`

```python
from langchain_jobo import JoboFilterValuesTool

values = JoboFilterValuesTool()
print(values.invoke({"facet": "industries", "query": "soft"}))
print(values.invoke({"facet": "occupations", "query": "nurse"}))
```

Lists the values a search filter accepts, each with its number of live jobs. `facet` is one of `sources`,
`industries`, `skills`, `company_categories`, `job_categories`, `job_subcategories` and `occupations` (the
ids the `occupation_ids` filter takes). `query` narrows the list and is optional; without it the most common
values come back. `limit` is 1-50 and defaults to 20.

For `occupations`, `query` is searched in the occupation *names* (and their category and subcategory), so
`"nurse"` finds `Registered Nurse` and its id. That search reads the occupation taxonomy
(`GET /api/occupations`), which the tool keeps for ten minutes.

### `JoboLocationSearchTool`

```python
from langchain_jobo import JoboLocationSearchTool

places = JoboLocationSearchTool()
print(places.invoke({"query": "Springfield"}))
```

Returns matching cities, regions and countries, written the way the `location` filter takes them. The
search resolves a free-form location by itself, so this is for an ambiguous name, or for a search that
reports a location it could not resolve.

### `JoboHiringSignalsTool`

A hiring signal is one event at one company, detected from its job postings: a hiring surge, a first job
in a new country or function, a leadership search, a new technology, a slowdown or a hiring freeze. This is
the tool for a sales, recruiting or investor research agent: "which fintechs in Germany started hiring a
sales team this month?", "is acme.com growing or freezing?". See the
[hiring signals guide](https://jobo.world/docs/guides/hiring-signals) for the types and filters.

```python
from langchain_jobo import JoboHiringSignalsTool

signals = JoboHiringSignalsTool(max_signals=10)

print(
    signals.invoke(
        {
            "types": ["function_surge", "new_function"],
            "functions": ["Sales"],
            "countries": ["DE"],
            "industries": ["Financial Services"],
            "detected_after": "2026-11-01",
        }
    )
)
```

```text
1 hiring signal, newest detection day first. No more match.

1. Ledgerly (ledgerly.example) - Function surge
   12 new Sales jobs in 14 days vs 2.5 normally
   Detected: 2026-11-18 | Strength: Strong (71/100)
   Company ID: 6c1f0a9e-3b7d-4c55-8e21-5a4d9f0b7c33
   Evidence jobs: a41c7e02-58b1-4d0f-b6c9-0e2f7d5a1c88, 0b9e6a37-2c4d-4f1e-8a5b-9d3c7e1f2a46

Evidence jobs are the postings behind a signal: read one free with jobo_job_details.

Billed: 1 hiring signal (0 of 250 included signals left this month; 30 credits charged to the wallet; wallet balance 4,970 credits). Reading the same signals again is billed again.
```

Each signal comes with its company and domain, its type, the one-line summary Jobo's dashboard shows for
it, the day it was detected, its strength (1 to 100, comparable within one type) as a band and a number,
and the ids of up to five jobs behind it, which `jobo_job_details` reads for free. The last line says what
the call was billed, from the response's metering headers. A type added to the API after this release is
shown by its name, never an error. Called with a `ToolCall`, the artifact holds `signals` (the raw signal
objects, `details` and `focus` included), `next_cursor`, `has_more` and `usage`.

Arguments: `types`, `domains`, `company_id`, `countries`, `company_sizes`, `industries`, `funding_stages`,
`functions`, `roles`, `seniority`, `technologies`, `skills`, `job_countries`, `min_strength`,
`detected_after`, `detected_before`, `created_after`, `exclude_agencies`, `limit` and `cursor`.

- **Filters combine with AND**; the values of one list match any of them. `types`, `seniority` and
  `company_sizes` are closed sets, checked before any request.
- **A value cannot contain a comma.** The API separates list values with commas, so a value with one is
  refused before the request instead of being read as two. A single string is split on commas for codes
  (`domains`, `countries`, `technologies`, `skills`, ...) and kept whole for names (`industries`,
  `funding_stages`, `functions`, `roles`). A list takes at most 100 values.
- **`company_id`** reads one company from `GET /api/companies/{id}/signals`, which takes only `types`,
  `min_strength`, `detected_after`, `detected_before` and `created_after`. Any other filter beside it is
  refused with a message that names it, because that endpoint would ignore it. To name companies by
  website, use `domains`.
- **`limit`** defaults to 5. The tool never asks for more than `max_signals` (default 25, at most 200, the
  API's own limit), and a larger number is reduced to it with a note in the answer.
- **One call is one request.** The tool never pages on its own: when more signals match, the answer gives
  the model the `cursor` to ask for the next page.

The other filters of the API (`exclude_domains`, `categories`, `founded_after`, `technology_categories`,
`job_regions`, `job_cities`, `min_new_jobs`, `min_days_open`, `reposted_only`) are left off the tool.
`langchain_jobo.signals.build_signals_request()` takes them all, for your own code.

The tool is not in `get_jobo_tools()` unless you ask for it: every signal it returns is billed at ten times
the price of a job, and a signal read again is billed again.

```python
from langchain_jobo import get_jobo_tools

tools = get_jobo_tools(include_hiring_signals=True, max_signals=10)
```

Hiring signals need a live account API key: a key that belongs to no account gets `403
account_key_required`, and the tool tells the model to stop. `langchain_jobo.signals.describe_signal()`
writes the one-line summary for a signal you hold.

### All five at once

```python
from langchain_jobo import get_jobo_tools

tools = get_jobo_tools(max_page_size=10)
```

| Parameter | Default | |
| --- | --- | --- |
| `api_key` | `JOBO_API_KEY` | The only one that can also be passed by position |
| `max_page_size` | 25 | Cap on jobs the search tool requests per call |
| `include_lookups` | `True` | `False` returns the search and details tools only |
| `include_hiring_signals` | `False` | `True` adds `JoboHiringSignalsTool`, last. Billed per signal at ten times a job |
| `max_signals` | 25 | Cap on signals the hiring signals tool requests per call, at most 200 |
| `handle_tool_error` | `False` | Hand API errors to the model instead of raising (see [Agent](#agent)) |
| `handle_validation_error` | `False` | The same for arguments the model got wrong |
| `base_url`, `timeout`, `max_retries`, `http_client`, `http_async_client` | | As in [Configuration](#configuration), applied to every tool |

Every parameter is named, so a mistyped one is a `TypeError` and not a silently ignored keyword. The tools
share one connection pool.

## Agent

Install LangChain and the integration package for the chat model you use, and say which model to run:

```bash
pip install -U langchain langchain-jobo
pip install -U langchain-openai   # or langchain-anthropic, or another chat model integration
export AGENT_MODEL="<provider>:<model>"
```

`AGENT_MODEL` is this example's own variable, not one LangChain or Jobo reads. `create_agent` takes the
model as a `provider:model` string, the form LangChain's `init_chat_model` understands, or as a chat model
instance.

```python
import os

from langchain.agents import create_agent
from langchain_jobo import get_jobo_tools

agent = create_agent(
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
failure should not be retried by the model. `handle_validation_error=True` does the same for arguments the
model got wrong, such as an unknown `work_model`.

## Retriever

```python
from langchain_jobo import JoboRetriever, JoboSearchFilters

retriever = JoboRetriever(
    k=5,
    filters=JoboSearchFilters(
        location="London",
        work_model=["remote", "hybrid"],
        exclude_companies=["Acme Staffing"],
    ),
)

for doc in retriever.invoke("staff data engineer"):
    print(doc.metadata["title"], "-", doc.metadata["company"], doc.metadata["source"])
```

- **The query is sent as keywords.** It is matched as words against job titles, companies, skills and
  summaries, not read as a question: `"remote senior nursing jobs in Ohio paying over $90k"` looks for those
  words, and is not turned into filters. Pass the keywords (`"registered nurse"`) as the query and put the
  rest in `filters=`, or give an agent `JoboSearchTool`, where the model fills in the filters itself.
- `filters` takes the same filter names as the search tool, plus `description_format`, and applies them to
  every query. Pass a `JoboSearchFilters`, or the same keys as a dict: `filters={"location": "London"}`.
  Both are checked when the retriever is built, and unknown names are rejected. See [Typing](#typing).
- Per call, `invoke` takes `k`, `filters` and `page`. Per-call filters are laid over the retriever's own:
  a key you pass wins, the others stay, and `None` removes one. An unknown keyword is a `TypeError`.

  ```python
  from langchain_jobo import JoboRetriever

  retriever = JoboRetriever(k=5, filters={"work_model": ["remote"]})
  retriever.invoke("nurse", k=2)
  retriever.invoke("nurse", filters={"location": "Berlin"}, page=2)
  retriever.invoke("nurse", filters={"work_model": None})
  ```

- `page_content` is the same prose `JoboJobDetailsTool` returns. `include_description=False` leaves the
  employer's description out.
- `metadata` holds `id`, `source`, `url`, `apply_url`, `title`, `company`, `company_id`, `location`,
  `workplace_type`, `employment_type`, `experience_level`, `salary`, `salary_min`, `salary_max`,
  `salary_currency`, `salary_period`, `date_posted`, `created_at`, `updated_at`, `valid_through` and
  `ats`. `source` follows the LangChain convention and says where the document came from: it is the
  listing URL, the same value as `url`. `ats` is the applicant tracking system the listing was collected
  from, such as `greenhouse`. A key is present only when the posting has the value; nothing is filled in.
- `Document.id` is the Jobo job id.
- `k` is 1-100. Each query bills up to `k` jobs.
- **Warnings.** A filter value the API cannot use does not fail the search. It comes back as a warning,
  often with no jobs at all: a `companies` value that matches no company is the usual case. Each warning is
  logged on the `langchain_jobo` logger at `WARNING`, and the warnings of the most recent query are on
  `retriever.last_warnings`:

  ```python
  from langchain_jobo import JoboRetriever

  retriever = JoboRetriever(filters={"companies": ["Glenmark"]})
  docs = retriever.invoke("chemist")
  if not docs and retriever.last_warnings:
      print(retriever.last_warnings[0]["message"])  # No company matched: Glenmark.
  ```

## Document loader

```python
from langchain_jobo import IncludeExclude, JoboFeedFilters, JoboFeedLoader

loader = JoboFeedLoader(
    max_jobs=500,
    filters=JoboFeedFilters(
        sources=["greenhouse"],
        work_models=["remote"],
        job_categories=IncludeExclude(include=["Healthcare"]),
        exclude_employment_types=["internship"],
    ),
)

for doc in loader.lazy_load():
    print(doc.metadata["title"])

if loader.has_more:
    print("resume from:", loader.last_cursor)
```

`max_jobs` is required and has no default, because the feed streams every matching job and bills each one
that is new to you. Pass a number, or `max_jobs=None` to opt in to reading the whole feed. The loader needs
a live key: a sandbox key is refused when the loader is built.

- **Whole batches only.** The loader never requests more jobs than `max_jobs` allows, so nothing is
  delivered and billed beyond the budget. It plans equal batches of at most `batch_size` (default and
  maximum 1,000) that fit under `max_jobs`; when `max_jobs` is not a whole number of batches it stops
  slightly short (`max_jobs=2500` loads 2,499 in three batches of 833). It never asks for a smaller last
  batch, because the feed keeps the smallest batch size a scan has used for every later page.
- **Resuming.** `loader.last_cursor` is the cursor to continue from, and `loader.has_more` says whether
  there is anything left. Persist the cursor and pass it back:

  ```python
  from langchain_jobo import JoboFeedLoader

  first = JoboFeedLoader(max_jobs=500)
  documents = first.load()

  later = JoboFeedLoader(max_jobs=500, cursor=first.last_cursor)
  ```

  A cursor keeps the filters of the scan it came from, so `filters` are ignored when `cursor` is given. Use
  the same `max_jobs` and `batch_size` on every run of one scan to keep its batch size steady. A cursor
  also keeps the smallest batch size its scan has used: when a resumed scan serves smaller pages than this
  run planned (the API warns `batch_size_capped`), the loader reads pages of the scan's size, as many as fit
  under `max_jobs`.
- **When the cursor moves.** `last_cursor` advances once every job of a batch has been yielded. If you stop
  iterating part-way through a batch, it still points at the start of that batch: resuming re-delivers the
  batch, and the feed does not bill again for jobs you received in the last 30 days. Dedupe on
  `Document.id`.
- **Filters** are the feed's own (`JoboFeedFilters`), not the search tool's: plural list names with
  `exclude_` twins (`work_models`, `exclude_work_models`, ...), structured `locations`
  (`FeedLocation(country=..., region=..., city=...)`), and `IncludeExclude(include=[...], exclude=[...])`
  for `companies`, `job_categories`, `job_subcategories`, `occupation_ids`, `industries` and
  `company_primary_industries`, with `CompanyCategoryFilter`, `CompanySizeFilter` and `CompanyRevenueFilter`
  for `company_categories`, `company_sizes` and `company_revenues`, plus `company_is_agency`,
  `posted_after`, `updated_after`, `include_reactivated` and `description_format`. The same keys work as a
  dict (`{"job_categories": {"include": ["Healthcare"]}}`). Unknown names are rejected: the feed itself
  ignores them silently, and a mistyped filter would stream an unfiltered feed.
- **Locations need a country.** `FeedLocation(city="Columbus")` is rejected when the loader is built, as
  the API would reject it once called. `FeedLocation(country="Remote")` matches remote jobs.
- Also on the loader after a pass: `jobs_loaded`, `jobs_billed` (the API's `X-Jobs-Billed` count),
  `estimated_total` and `warnings`. Warnings are logged on the `langchain_jobo` logger as well.
- `alazy_load()` and `aload()` are the async equivalents.

Use the retriever or the search tool to answer a question. Use the loader to copy jobs into your own index.

### Dropping closed jobs from your index

An index built from the feed keeps jobs that have since closed. `iter_expired_job_ids()` yields the ids of
jobs that expired, from the free `GET /api/jobs/expired`. They are the `Document.id`s the loader yields, so
they can go straight to a vector store's `delete`. It needs a live key, and no loader:

```python
from datetime import datetime

from langchain_core.vectorstores import VectorStore
from langchain_jobo import iter_expired_job_ids


def drop_closed_jobs(vector_store: VectorStore, last_sync: datetime) -> int:
    """Delete the jobs that closed since `last_sync` from the index."""
    closed = list(iter_expired_job_ids(expired_since=last_sync))
    if closed:
        vector_store.delete(ids=closed)
    return len(closed)
```

`iter_expired_job_ids()` takes `api_key`, `batch_size` (1-10,000 ids per request), `base_url`, `timeout`,
`max_retries` and `http_client`, and closes the connection it opened once the ids run out.
`aiter_expired_job_ids()` is the async equivalent (`async for`, with `http_async_client`). A loader you
already have does the same over its own connection: `loader.expired_job_ids()`, `lazy_expired_job_ids()`,
`aexpired_job_ids()` and `alazy_expired_job_ids()`. The call is never billed and is not counted against
`max_jobs`.

`expired_since` defaults to the last 24 hours, and the API goes back 7 days at most. An older value, from a
weekly sync that ran late for example, is moved forward to just under 7 days ago instead of failing, and a
warning is logged on the `langchain_jobo` logger: jobs that closed before then are not listed. Sync at
least once a week, or rebuild the index, if every closed job must go.

## Typing

The package is typed (`py.typed`) and its documented calls pass `mypy --strict` and `pyright`.

- **Filters come in two spellings.** `JoboSearchFilters` and `JoboFeedFilters` are pydantic models, with
  `IncludeExclude`, `FeedLocation`, `CompanyCategoryFilter`, `CompanySizeFilter` and
  `CompanyRevenueFilter` for the nested feed filters. `JoboSearchFiltersDict` and `JoboFeedFiltersDict`
  type the same keys as plain dicts (with `IncludeExcludeDict`, `FeedLocationDict`,
  `CompanyCategoryFilterDict`, `CompanySizeFilterDict` and `CompanyRevenueFilterDict`), so an editor
  completes the keys and values of `filters={...}` and a type checker catches a misspelt one.
- **Fixed value sets are `Literal` aliases**: `WorkModel`, `EmploymentType`, `ExperienceLevel`,
  `CompanyCategory`, `CompanySize`, `CompanyRevenue`, `SearchSort`, `DescriptionFormat`, `FilterFacet`,
  and for hiring signals `SignalType` and `SignalSeniority`.
- Everything above is importable from `langchain_jobo`.

```python
from langchain_jobo import JoboRetriever, JoboSearchFiltersDict, WorkModel

preferred: list[WorkModel] = ["remote", "hybrid"]
filters: JoboSearchFiltersDict = {"work_model": preferred, "sort": "discovered_desc"}
retriever = JoboRetriever(filters=filters)
```

A dict of unknown shape, loaded from configuration for example, is not a typed dict to a type checker.
Validate it into a model first: `JoboSearchFilters.model_validate(config)`.

Constructor arguments are checked at run time too. An unknown one, such as `JoboSearchTool(max_pagesize=3)`
or `JoboRetriever(K=2)`, raises a pydantic `ValidationError` instead of being ignored. If you subclass a
component and add fields, give the subclass an `__init__` (or pass `api_key` as a `SecretStr`): a type
checker builds the constructor of a pydantic subclass from its fields, where `api_key` is a `SecretStr`.

## Cost

Jobo bills per job, not per request. These figures are from the
[billing docs](https://jobo.world/docs/billing), which are the source of truth:

- **Search** (`JoboSearchTool`, `JoboRetriever`) bills every job a response returns, including a job an
  earlier search already returned. An empty result costs nothing.
- **Feed** (`JoboFeedLoader`) bills a job you have not received through Feed, Export or Outbound in the
  last 30 days. A job you already received is delivered again at no charge.
- A Job Search plan's included jobs are used first, then the prepaid wallet at the pay-as-you-go rate of
  $3.00 per 1,000 jobs. The Unlimited plan has no per-job charge on Search or Feed.
- **Hiring signals** (`JoboHiringSignalsTool`) bill every signal a response returns, at ten times the price
  of a job, and a signal read again is billed again. A plan's included signals are used first, then the
  wallet; the Unlimited plan has no per-signal charge. A request that returns no signals is free. See the
  [hiring signals guide](https://jobo.world/docs/guides/hiring-signals#pricing).
- `GET /api/jobs/{id}` (`JoboJobDetailsTool`) is free, and so are the company, filter-value and location
  lookups and the expired-job ids. They still count toward rate limits.
- Before a billed request runs, the API checks that included jobs plus the wallet cover a *full* page
  (`page_size` on search, `batch_size` on the feed) and answers `402` if they do not. A smaller page needs
  less cover.

What this package does about it:

- The search tool asks for 5 jobs by default and caps every call at `max_page_size`.
- The hiring signals tool is only in `get_jobo_tools()` with `include_hiring_signals=True`. It asks for 5
  signals by default, caps every call at `max_signals`, never pages on its own, and ends every answer with
  what the call was billed.
- The retriever returns `k` jobs (default 5).
- The feed loader will not run without an explicit `max_jobs`.
- `402` is never retried. Neither is a read timeout, nor a `500` on search or hiring signals: a page may
  already have been delivered and billed.
- Search and hiring signals tool artifacts carry the `usage` the API reported for the call.

An agent that searches in a loop spends per job returned on every call, so keep `max_page_size` low and
say in your prompt when to stop searching.

## Errors and retries

Requests that fail with `429`, `500` or `503`, or that cannot connect, are retried up to `max_retries`
times (default 2). `Retry-After` is honoured when the API sends it; otherwise the wait backs off
exponentially. A `Retry-After` longer than 60 seconds is not slept through: the error is raised with
`retry_after` set.

`402` (insufficient credits) and `409` (for example `feed_cursor_restart_required`) are never retried.
Neither are:

- a `503` without `Retry-After`. The API sends one when a search ran out of time ("Search took too long"),
  which fails the same way again;
- a `500` on search (`JoboSearchTool`, `JoboRetriever`) or on hiring signals (`JoboHiringSignalsTool`).
  Both take their charge before they build the response, so a `500` can follow a charge and a replay would
  bill again. They retry only `429` and a `503` with `Retry-After`, which come before any work. A feed
  request is safe to replay, because a job delivered again is free.

The hiring signals tool adds what the model can do to the message of a `402` (ask for a smaller `limit`),
a `403` (stop: `account_key_required` means the key belongs to no account), a `429` (wait) and a `400`
(correct the arguments).

| Exception | Status | Notes |
| --- | --- | --- |
| `JoboConfigurationError` | | Raised when a component is built: no API key, or a sandbox key given to `JoboFeedLoader` or `iter_expired_job_ids()` |
| `JoboValidationError` | 400 | `errors` holds the API's per-field messages, which are also in the exception message. Includes `invalid_cursor` on the feed |
| `JoboAuthenticationError` | 401 | Missing, invalid or expired key |
| `JoboInsufficientCreditsError` | 402 | `credits_required`, `credits_balance`, `top_up_url`, `upgrade_url` |
| `JoboPermissionError` | 403 | |
| `JoboNotFoundError` | 404 | Unknown id, or `not_available_in_sandbox` (`error.not_available_in_sandbox`) |
| `JoboConflictError` | 409 | |
| `JoboFeedCursorRestartRequiredError` | 409 | Discard the cursor and start a new scan; `replacement_request` |
| `JoboRateLimitError` | 429 | `retry_after`; the message says how many seconds to wait |
| `JoboServerError` | 5xx | |
| `JoboConnectionError` | | No HTTP response (DNS, TLS, socket, timeout) |

All derive from `JoboError`; the HTTP ones derive from `JoboAPIError` and carry `status_code`, the stable
`code` when the API sends one, `detail` and the parsed `body`. The retriever and the loader raise these
directly. The tools wrap them in `ToolException` (the original is the `__cause__`).

```python
from langchain_jobo import (
    JoboInsufficientCreditsError,
    JoboRetriever,
    JoboValidationError,
)

try:
    docs = JoboRetriever(k=5).invoke("data engineer")
except JoboInsufficientCreditsError as error:
    print("Top up:", error.top_up_url)
except JoboValidationError as error:
    print(error.errors)  # e.g. {"page_size": ["page_size must be between 1 and 100."]}
```

Mistakes the package can see are caught before any request, as a pydantic `ValidationError`: an unknown
filter name or fixed value, an unknown constructor argument, a feed location without a country.

## Configuration

Every component accepts:

| Argument | Default | |
| --- | --- | --- |
| `api_key` | `JOBO_API_KEY` | A `str` or a `SecretStr` |
| `base_url` | `https://connect.jobo.world` | |
| `timeout` | 30 seconds (120 on the feed loader) | Response timeout |
| `max_retries` | 2 | Retries after the first attempt |
| `http_client` | | An `httpx.Client` to send sync requests with |
| `http_async_client` | | An `httpx.AsyncClient` to send async requests with |

Every request carries `X-Jobo-Client: langchain` and a `langchain-jobo/<version>` user agent.

### Connections

A component keeps one HTTP connection pool and reuses it for every call, sync and async. Close it when you
are done with `close()` or `aclose()`, or use the component as a context manager:

```python
from langchain_jobo import JoboRetriever

with JoboRetriever(k=3) as retriever:
    docs = retriever.invoke("data engineer")
```

```python
import asyncio

from langchain_jobo import JoboSearchTool


async def main() -> None:
    async with JoboSearchTool() as tool:
        print(await tool.ainvoke({"query": "data engineer"}))


asyncio.run(main())
```

- A closed component can be used again: its next call opens a new pool.
- An `http_client` or `http_async_client` you pass in is yours. It is used as it is and never closed.
- An async pool belongs to the event loop it was opened on. A component called from several event loops
  keeps one per loop, and `aclose()` closes the one of the loop it is awaited on.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -U "pip>=25.1"   # --group (dependency groups) needs pip 25.1 or later
.venv/bin/pip install -e . --group dev

.venv/bin/pytest          # no network, no API key: the HTTP layer is mocked
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
```

With [uv](https://docs.astral.sh/uv/), which does not go through pip: `uv venv && uv pip install -e . --group dev`.

The test suite runs LangChain's standard tool and retriever tests (`langchain-tests`) against a mocked
API, and a drift test that compares the tool schema, the fixed value sets and the lookup facets with the
shared connector filter contract when this package is checked out inside the Jobo connectors monorepo.

## License

MIT. See [LICENSE](LICENSE).
