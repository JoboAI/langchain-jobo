# Changelog

## 0.1.1

### Added

- **`iter_expired_job_ids()`** and **`aiter_expired_job_ids()`**: the ids of closed jobs from the free
  `GET /api/jobs/expired`, without a `JoboFeedLoader`. They take the key and connection settings and close
  the connection they open. The loader's `expired_job_ids()` methods share their code.

### Fixed

- **The feed loader no longer stops early on a scan that serves smaller pages.** A cursor keeps the
  smallest batch size its scan has used, so after a resume with a small `batch_size`, a later run with a
  large `max_jobs` got small pages, saw less room than one planned batch and stopped after the first page.
  When a page comes back shorter than asked for while the feed has more (the API warns
  `batch_size_capped`), the loader now plans with the scan's size, asks for that size, and keeps going while
  a whole page fits under `max_jobs`.
- **Search is not replayed after a `500`.** `POST /api/jobs/search` bills every job it returns and takes
  the charge before it builds the response, so a `500` can follow a charge and a retry billed the jobs
  again. Search now retries only `429` and a `503` with `Retry-After`.
- **A `503` without `Retry-After` is not retried**, on any request. The API sends it when a search ran out
  of time, which fails the same way again and held a search thread on the server each time.
- **`expired_since` older than 7 days is moved forward instead of failing.** The API answers `400` beyond
  7 days, so the README's `drop_closed_jobs(last_sync)` recipe failed whenever a weekly sync ran late. The
  value is now moved to just under 7 days ago, with a warning on the `langchain_jobo` logger: jobs that
  closed before then are not listed.
- **Search asks for no facets** (`include_facets: []`). Nothing reads them, and the API computed its
  default facets on every search.

### Changed

- The `language` filter's description says that names such as `English` work as well as ISO 639-1 codes,
  and the contract drift test checks that every language code of the shared contract is accepted.
- The README says that `JoboRetriever` sends its query as keywords, not as a question to turn into
  filters, and uses `iter_expired_job_ids()` in the closed-jobs recipe.

## 0.1.0

First release: the Jobo jobs API as LangChain components.

### Added

- **`JoboSearchTool`** (`jobo_job_search`): structured job search over `POST /api/jobs/search`, with every
  filter of the shared Jobo connector contract, including the include/exclude pairs (`companies`, `skills`,
  `industries`, `company_categories`, `job_categories`, `job_subcategories`, `occupation_ids`). Returns
  compact, citeable text and the raw jobs as the tool artifact.
  - `query` is the first argument, and `location` takes one place or a list.
  - `page_size` defaults to 5 and is capped at `max_page_size` (default 25), because search is billed per
    job returned. The schema states the cap as the maximum, and a larger `page_size` is reduced to it with a
    note instead of failing the call.
  - `company_categories` and `exclude_company_categories` are a closed set, like the other fixed-value
    filters: an unknown value is rejected before any request.
  - A single string given to a list of names (`companies`, `industries`, `job_categories`,
    `job_subcategories` and their `exclude_` twins) is one value and is never split on commas, so
    `"FedEx, Inc."` stays one company. Ids, codes, fixed values and skills are still split.
  - `description_format` is an option of the tool, not an argument the model sees: it only changes the
    artifact.
- **`JoboJobDetailsTool`** (`jobo_job_details`): one listing by id as self-contained prose, from the free
  `GET /api/jobs/{id}`. An id the API reports as unknown, or one that is not a UUID, is a plain answer for
  the model. Any other `404` raises, so a wrong `base_url` is not read as "no such job".
- **`JoboCompanySearchTool`** (`jobo_company_search`): resolves a company name or domain prefix to the
  company ids the search filter matches exactly.
- **`JoboFilterValuesTool`** (`jobo_filter_values`): the values a search filter accepts, with live job
  counts, for `sources`, `industries`, `skills`, `company_categories`, `job_categories`,
  `job_subcategories` and `occupations`. Occupations are searched by name in the occupation taxonomy.
- **`JoboLocationSearchTool`** (`jobo_location_search`): a place name to the locations the search takes.
- **`JoboRetriever`**: query to job `Document`s, with constructor-level `filters` and `k`.
  - `invoke(query, k=..., filters=..., page=...)` overrides them for one call; per-call filters are laid
    over the retriever's own. An unknown per-call keyword is a `TypeError`.
  - API warnings (for example a `companies` value that matched no company, which returns no jobs) are
    logged on the `langchain_jobo` logger and kept on `last_warnings`.
- **`JoboFeedLoader`**: `lazy_load()` / `alazy_load()` over `POST /api/jobs/feed` with cursor paging, a
  required `max_jobs` budget, and `last_cursor` / `has_more` for resuming.
  - `expired_job_ids()` (with `lazy_`, `a` and `alazy_` variants) reads the free `GET /api/jobs/expired`,
    to drop closed jobs from an index built with the loader.
  - A feed location without a country is rejected when the loader is built.
- **`get_jobo_tools()`**: the five tools in one call, sharing one connection pool. Every parameter is
  named; `include_lookups=False` leaves out the three lookup tools.
- **Document metadata**: `source` is the listing URL, following the LangChain convention, and `ats` is the
  applicant tracking system the listing was collected from.
- **Typing**: `api_key` accepts a `str` or a `SecretStr`. Filters are pydantic models
  (`JoboSearchFilters`, `JoboFeedFilters`, `IncludeExclude`, `FeedLocation`, `CompanyCategoryFilter`,
  `CompanySizeFilter`, `CompanyRevenueFilter`) or the same keys as typed dicts (`JoboSearchFiltersDict`,
  `JoboFeedFiltersDict`, ...). The fixed value sets are exported as `Literal` aliases (`WorkModel`,
  `EmploymentType`, `ExperienceLevel`, `CompanyCategory`, `CompanySize`, `CompanyRevenue`, `SearchSort`,
  `DescriptionFormat`, `FilterFacet`). The README's examples pass `mypy --strict` and `pyright`.
- **Fail early**:
  - A missing or blank API key raises `JoboConfigurationError` when a component is built, with where to
    create a key.
  - `JoboFeedLoader` refuses a sandbox key (`jbe_test_`) when it is built: the sandbox does not serve the
    feed.
  - An unknown constructor argument, such as `JoboSearchTool(max_pagesize=3)` or `JoboRetriever(K=2)`, is
    rejected instead of ignored.
- **Sandbox keys**: the three lookup tools answer the model with a plain note when the sandbox refuses them
  (`404 not_available_in_sandbox`) instead of raising, so `get_jobo_tools()` works with a sandbox key.
- Sync and async support throughout, on `httpx`. A component reuses one connection pool, and has `close()`,
  `aclose()` and context-manager support.
- Retries for `429`, `500` and `503` that honour `Retry-After`; `402` and `409` are never retried.
- Typed exceptions (`JoboInsufficientCreditsError`, `JoboRateLimitError`,
  `JoboFeedCursorRestartRequiredError`, ...) under `JoboError`. A `400`'s per-field `errors` are part of
  the message and on `JoboValidationError.errors`; a `429`'s message says how many seconds to wait.
- `X-Jobo-Client: langchain` on every request.
