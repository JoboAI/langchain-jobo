# Changelog

## 0.1.0

First release: the Jobo jobs API as LangChain components.

### Added

- **`JoboSearchTool`** (`jobo_job_search`): structured job search over `POST /api/jobs/search`, with every
  filter of the shared Jobo connector contract, including the include/exclude pairs (`companies`, `skills`,
  `industries`, `company_categories`, `job_categories`, `job_subcategories`, `occupation_ids`). Returns
  compact, citeable text and the raw jobs as the tool artifact. `page_size` defaults to 5 and is capped at
  `max_page_size` (default 25), because search is billed per job returned.
- **`JoboJobDetailsTool`** (`jobo_job_details`): one listing by id as self-contained prose, from the free
  `GET /api/jobs/{id}`.
- **`JoboCompanySearchTool`** (`jobo_company_search`): resolves a company name or domain prefix to the
  company ids the search filter matches exactly.
- **`JoboRetriever`**: query to job `Document`s, with constructor-level filters and `k`.
- **`JoboFeedLoader`**: `lazy_load()` / `alazy_load()` over `POST /api/jobs/feed` with cursor paging, a
  required `max_jobs` budget, and `last_cursor` / `has_more` for resuming.
- **`get_jobo_tools()`**: the three tools in one call.
- Sync and async support throughout, on `httpx`.
- Retries for `429`, `500` and `503` that honour `Retry-After`; `402` and `409` are never retried.
- Typed exceptions (`JoboInsufficientCreditsError`, `JoboRateLimitError`,
  `JoboFeedCursorRestartRequiredError`, ...) under `JoboError`.
- `X-Jobo-Client: langchain` on every request.
