"""Test helpers: a recording mock of the Jobo API and sample payloads."""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import count
from typing import Any

import httpx

Handler = Callable[[httpx.Request], httpx.Response]

#: A live-looking key: the feed loader refuses a sandbox one.
TEST_KEY = "jbe_live_unit-test-key"
SANDBOX_KEY = "jbe_test_unit-test-key"


def make_job(index: int = 0, **overrides: Any) -> dict[str, Any]:
    """A complete job object in the API's `JobDto` shape."""
    job: dict[str, Any] = {
        "id": f"00000000-0000-4000-8000-{index:012d}",
        "title": f"Senior Backend Engineer {index}",
        "normalized_title": "Backend Engineer",
        "occupation": {
            "id": "jobo:backend-developer",
            "name": "Backend Developer",
            "category": "Technology",
            "subcategory": "Software Engineering",
            "isco_code": "2512",
            "onet_soc_codes": [],
        },
        "company": {
            "id": "11111111-1111-4111-8111-111111111111",
            "name": "Acme Corp",
            "website": "https://acme.example",
            "logo_url": None,
            "summary": "Acme builds rockets for coyotes.",
            "industries": ["Aerospace", "Software Development"],
            "categories": ["b2b"],
            "linkedin_url": None,
            "crunchbase_url": None,
            "details_url": None,
        },
        "description": "We are hiring a **backend engineer** to build the API.",
        "language": "en",
        "summary": "Build and run the public API.",
        "listing_url": f"https://jobs.acme.example/{index}",
        "apply_url": f"https://jobs.acme.example/{index}/apply",
        "locations": [
            {
                "location": "Berlin, Germany",
                "city": "Berlin",
                "region": "Berlin",
                "country": "Germany",
                "latitude": 52.52,
                "longitude": 13.405,
            }
        ],
        "compensation": {
            "min": 80000,
            "max": 100000,
            "currency": "EUR",
            "period": "yearly",
        },
        "employment_type": "Full-time",
        "workplace_type": "Hybrid",
        "experience_level": "Senior",
        "source": "greenhouse",
        "created_at": "2026-11-17T08:00:00Z",
        "updated_at": "2026-11-18T09:30:00Z",
        "date_posted": "2026-11-16T00:00:00Z",
        "valid_through": None,
        "qualifications": {
            "must_have": {
                "education": ["BSc in Computer Science"],
                "certifications": [],
                "skills": [{"name": "Python", "type": "hard"}],
            },
            "preferred": {
                "education": [],
                "certifications": ["AWS Solutions Architect"],
                "skills": [{"name": "Kubernetes", "type": "hard"}],
            },
        },
        "responsibilities": ["Design APIs", "Run the on-call rota"],
        "benefits": ["30 days of leave"],
        "is_work_auth_required": True,
        "is_h1b_sponsor": None,
        "is_clearance_required": False,
    }
    job.update(overrides)
    return job


def search_page(
    count: int, *, total: int | None = None, page: int = 1, **extra: Any
) -> dict[str, Any]:
    """A `JobSearchResponse` body holding `count` jobs."""
    return {
        "jobs": [make_job(index) for index in range(count)],
        "total": count if total is None else total,
        "page": page,
        "page_size": count,
        "total_pages": 1 if total is None else max(1, -(-total // max(count, 1))),
        "facets": {},
        **extra,
    }


class MockJobo:
    """Records every request and answers it with `handler`."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))
        self.async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(self._handle)
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Constructor arguments that point a component at this mock."""
        return {
            "api_key": TEST_KEY,
            "http_client": self.client,
            "http_async_client": self.async_client,
        }

    @property
    def bodies(self) -> list[Any]:
        """The JSON body of every request that had one."""
        return [json.loads(r.content) for r in self.requests if r.content]

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def search_handler(request: httpx.Request) -> httpx.Response:
    """Answer a search with exactly `page_size` jobs, as the API would."""
    body = json.loads(request.content) if request.content else {}
    size = int(body.get("page_size", 25))
    return httpx.Response(
        200,
        json=search_page(size, total=1234, page=int(body.get("page", 1))),
        headers={"X-Credits-Deducted": str(size * 3), "X-Credits-Balance": "4000"},
    )


def reply(
    status: int,
    body: Any = None,
    *,
    headers: dict[str, str] | None = None,
    text: str | None = None,
) -> Handler:
    """A handler that answers every request with a fresh copy of one response."""

    def handler(_request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    return handler


def sequence(*handlers: Handler) -> Handler:
    """A handler that uses each given handler in turn, repeating the last."""
    calls = count()

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(next(calls), len(handlers) - 1)
        return handlers[index](request)

    return handler


INSUFFICIENT_CREDITS = {
    "error": "Insufficient credits",
    "detail": (
        "This request needs 75 credits ($0.075). Your wallet balance is 12 credits "
        "($0.012). Top up your wallet to continue."
    ),
    "group": "JobSearch",
    "credits_required": 75,
    "credits_balance": 12,
    "quota_remaining": None,
    "cost_per_request_usd": 0.075,
    "upgrade_url": "https://enterprise.jobo.world/credits?plan=job_search&tier=2",
    "top_up_url": "https://enterprise.jobo.world/credits?topup=10",
    "recommended_plan": None,
}

RATE_LIMITED = {
    "code": "rate_limit_exceeded",
    "error": "Rate limit exceeded",
    "detail": "Per-minute JobSearch request limit of 60 exceeded. Try again shortly.",
    "group": "JobSearch",
    "retry_after_seconds": 7,
}

#: What a sandbox key gets from an endpoint the sandbox does not serve.
NOT_IN_SANDBOX = {
    "type": "https://tools.ietf.org/html/rfc9110#section-15.5.5",
    "title": "Not available in the sandbox",
    "status": 404,
    "code": "not_available_in_sandbox",
    "detail": (
        "Sandbox keys (jbe_test_…) serve job search, job and company lookups, "
        "Auto Apply applications, and reading mailboxes. Use a live key for "
        "every other endpoint."
    ),
    "instance": "/api/connectors/companies/suggest",
}

#: `GET /api/jobs/{id}` for an id no job has. The API sends no `code` here.
JOB_NOT_FOUND = {
    "type": "https://tools.ietf.org/html/rfc9110#section-15.5.5",
    "title": "Job not found",
    "status": 404,
    "detail": "No job exists with id '00000000-0000-4000-8000-000000000042'.",
}

#: An ASP.NET validation problem: the reasons are in `errors`, not `detail`.
VALIDATION_PROBLEM = {
    "type": "https://tools.ietf.org/html/rfc9110#section-15.5.1",
    "title": "One or more validation errors occurred.",
    "status": 400,
    "errors": {
        "page_size": ["page_size must be between 1 and 100."],
        "locations[0].country": [
            "country is required when city or region is set (a name or ISO "
            'code, e.g. "United States" or "US").'
        ],
    },
}

CURSOR_RESTART = {
    "title": "Feed cursor restart required",
    "status": 409,
    "code": "feed_cursor_restart_required",
    "detail": "This cursor reached the pagination limit.",
    "replacement_request": {"stable_scan": True, "batch_size": 1000},
}
