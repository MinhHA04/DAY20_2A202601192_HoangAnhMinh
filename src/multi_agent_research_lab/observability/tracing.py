"""Local spans plus optional LangSmith configuration."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith when configured; local state tracing is always active."""

    if not settings.langsmith_api_key or settings.offline_mode:
        return False
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    return True


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Capture a portable span that can be exported with the final state."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "started_at": datetime.now(UTC).isoformat(),
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
