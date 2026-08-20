"""Reproducible heuristic benchmark for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.agents.helpers import citation_ids
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, tokens, completion, citations, and transparent heuristic quality."""

    started = perf_counter()
    try:
        state = runner(query)
        latency = perf_counter() - started
    except Exception as exc:
        state = ResearchState.model_validate(
            {"request": {"query": query}, "errors": [f"{type(exc).__name__}: {exc}"]}
        )
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=perf_counter() - started,
            quality_score=0.0,
            citation_coverage=0.0,
            failure_rate=1.0,
            notes=f"failed: {type(exc).__name__}",
        )
        return state, metrics

    answer = state.final_answer or ""
    valid_ids = {source.source_id for source in state.sources}
    cited_ids = citation_ids(answer) & valid_ids
    citation_coverage = len(cited_ids) / len(valid_ids) if valid_ids else 0.0
    quality = _heuristic_quality(query, answer, citation_coverage)
    modes = sorted(
        {
            str(result.metadata.get("mode"))
            for result in state.agent_results
            if result.metadata.get("mode")
        }
    )
    notes = f"heuristic; modes={','.join(modes) or 'n/a'}; errors={len(state.errors)}"
    if state.errors:
        notes += f"; first_error={state.errors[0][:160]}"
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.estimated_cost_usd,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        quality_score=quality,
        citation_coverage=citation_coverage,
        failure_rate=0.0 if answer.strip() else 1.0,
        notes=notes,
    )
    return state, metrics


def _heuristic_quality(query: str, answer: str, citation_coverage: float) -> float:
    """Score observable properties only; this is not an LLM or human judge."""

    if not answer.strip():
        return 0.0
    length_score = 2.0 if len(answer) >= 500 else 1.0 if len(answer) >= 150 else 0.5
    query_terms = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 4}
    answer_terms = set(re.findall(r"[a-z0-9]+", answer.lower()))
    term_score = 2.0 * len(query_terms & answer_terms) / len(query_terms) if query_terms else 1.0
    lower_answer = answer.lower()
    limitation_score = 1.0 if any(x in lower_answer for x in ("limitation", "giới hạn")) else 0.0
    counter_score = (
        1.0
        if any(x in lower_answer for x in ("counter", "single-agent", "single agent", "mặt khác"))
        else 0.0
    )
    return round(
        min(
            10.0,
            length_score + term_score + 4.0 * citation_coverage + limitation_score + counter_score,
        ),
        2,
    )
