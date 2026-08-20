"""Command-line interface for running and benchmarking the completed lab."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.baseline import SingleAgentRunner
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing
from multi_agent_research_lab.services.storage import LocalArtifactStore

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(help="Run and benchmark the Multi-Agent Research Lab")
console = Console()


def _init(offline: bool = False) -> Settings:
    settings = get_settings()
    if offline and not settings.offline_mode:
        settings = settings.model_copy(update={"offline_mode": True})
    configure_logging(settings.log_level)
    configure_tracing(settings)
    return settings


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Disable provider calls and use the local corpus"),
    ] = False,
) -> None:
    """Run the one-call single-agent comparison baseline."""

    settings = _init(offline)
    request = _parse_query(query)
    result, metrics = run_benchmark(
        "baseline",
        request.query,
        SingleAgentRunner(settings).run,
    )
    if not result.final_answer:
        console.print(Panel.fit("; ".join(result.errors), title="Baseline failed", style="red"))
        raise typer.Exit(code=2)
    console.print(Panel(Text(result.final_answer), title="Single-Agent Baseline"))
    console.print(
        f"latency={metrics.latency_seconds:.2f}s  tokens="
        f"{metrics.input_tokens + metrics.output_tokens}  "
        f"quality={metrics.quality_score:.1f}/10  citations={metrics.citation_coverage:.0%}"
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Disable provider calls and use the local corpus"),
    ] = False,
) -> None:
    """Run Supervisor -> Researcher -> Analyst -> Writer (+ citation audit)."""

    settings = _init(offline)
    request = _parse_query(query)
    result, metrics = run_benchmark(
        "multi-agent",
        request.query,
        lambda value: _run_multi_query(value, settings),
    )
    if not result.final_answer:
        console.print(Panel.fit("; ".join(result.errors), title="Workflow failed", style="red"))
        raise typer.Exit(code=2)
    console.print(Panel(Text(result.final_answer), title="Multi-Agent Result"))
    console.print(
        f"routes={' → '.join(result.route_history)}  latency={metrics.latency_seconds:.2f}s  "
        f"tokens={metrics.input_tokens + metrics.output_tokens}  "
        f"quality={metrics.quality_score:.1f}/10  citations={metrics.citation_coverage:.0%}"
    )


def _run_multi_query(query: str, settings: Settings) -> ResearchState:
    return MultiAgentWorkflow(settings).run(ResearchState(request=ResearchQuery(query=query)))


@app.command()
def benchmark(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, help="YAML benchmark config"),
    ] = Path("configs/lab_default.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="Markdown report path"),
    ] = Path("reports/benchmark_report.md"),
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Run a deterministic zero-provider-cost benchmark"),
    ] = False,
) -> None:
    """Run all configured queries through both architectures and save traces."""

    settings = _init(offline)
    config_data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    queries = config_data.get("benchmark", {}).get("queries", [])
    if not queries or not all(isinstance(item, str) for item in queries):
        console.print(Panel.fit("No benchmark queries found in config", style="red"))
        raise typer.Exit(code=1)

    metrics = []
    artifact_store = LocalArtifactStore(output.parent)
    baseline_runner = SingleAgentRunner(settings)
    for index, query in enumerate(queries, start=1):
        console.print(f"[{index}/{len(queries)}] {query}")
        baseline_state, baseline_metrics = run_benchmark(
            f"baseline-q{index}", query, baseline_runner.run
        )
        multi_state, multi_metrics = run_benchmark(
            f"multi-agent-q{index}", query, lambda value: _run_multi_query(value, settings)
        )
        metrics.extend([baseline_metrics, multi_metrics])
        artifact_store.write_text(
            f"traces/query_{index}_baseline.json",
            baseline_state.model_dump_json(indent=2),
        )
        artifact_store.write_text(
            f"traces/query_{index}_multi_agent.json",
            multi_state.model_dump_json(indent=2),
        )

    report = render_markdown_report(metrics)
    report_path = artifact_store.write_text(output.name, report)
    machine_metrics = [item.model_dump(mode="json") for item in metrics]
    artifact_store.write_text("benchmark_metrics.json", json.dumps(machine_metrics, indent=2))
    console.print(Panel.fit(str(report_path), title="Benchmark complete", style="green"))


if __name__ == "__main__":
    app()
