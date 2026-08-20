"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render metrics, method limits, comparison, and known failure modes."""

    lines = [
        "# Báo cáo benchmark",
        "",
        "> Được tạo bởi `malab benchmark`. Quality score là một structural heuristic minh bạch, "
        "không phải human/LLM judge score.",
        "",
        "## Kết quả",
        "",
        "| Run | Latency (s) | Input tokens | Output tokens | Cost (USD) | Quality /10 "
        "| Citation coverage | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        notes = item.notes.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {item.input_tokens} "
            f"| {item.output_tokens} | {cost} | {quality} | {citation} | {failure} | {notes} |"
        )
    baseline_items = [item for item in metrics if item.run_name.startswith("baseline")]
    multi_items = [item for item in metrics if item.run_name.startswith("multi-agent")]
    observed_lines: list[str] = []
    if baseline_items and multi_items:
        lines.extend(
            [
                "",
                "## Trung bình theo kiến trúc",
                "",
                "| Architecture | Mean latency (s) | Mean tokens | Mean quality /10 | "
                "Mean citation coverage |",
                "|---|---:|---:|---:|---:|",
                _average_row("Baseline", baseline_items),
                _average_row("Multi-agent", multi_items),
            ]
        )
        baseline_latency, baseline_tokens, baseline_quality, baseline_citations = _averages(
            baseline_items
        )
        multi_latency, multi_tokens, multi_quality, multi_citations = _averages(multi_items)
        fallback_runs = sum("offline-fallback" in item.notes for item in multi_items)
        observed_lines = [
            "",
            f"Số liệu quan sát: baseline có mean latency {baseline_latency:.2f}s và "
            f"{baseline_tokens:.0f} tokens; multi-agent có mean latency {multi_latency:.2f}s và "
            f"{multi_tokens:.0f} tokens. Quality heuristic thay đổi từ {baseline_quality:.1f} "
            f"lên {multi_quality:.1f}; citation coverage thay đổi từ "
            f"{baseline_citations:.0%} lên {multi_citations:.0%}.",
        ]
        if fallback_runs:
            observed_lines.append(
                f"Có {fallback_runs} multi-agent run dùng deterministic fallback sau lỗi online "
                "validation. Structural score cao của fallback text có thể chỉ phản ánh template "
                "completeness, vì vậy vẫn cần peer review."
            )
    lines.extend(
        [
            "",
            "## Phương pháp đo",
            "",
            "- Latency là end-to-end wall-clock time. Token counts lấy từ provider khi dùng "
            "online model; offline fallback báo 0 tokens.",
            "- Citation coverage là tỷ lệ retrieved source IDs được cite ít nhất một lần. Chỉ số "
            "này không chứng minh mỗi citation thực sự entail claim tương ứng.",
            "- Quality heuristic (0–10) xét độ dài, query-term coverage, valid source-ID coverage, "
            "counterargument và explicit limitations. Khi chấm cuối, peer review nên thay thế "
            "heuristic này.",
            "- Failure rate bằng 1 khi run không có final answer hoặc gặp uncaught exception; "
            "các trường hợp còn lại bằng 0.",
            "",
            "## Diễn giải",
            "",
            "Multi-agent workflow được dự kiến sẽ dùng nhiều calls/tokens hơn. Kiến trúc này chỉ "
            "tạo thêm giá trị nếu các research/analysis handoffs cải thiện evidence coverage hoặc "
            "review quality đủ để bù coordination overhead. Cần so sánh paired runs của cùng query "
            "và không suy ra multi-agent luôn vượt trội từ mẫu lab nhỏ này.",
            *observed_lines,
            "",
            "## Failure modes và cách giảm thiểu",
            "",
            "- **Cascading hallucination:** research note yếu có thể làm nhiễm các downstream "
            "stages. Hệ thống giữ source IDs và audit citations với source ledger gốc.",
            "- **Coordination overhead:** extra calls có thể tăng latency/cost mà không tạo thông "
            "tin mới. Cần giữ single-agent baseline và ablate agent không cải thiện metrics.",
            "- **Provider/search outage:** requests có bounded retries và timeout; workflow chuyển "
            "sang versioned local corpus và ghi lỗi trong state.",
            "- **Infinite routing:** Supervisor có hard iteration cap và graph có recursion limit.",
            "",
            "## Trace artifacts",
            "",
            "Per-run JSON state, gồm route events và timed spans, được lưu trong "
            "`reports/traces/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _average_row(label: str, items: list[BenchmarkMetrics]) -> str:
    latency, tokens, quality, citation = _averages(items)
    return f"| {label} | {latency:.2f} | {tokens:.0f} | {quality:.1f} | {citation:.0%} |"


def _averages(items: list[BenchmarkMetrics]) -> tuple[float, float, float, float]:
    count = len(items)
    latency = sum(item.latency_seconds for item in items) / count
    tokens = sum(item.input_tokens + item.output_tokens for item in items) / count
    qualities = [item.quality_score for item in items if item.quality_score is not None]
    citations = [item.citation_coverage for item in items if item.citation_coverage is not None]
    quality = sum(qualities) / len(qualities) if qualities else 0.0
    citation = sum(citations) / len(citations) if citations else 0.0
    return latency, tokens, quality, citation
