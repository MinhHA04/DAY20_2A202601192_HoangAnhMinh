"""Prompt, citation, and deterministic fallback helpers shared by agents."""

import re

from multi_agent_research_lab.core.schemas import SourceDocument

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\](?!\()")
SOURCE_ATTRIBUTE_RE = re.compile(
    r"\[source_id\s*=\s*['\"]([A-Za-z0-9][A-Za-z0-9_.:-]*)['\"]\]",
    flags=re.IGNORECASE,
)


def format_sources(sources: list[SourceDocument], max_chars_each: int = 1800) -> str:
    """Render source cards as clearly delimited, untrusted evidence."""

    blocks: list[str] = []
    for source in sources:
        source_kind = "synthetic" if source.metadata.get("is_synthetic") else "public summary"
        blocks.append(
            "--- BEGIN UNTRUSTED SOURCE CARD ---\n"
            f"SOURCE_ID: {source.source_id}\nKIND: {source_kind}\n"
            f"TITLE: {source.title}\nCONTENT: {source.snippet[:max_chars_each]}\n"
            "--- END UNTRUSTED SOURCE CARD ---"
        )
    return "\n\n".join(blocks)


def normalize_citations(text: str) -> str:
    """Normalize a common model variation to the required compact syntax."""

    return SOURCE_ATTRIBUTE_RE.sub(r"[\1]", text)


def citation_ids(text: str) -> set[str]:
    """Extract compact source citations while ignoring Markdown links."""

    return set(CITATION_RE.findall(text))


def validate_citations(text: str, sources: list[SourceDocument]) -> tuple[set[str], set[str]]:
    """Return valid and unknown citation IDs present in generated text."""

    available = {source.source_id for source in sources}
    cited = citation_ids(text)
    return cited & available, cited - available


def first_sentence(text: str, max_chars: int = 320) -> str:
    """Extract a compact readable sentence for offline synthesis."""

    cleaned = " ".join(text.split())
    sentence = SENTENCE_RE.split(cleaned, maxsplit=1)[0]
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max_chars - 1].rstrip() + "…"


def offline_research_notes(sources: list[SourceDocument]) -> str:
    """Produce evidence-preserving notes when an online model is unavailable."""

    lines = ["## Evidence ledger"]
    for source in sources:
        label = (
            "synthetic evidence" if source.metadata.get("is_synthetic") else "public source summary"
        )
        lines.append(
            f"- [{source.source_id}] **{source.title}** ({label}): {first_sentence(source.snippet)}"
        )
    lines.extend(
        [
            "",
            "## Research caveat",
            "The corpus contains embedded summaries. Provenance URLs identify the original public "
            "sources, but this offline run did not open those URLs.",
        ]
    )
    return "\n".join(lines)


def offline_analysis(sources: list[SourceDocument]) -> str:
    """Create a bounded evidence analysis without claiming unsupported conclusions."""

    citations = " ".join(f"[{source.source_id}]" for source in sources[:4])
    synthetic = [source.source_id for source in sources if source.metadata.get("is_synthetic")]
    synthetic_note = (
        "Synthetic items must be treated as benchmark evidence, not real publications: "
        + ", ".join(f"[{item}]" for item in synthetic)
        if synthetic
        else "No selected source card is labeled synthetic."
    )
    return "\n".join(
        [
            "## Key findings",
            "- The selected evidence supports a conditional, task-specific conclusion "
            f"{citations}.",
            "- Architecture quality depends on evidence provenance, explicit handoffs, validation, "
            "and measured cost/latency—not the number of agents alone.",
            "- A single-agent baseline remains necessary to show that added coordination creates "
            "value rather than duplicated compute.",
            "",
            "## Evidence quality and conflicts",
            f"- {synthetic_note}",
            "- Embedded source summaries establish bounded claims; cross-domain generalization "
            "remains uncertain.",
        ]
    )


def offline_final_answer(query: str, sources: list[SourceDocument], analysis: str) -> str:
    """Write a useful cited report directly from the source ledger."""

    findings = ["## Evidence-based answer"]
    for source in sources[:5]:
        qualifier = " (synthetic)" if source.metadata.get("is_synthetic") else ""
        findings.append(f"- {first_sentence(source.snippet)} [{source.source_id}]{qualifier}")
    findings.extend(
        [
            "",
            "## Synthesis",
            "The evidence favors a conditional design decision: use specialization when the task "
            "contains genuinely separable research, analysis, and verification work. For narrow "
            "tasks, a single agent is usually the stronger baseline because it avoids handoff and "
            "coordination overhead.",
            "",
            "## Recommended evaluation",
            "Compare the same queries and model across architectures. Measure factual quality, "
            "citation coverage, wall-clock latency, token usage, failures, and handoff defects. "
            "Keep provenance in shared state and use bounded retries, timeouts, output validation, "
            "and a deterministic fallback.",
            "",
            "## Limitations",
            "This answer is based on the supplied offline corpus and its embedded source "
            "summaries; "
            "the linked public sources were not independently reopened during offline retrieval.",
            "",
            f"_Research question: {query}_",
            "",
            "<details><summary>Analysis handoff</summary>",
            "",
            analysis,
            "",
            "</details>",
        ]
    )
    return "\n".join(findings)
