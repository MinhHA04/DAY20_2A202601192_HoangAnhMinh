"""Deterministic search over the supplied offline research corpus."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")


class SearchClient:
    """Ranks embedded source cards, preserving provenance and synthetic labels."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return the most relevant cards from all corpus topic files."""

        if max_results < 1:
            return []
        documents = self._load_documents(str(Path(self.settings.corpus_dir).resolve()))
        query_tokens = self._expand_query(self._tokens(query))
        ranked = sorted(
            documents,
            key=lambda item: self._score(item, query_tokens),
            reverse=True,
        )
        positive = [item for item in ranked if self._score(item, query_tokens) > 0]
        candidates = positive or ranked

        def distinct(items: list[SourceDocument]) -> list[SourceDocument]:
            result: list[SourceDocument] = []
            seen: set[str] = set()
            for item in items:
                if item.source_id not in seen:
                    result.append(item)
                    seen.add(item.source_id)
            return result

        public = distinct([item for item in candidates if not item.metadata.get("is_synthetic")])
        synthetic = distinct([item for item in candidates if item.metadata.get("is_synthetic")])
        # Favor independently traceable public summaries. Keep at most one labeled
        # synthetic item so the workflow can demonstrate source-quality reasoning.
        ordered = public[: max(1, max_results - 1)] + synthetic[:1]
        if len(ordered) < max_results:
            chosen_ids = {item.source_id for item in ordered}
            ordered.extend(item for item in candidates if item.source_id not in chosen_ids)
        unique: list[SourceDocument] = []
        seen_ids: set[str] = set()
        for item in ordered:
            if item.source_id in seen_ids:
                continue
            seen_ids.add(item.source_id)
            unique.append(item)
            if len(unique) == max_results:
                break
        return unique

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(TOKEN_RE.findall(value.lower()))

    @staticmethod
    def _expand_query(tokens: set[str]) -> set[str]:
        expanded = set(tokens)
        synonyms = {
            "graphrag": {"retrieval", "augmented", "grounding", "evidence", "graph"},
            "guardrail": {"reliability", "recovery", "security", "verification"},
            "guardrails": {"reliability", "recovery", "security", "verification"},
            "customer": {"enterprise", "workflow", "automation"},
            "support": {"enterprise", "workflow", "automation"},
        }
        for token in tokens:
            expanded.update(synonyms.get(token, set()))
        return expanded

    @classmethod
    def _score(cls, document: SourceDocument, query_tokens: set[str]) -> int:
        title_tokens = cls._tokens(document.title)
        topic_tokens = cls._tokens(str(document.metadata.get("topic", "")))
        body_tokens = cls._tokens(document.snippet)
        return (
            8 * len(query_tokens & title_tokens)
            + 5 * len(query_tokens & topic_tokens)
            + len(query_tokens & body_tokens)
        )

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_documents(corpus_dir: str) -> tuple[SourceDocument, ...]:
        root = Path(corpus_dir)
        if not root.is_dir():
            return ()

        documents: list[SourceDocument] = []
        for path in sorted(root.glob("*.json")):
            try:
                payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            topic = payload.get("topic", {})
            topic_name = str(topic.get("name", path.stem.replace("_", " ")))
            knowledge_base = payload.get("knowledge_base", {})
            for source in knowledge_base.get("source_documents", []):
                source_id = str(source.get("document_id", "unknown"))
                full_text = str(source.get("full_text", ""))
                takeaways = " ".join(str(x) for x in source.get("key_takeaways", []))
                snippet = (takeaways or full_text).strip()[:4000]
                documents.append(
                    SourceDocument(
                        source_id=source_id,
                        title=str(source.get("title", source_id)),
                        url=source.get("provenance_url"),
                        snippet=snippet,
                        metadata={
                            "topic": topic_name,
                            "document_class": source.get("document_class"),
                            "is_synthetic": bool(source.get("is_synthetic", False)),
                            "citation_label": source.get("citation_label", source_id),
                            "year": source.get("year"),
                            "corpus_file": path.name,
                        },
                    )
                )
        return tuple(documents)
