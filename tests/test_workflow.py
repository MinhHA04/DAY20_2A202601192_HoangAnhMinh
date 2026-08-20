from multi_agent_research_lab.agents.helpers import normalize_citations, validate_citations
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.baseline import SingleAgentRunner
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.search_client import SearchClient


def offline_settings() -> Settings:
    return Settings(_env_file=None).model_copy(update={"offline_mode": True, "max_retries": 0})


def test_search_preserves_source_provenance() -> None:
    sources = SearchClient(offline_settings()).search("single agent multi agent", 4)
    assert len(sources) == 4
    assert len({source.source_id for source in sources}) == 4
    assert all(source.metadata.get("corpus_file") for source in sources)


def test_citation_normalization_and_validation() -> None:
    source = SearchClient(offline_settings()).search("multi agent research", 1)[0]
    text = normalize_citations(f"Claim [source_id='{source.source_id}'] [not-provided]")
    valid, invalid = validate_citations(text, [source])
    assert valid == {source.source_id}
    assert invalid == {"not-provided"}


def test_offline_baseline_produces_cited_answer() -> None:
    state = SingleAgentRunner(offline_settings()).run("Compare single and multi agent systems")
    assert state.final_answer
    assert any(f"[{source.source_id}]" in state.final_answer for source in state.sources)
    assert state.input_tokens == 0


def test_offline_workflow_runs_expected_routes() -> None:
    settings = offline_settings()
    state = ResearchState(request=ResearchQuery(query="Explain production agent guardrails"))
    result = MultiAgentWorkflow(settings).run(state)
    assert result.final_answer
    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert [item.agent.value for item in result.agent_results] == [
        "researcher",
        "analyst",
        "writer",
        "critic",
    ]
    assert any(item.get("name") == "workflow_complete" for item in result.trace)
