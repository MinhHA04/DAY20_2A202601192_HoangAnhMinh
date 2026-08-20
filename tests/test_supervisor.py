"""Supervisor routing-policy tests."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_first_missing_artifact() -> None:
    settings = Settings(_env_file=None)
    supervisor = SupervisorAgent(settings)
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [
        SourceDocument(source_id="s1", title="Source", snippet="Evidence for the test")
    ]
    state.research_notes = "notes [s1]"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "analysis [s1]"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "answer [s1]"
    supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_enforces_iteration_limit() -> None:
    settings = Settings(_env_file=None).model_copy(update={"max_iterations": 1})
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=1,
    )
    SupervisorAgent(settings).run(state)
    assert state.route_history[-1] == "done"
    assert "maximum iterations" in state.errors[-1]
