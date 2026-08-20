"""Optional citation-audit agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.helpers import citation_ids
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Audit citation IDs and append a machine-readable finding."""

        if not state.final_answer:
            raise AgentExecutionError("Critic requires a final answer")
        valid_ids = {source.source_id for source in state.sources}
        cited_ids = citation_ids(state.final_answer)
        invalid_ids = sorted(cited_ids - valid_ids)
        coverage = len(cited_ids & valid_ids) / len(valid_ids) if valid_ids else 0.0
        finding = (
            f"Citation coverage={coverage:.0%}; invalid IDs="
            f"{', '.join(invalid_ids) if invalid_ids else 'none'}"
        )
        if invalid_ids:
            state.errors.append(f"critic found invalid citation IDs: {', '.join(invalid_ids)}")
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=finding,
                metadata={"citation_coverage": coverage, "invalid_ids": invalid_ids},
            )
        )
        state.add_trace_event("citation_audit", {"coverage": coverage, "invalid_ids": invalid_ids})
        return state
