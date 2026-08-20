"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.helpers import format_sources, offline_analysis
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Extract claims, trade-offs, conflicts, and weak evidence."""

        if not state.research_notes or not state.sources:
            raise AgentExecutionError("Analyst requires research notes and source documents")

        mode = "offline-fallback"
        analysis = offline_analysis(state.sources)
        if self.llm_client.available:
            try:
                response = self.llm_client.complete(
                    system_prompt=(
                        "You are the Analyst. Turn research notes into a structured "
                        "evidence analysis. Identify key claims, agreements, conflicts, "
                        "weak evidence, counterarguments, and deployment trade-offs. "
                        "Preserve exact [ID] citations, such as [autogen]; never invent one "
                        "and never use [source_id='ID']. "
                        "Do not write the final answer."
                    ),
                    user_prompt=(
                        f"Question: {state.request.query}\n\n"
                        f"RESEARCH NOTES\n{state.research_notes}\n\n"
                        f"SOURCE LEDGER\n{format_sources(state.sources, max_chars_each=700)}"
                    ),
                )
                analysis = response.content
                mode = "openai"
                state.add_usage(
                    response.input_tokens,
                    response.output_tokens,
                    response.cost_usd,
                )
            except AgentExecutionError as exc:
                state.errors.append(f"analyst online fallback: {exc}")

        state.analysis_notes = analysis
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=analysis,
                metadata={"mode": mode},
            )
        )
        return state
