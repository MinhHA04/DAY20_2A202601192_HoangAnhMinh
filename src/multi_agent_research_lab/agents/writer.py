"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.helpers import (
    normalize_citations,
    offline_final_answer,
    validate_citations,
)
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize the final cited response from the two handoff artifacts."""

        if not state.research_notes or not state.analysis_notes:
            raise AgentExecutionError("Writer requires both research and analysis notes")

        mode = "offline-fallback"
        answer = offline_final_answer(
            state.request.query,
            state.sources,
            state.analysis_notes,
        )
        if self.llm_client.available:
            try:
                valid_ids = ", ".join(source.source_id for source in state.sources)
                response = self.llm_client.complete(
                    system_prompt=(
                        "You are the Writer in a multi-agent research workflow. Answer clearly for "
                        f"{state.request.audience}. Synthesize rather than copy the handoffs. Cite "
                        "major factual claims using exact compact [ID] syntax, for example "
                        "[autogen]. Never use [source_id='ID'] or an ID outside the valid "
                        "list. Distinguish synthetic evidence, include counterarguments and "
                        "limitations, and make no claim beyond the supplied "
                        "evidence. Match the language of the user's question."
                    ),
                    user_prompt=(
                        f"Question: {state.request.query}\n"
                        f"Valid citation IDs only: {valid_ids}\n\n"
                        f"RESEARCH NOTES\n{state.research_notes}\n\n"
                        f"ANALYSIS NOTES\n{state.analysis_notes}"
                    ),
                )
                state.add_usage(
                    response.input_tokens,
                    response.output_tokens,
                    response.cost_usd,
                )
                candidate = normalize_citations(response.content)
                valid_citations, invalid_citations = validate_citations(
                    candidate,
                    state.sources,
                )
                if not valid_citations or invalid_citations:
                    invalid_text = ", ".join(sorted(invalid_citations)) or "none valid"
                    raise AgentExecutionError(f"writer citation validation failed: {invalid_text}")
                answer = candidate
                mode = "openai"
            except AgentExecutionError as exc:
                state.errors.append(f"writer online fallback: {exc}")

        state.final_answer = answer
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=answer,
                metadata={"mode": mode},
            )
        )
        return state
