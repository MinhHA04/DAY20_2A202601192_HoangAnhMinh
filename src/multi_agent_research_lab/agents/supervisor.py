"""Supervisor / deterministic router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Choose the first missing artifact, or stop at the iteration guard."""

        if state.iteration >= self.settings.max_iterations:
            route = "done"
            if not state.final_answer:
                state.errors.append(
                    f"maximum iterations reached ({self.settings.max_iterations}) before completion"
                )
        elif not state.research_notes or not state.sources:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        elif not state.final_answer:
            route = "writer"
        else:
            route = "done"

        state.record_route(route)
        state.add_trace_event(
            "route",
            {"next": route, "iteration": state.iteration, "errors": len(state.errors)},
        )
        return state
