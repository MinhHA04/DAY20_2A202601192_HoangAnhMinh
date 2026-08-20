"""Bounded LangGraph workflow for the specialized research team."""

from collections.abc import Callable
from time import perf_counter
from typing import Any

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        shared_llm = LLMClient(self.settings)
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent(
            SearchClient(self.settings),
            shared_llm,
        )
        self.analyst = analyst or AnalystAgent(shared_llm)
        self.writer = writer or WriterAgent(shared_llm)
        self.critic = critic or CriticAgent()
        self._started_at = 0.0

    @staticmethod
    def _coerce_state(value: ResearchState | dict[str, Any]) -> ResearchState:
        return value if isinstance(value, ResearchState) else ResearchState.model_validate(value)

    def _check_timeout(self) -> None:
        if self._started_at and perf_counter() - self._started_at > self.settings.timeout_seconds:
            raise AgentExecutionError(
                f"workflow exceeded timeout of {self.settings.timeout_seconds} seconds"
            )

    def _node(self, agent: BaseAgent) -> Callable[[ResearchState], dict[str, Any]]:
        def invoke(value: ResearchState) -> dict[str, Any]:
            state = self._coerce_state(value)
            self._check_timeout()
            last_error: Exception | None = None
            for attempt in range(1, self.settings.max_retries + 2):
                span: dict[str, Any]
                try:
                    with trace_span(
                        f"agent.{agent.name}",
                        {"attempt": attempt, "iteration": state.iteration},
                    ) as span:
                        state = agent.run(state)
                    state.trace.append(span)
                    return state.model_dump()
                except Exception as exc:
                    last_error = exc
                    state.trace.append(span)
                    if attempt <= self.settings.max_retries:
                        continue
            message = (
                f"{agent.name} failed after {self.settings.max_retries + 1} attempt(s): "
                f"{last_error}"
            )
            state.errors.append(message)
            return state.model_dump()

        return invoke

    @staticmethod
    def _next_route(value: ResearchState) -> str:
        state = MultiAgentWorkflow._coerce_state(value)
        return state.route_history[-1] if state.route_history else "done"

    def build(self) -> Any:
        """Compile nodes, conditional routing, and the terminal edge."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - dependency installation failure
            raise AgentExecutionError("Install the 'llm' extra to build the LangGraph") from exc

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._node(self.supervisor))  # type: ignore[call-overload]
        graph.add_node("researcher", self._node(self.researcher))  # type: ignore[call-overload]
        graph.add_node("analyst", self._node(self.analyst))  # type: ignore[call-overload]
        graph.add_node("writer", self._node(self.writer))  # type: ignore[call-overload]
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "done": END},
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Invoke the compiled graph, audit citations, and validate completion."""

        self._started_at = perf_counter()
        result = self.build().invoke(
            state.model_dump(),
            config={"recursion_limit": self.settings.max_iterations * 2 + 4},
        )
        final_state = ResearchState.model_validate(result)
        if not final_state.final_answer:
            details = "; ".join(final_state.errors) or "writer did not produce an answer"
            raise AgentExecutionError(f"Workflow ended without a final answer: {details}")

        with trace_span("agent.critic", {"iteration": final_state.iteration}) as span:
            final_state = self.critic.run(final_state)
        final_state.trace.append(span)
        final_state.add_trace_event(
            "workflow_complete",
            {
                "latency_seconds": perf_counter() - self._started_at,
                "iterations": final_state.iteration,
                "input_tokens": final_state.input_tokens,
                "output_tokens": final_state.output_tokens,
            },
        )
        return final_state
