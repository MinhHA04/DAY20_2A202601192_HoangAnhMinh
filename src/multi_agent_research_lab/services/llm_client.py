"""Provider adapter for OpenAI's Responses API."""

from dataclasses import dataclass
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None


class LLMClient:
    """Small Responses API client with SDK retries and request timeout."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        """Whether online generation is enabled and credentials are present."""

        return bool(self.settings.openai_api_key) and not self.settings.offline_mode

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.available:
            raise AgentExecutionError(
                "OpenAI is unavailable; use the deterministic offline fallback"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - install error is environment-specific
            raise AgentExecutionError("Install the 'llm' extra to use OpenAI") from exc

        client: Any = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=float(self.settings.timeout_seconds),
            max_retries=self.settings.max_retries,
        )
        if self.settings.langsmith_api_key:
            try:
                from langsmith.wrappers import wrap_openai

                client = wrap_openai(client)
            except (ImportError, RuntimeError):
                # Local JSON tracing remains available even without the optional wrapper.
                pass
        self._client = client
        return client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return one text response and provider-reported token usage."""

        try:
            response = self._get_client().responses.create(
                model=self.settings.openai_model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=self.settings.max_output_tokens,
                store=False,
            )
        except Exception as exc:
            raise AgentExecutionError(
                f"OpenAI request failed: {type(exc).__name__}: {exc}"
            ) from exc

        content = response.output_text.strip()
        if not content:
            raise AgentExecutionError("OpenAI returned an empty text response")
        usage = response.usage
        return LLMResponse(
            content=content,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            model=getattr(response, "model", self.settings.openai_model),
        )
