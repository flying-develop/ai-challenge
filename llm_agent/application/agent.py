"""SimpleAgent: orchestrates LLM interactions."""

from __future__ import annotations

from llm_agent.domain.protocols import LLMClientProtocol


class SimpleAgent:
    """Application-layer agent that delegates to an LLM client.

    Accepts any object satisfying LLMClientProtocol (structural subtyping).
    Does not know about HTTP, endpoints, or authentication — those details
    belong to the infrastructure layer.
    """

    def __init__(self, llm_client: LLMClientProtocol) -> None:
        self._llm_client = llm_client

    def ask(self, prompt: str) -> str:
        """Send a prompt and return the response text.

        Args:
            prompt: The user's input. Must be non-empty after stripping whitespace.

        Returns:
            The model's response, stripped of leading/trailing whitespace.

        Raises:
            ValueError: If prompt is empty or whitespace-only.
        """
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        response = self._llm_client.generate(prompt)
        return response.text.strip()
