"""Structural protocols for dependency inversion."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_agent.domain.models import LLMResponse


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Any object with a generate() method satisfies this protocol."""

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for the given prompt."""
        ...
