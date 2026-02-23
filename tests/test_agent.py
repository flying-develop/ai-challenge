"""Tests for the application layer (SimpleAgent)."""

from __future__ import annotations

import pytest

from llm_agent.application.agent import SimpleAgent
from llm_agent.domain.models import LLMResponse


class MockLLMClient:
    """Minimal mock satisfying LLMClientProtocol."""

    def __init__(self, text: str = "mock response") -> None:
        self._text = text
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(text=self._text, model="mock-model", usage={})


def test_agent_rejects_empty_prompt() -> None:
    agent = SimpleAgent(llm_client=MockLLMClient())
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        agent.ask("")


def test_agent_rejects_whitespace_prompt() -> None:
    agent = SimpleAgent(llm_client=MockLLMClient())
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        agent.ask("   ")


def test_agent_returns_text_from_client() -> None:
    mock_client = MockLLMClient(text="  Hello from mock!  ")
    agent = SimpleAgent(llm_client=mock_client)
    result = agent.ask("Say hello")
    assert result == "Hello from mock!"  # stripped


def test_agent_passes_prompt_to_client() -> None:
    mock_client = MockLLMClient()
    agent = SimpleAgent(llm_client=mock_client)
    agent.ask("What is 2 + 2?")
    assert mock_client.last_prompt == "What is 2 + 2?"
