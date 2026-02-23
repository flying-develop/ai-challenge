"""Core domain data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """Represents a single message in a chat conversation."""

    role: str
    content: str


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""

    text: str
    model: str
    usage: dict = field(default_factory=dict)
