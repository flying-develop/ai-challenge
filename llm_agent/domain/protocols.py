"""Структурные протоколы для инверсии зависимостей."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_agent.domain.models import ChatMessage, LLMResponse


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Любой объект с методом generate() удовлетворяет этому протоколу."""

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Сгенерировать ответ для списка сообщений чата."""
        ...
