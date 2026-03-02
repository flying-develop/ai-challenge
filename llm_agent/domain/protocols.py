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


@runtime_checkable
class ChatHistoryRepositoryProtocol(Protocol):
    """Протокол хранилища истории чата.

    Любой объект с методами load/append/clear удовлетворяет этому протоколу.
    Позволяет заменять SQLite-реализацию любой другой (in-memory, файловой и т.д.).
    """

    def load(self) -> list[ChatMessage]:
        """Загрузить все сообщения текущей сессии."""
        ...

    def append(self, message: ChatMessage) -> None:
        """Добавить сообщение в хранилище."""
        ...

    def clear(self) -> None:
        """Удалить все сообщения текущей сессии."""
        ...


@runtime_checkable
class TokenCounterProtocol(Protocol):
    """Любой объект, умеющий считать токены, удовлетворяет этому протоколу."""

    def count_tokens(self, text: str) -> int:
        """Подсчитать количество токенов в произвольном тексте."""
        ...

    def count_messages_tokens(self, messages: list[ChatMessage]) -> int:
        """Подсчитать токены для списка сообщений в формате OpenAI chat."""
        ...
