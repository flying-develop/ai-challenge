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


@dataclass
class TokenUsage:
    """Статистика токенов для одного хода агента."""

    request_tokens: int = 0      # токены только текущего сообщения пользователя
    history_tokens: int = 0      # токены всего контекста, отправленного в модель
    response_tokens: int = 0     # токены ответа модели
    total_tokens: int = 0        # history_tokens + response_tokens
    context_limit: int = 0       # максимальный контекст модели (0 = неизвестен)

    @property
    def context_usage_percent(self) -> float:
        """Процент использования контекстного окна (по отправленным токенам)."""
        if self.context_limit <= 0:
            return 0.0
        return (self.history_tokens / self.context_limit) * 100

    @property
    def is_near_limit(self) -> bool:
        """True, если использовано более 80% контекстного окна."""
        return self.context_limit > 0 and self.context_usage_percent >= 80.0

    @property
    def would_exceed_limit(self) -> bool:
        """True, если history_tokens превышает context_limit."""
        return self.context_limit > 0 and self.history_tokens > self.context_limit


class ContextLimitError(Exception):
    """Вызывается при превышении лимита контекста модели."""

    def __init__(self, tokens: int, limit: int) -> None:
        self.tokens = tokens
        self.limit = limit
        super().__init__(
            f"Превышен лимит контекста: {tokens} токенов при лимите {limit} "
            f"(+{tokens - limit} лишних токенов)"
        )
