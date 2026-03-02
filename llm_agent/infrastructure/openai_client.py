"""Адаптер для OpenAI API через официальный SDK (openai>=1.0).

Использует gpt-3.5-turbo — модель с контекстом 4096 токенов,
что удобно для демонстрации переполнения контекста.

Установка:
    pip install openai tiktoken

Конфигурация (в .env или переменных окружения):
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-3.5-turbo          # опционально
    OPENAI_BASE_URL=https://...         # опционально (для совместимых API)
"""

from __future__ import annotations

import os

from openai import OpenAI

from llm_agent.domain.models import ChatMessage, LLMResponse

# Лимиты контекстных окон (токены)
CONTEXT_LIMITS: dict[str, int] = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16_385,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}


class OpenAIClient:
    """Реализует LLMClientProtocol через официальный openai SDK.

    Автоматически читает OPENAI_API_KEY из окружения.
    Можно передать кастомный base_url для совместимых API (Azure, Together, etc.)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-3.5-turbo",
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )

    @property
    def context_limit(self) -> int:
        """Лимит контекста для выбранной модели (0 если неизвестен)."""
        return CONTEXT_LIMITS.get(self._model, 0)

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Отправить историю диалога в OpenAI API и вернуть ответ.

        Args:
            messages: Полная история чата. Системное сообщение (если есть)
                      должно быть первым элементом с role='system'.

        Returns:
            LLMResponse с usage.prompt_tokens и usage.completion_tokens.

        Raises:
            openai.AuthenticationError: При неверном API-ключе.
            openai.RateLimitError: При превышении лимита запросов.
            openai.BadRequestError: При context overflow (HTTP 400).
        """
        oai_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict = {"model": self._model, "messages": oai_messages}
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            usage=usage,
        )
