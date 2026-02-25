"""HTTP-адаптер для Anthropic Messages API.

Использует claude-haiku-4-5 как модель с маленьким контекстом (4096 токенов,
аналогично GPT-3.5-turbo). Реальный контекст Haiku — 200k, но мы ограничиваем
искусственно через параметр context_limit агента.

Аутентификация — Bearer-токен (совместим с Claude Code session ingress token).
Читает токен из переменной ANTHROPIC_API_KEY или файла CLAUDE_SESSION_INGRESS_TOKEN_FILE.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from llm_agent.domain.models import ChatMessage, LLMResponse

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

# Лимиты контекстных окон Anthropic-моделей (реальные, в токенах)
CONTEXT_LIMITS: dict[str, int] = {
    "claude-haiku-4-5-20251001": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
}

# Для демонстрации «маленького» контекста (эмуляция GPT-3.5-turbo)
DEMO_CONTEXT_LIMIT = 4096


def _load_token() -> str:
    """Загрузить Bearer-токен из окружения или файла.

    Приоритет:
    1. ANTHROPIC_API_KEY
    2. CLAUDE_SESSION_INGRESS_TOKEN_FILE (Claude Code remote)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key

    token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE", "").strip()
    if token_file and os.path.exists(token_file):
        return open(token_file).read().strip()

    raise ValueError(
        "Не найден Anthropic API-токен. "
        "Установите ANTHROPIC_API_KEY или убедитесь, что "
        "CLAUDE_SESSION_INGRESS_TOKEN_FILE указывает на существующий файл."
    )


class AnthropicHttpClient:
    """Реализует LLMClientProtocol для Anthropic Messages API.

    Особенности формата Anthropic (отличия от OpenAI):
    - system-сообщение передаётся отдельным полем, не в messages[]
    - ответ: content[0].text, usage.input_tokens / output_tokens
    - required параметр max_tokens в каждом запросе
    """

    def __init__(
        self,
        token: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        self._token = token or _load_token()
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def context_limit(self) -> int:
        """Реальный лимит контекста для модели (не демонстрационный)."""
        return CONTEXT_LIMITS.get(self._model, 200_000)

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Отправить историю диалога в Anthropic API и вернуть ответ.

        Системное сообщение (role='system') автоматически выделяется
        в отдельное поле system, так как Anthropic не принимает его в messages[].

        Returns:
            LLMResponse с полем usage, содержащим prompt_tokens и completion_tokens
            (переименованы из input_tokens / output_tokens для единообразия с OpenAI).

        Raises:
            urllib.error.HTTPError: При HTTP-ошибке API.
            ValueError: При некорректном формате ответа.
        """
        system_content = ""
        chat_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": chat_messages,
        }
        if system_content:
            payload["system"] = system_content

        req = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._token}",
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise urllib.error.HTTPError(
                exc.url, exc.code,
                f"Ошибка Anthropic API {exc.code}: {body}",
                exc.headers, None,
            ) from exc

        text = data["content"][0]["text"]
        raw_usage = data.get("usage", {})

        # Нормализуем названия полей — приводим к OpenAI-совместимому виду
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": (
                raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0)
            ),
        }

        return LLMResponse(
            text=text,
            model=data.get("model", self._model),
            usage=usage,
        )
