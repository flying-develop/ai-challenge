"""HTTP-адаптер для OpenAI API (gpt-3.5-turbo и другие модели)."""

from __future__ import annotations

import httpx

from llm_agent.domain.models import ChatMessage, LLMResponse

OPENAI_BASE_URL = "https://api.openai.com/v1/"

# Лимиты контекстных окон для справки (в токенах)
CONTEXT_LIMITS: dict[str, int] = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
}


class OpenAIHttpClient:
    """Реализует LLMClientProtocol для OpenAI API.

    Совместим с любым OpenAI-совместимым провайдером (Azure, Groq и т.д.)
    — достаточно передать нужный base_url.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: str = OPENAI_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        normalized_base = base_url.rstrip("/") + "/"
        self._client = httpx.Client(
            base_url=normalized_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    @property
    def context_limit(self) -> int:
        """Лимит контекста для текущей модели (0 если неизвестен)."""
        return CONTEXT_LIMITS.get(self._model, 0)

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Отправить историю диалога в OpenAI API и вернуть структурированный ответ.

        Raises:
            httpx.HTTPStatusError: При HTTP-ответе с кодом не 2xx.
            httpx.TimeoutException: Если запрос превысил таймаут.
        """
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

        response = self._client.post("chat/completions", json=payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"Ошибка OpenAI API {exc.response.status_code}: {exc.response.text}",
                request=exc.request,
                response=exc.response,
            ) from exc

        data = response.json()
        choice = data["choices"][0]
        return LLMResponse(
            text=choice["message"]["content"],
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
        )

    def close(self) -> None:
        """Закрыть пул HTTP-соединений."""
        self._client.close()

    def __enter__(self) -> "OpenAIHttpClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
