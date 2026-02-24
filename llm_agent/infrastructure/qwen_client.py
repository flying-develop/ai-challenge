"""HTTP-адаптер для API Qwen (Alibaba Cloud).

URL эндпоинта можно изменить через QWEN_BASE_URL без затрагивания
бизнес-логики — этот файл единственный, кто знает о HTTP-формате запросов.
"""

from __future__ import annotations

import httpx

from llm_agent.domain.models import ChatMessage, LLMResponse


class QwenHttpClient:
    """Реализует LLMClientProtocol для HTTP API Qwen.

    Использует httpx.Client (синхронный) с единым пулом соединений.
    Поддерживает использование как контекстный менеджер для детерминированного
    освобождения ресурсов.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        # Добавляем trailing slash, чтобы httpx корректно объединял пути.
        # По RFC 3986: путь с ведущим слешем означает «абсолютный путь от origin»,
        # что сбрасывает любой префикс base URL. Используем относительный путь.
        normalized_base = base_url.rstrip("/") + "/"
        self._client = httpx.Client(
            base_url=normalized_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Отправить историю диалога в Qwen API и вернуть структурированный ответ.

        Args:
            messages: Полная история чата в виде списка ChatMessage.
                      Ответственность за порядок лежит на вызывающем коде:
                      системное сообщение первым (если есть), затем чередование
                      user/assistant.

        Returns:
            LLMResponse с текстом ответа модели, именем модели и статистикой токенов.

        Raises:
            httpx.HTTPStatusError: При HTTP-ответе с кодом не 2xx (включает статус и тело).
            httpx.TimeoutException: Если запрос превысил настроенный таймаут.
            httpx.RequestError: При сбоях сетевого соединения.
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
                f"Ошибка Qwen API {exc.response.status_code}: {exc.response.text}",
                request=exc.request,
                response=exc.response,
            ) from exc

        data = response.json()
        # Примечание: choices может быть пустым при блокировке контент-модерацией (200, но нет choices).
        choice = data["choices"][0]
        return LLMResponse(
            text=choice["message"]["content"],
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
        )

    def close(self) -> None:
        """Закрыть пул HTTP-соединений."""
        self._client.close()

    def __enter__(self) -> "QwenHttpClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
