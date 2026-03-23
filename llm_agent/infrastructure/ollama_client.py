"""HTTP-адаптер для Ollama — локального LLM-сервера.

Использует /api/chat endpoint с поддержкой системных промптов и истории.
Работает без API-ключей: только локальный сервер на localhost:11434.

Образовательная концепция: локальная модель работает без API-ключей,
без интернета, без оплаты. Для RAG и несложных задач вполне применима.

Пример использования:
    from llm_agent.infrastructure.ollama_client import OllamaHttpClient

    client = OllamaHttpClient(model="qwen2.5:3b")
    if not client.is_available():
        print("Запустите Ollama: ollama serve")
    else:
        response = client.generate([ChatMessage(role="user", content="Привет!")])
        print(response.text)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from llm_agent.domain.models import ChatMessage, LLMResponse


class OllamaHttpClient:
    """Реализует LLMClientProtocol для локального Ollama-сервера.

    Взаимодействует через /api/chat — поддерживает системный промпт
    и многоходовую историю диалога. stream: false — синхронный ответ.

    Таймаут 120с: локальная модель может генерировать медленнее облачных.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def context_limit(self) -> int:
        """Контекст qwen2.5:3b — 32k, для агента ограничиваем консервативно."""
        return 4096

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Отправить историю диалога в Ollama /api/chat и вернуть ответ.

        Системное сообщение (role='system') передаётся внутри messages[] —
        Ollama принимает его наравне с OpenAI-форматом.

        Args:
            messages: История диалога (ChatMessage с role и content).

        Returns:
            LLMResponse с текстом и статистикой токенов.

        Raises:
            RuntimeError: Если Ollama недоступна или модель не загружена.
            urllib.error.URLError: При низкоуровневой сетевой ошибке.
        """
        payload_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        body = json.dumps({
            "model": self._model,
            "messages": payload_messages,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama недоступна по адресу {self._base_url}.\n"
                f"Запустите сервер: ollama serve\n"
                f"  Причина: {exc.reason}"
            ) from exc

        text = data["message"]["content"]

        # Ollama возвращает prompt_eval_count и eval_count (токены ответа)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return LLMResponse(
            text=text,
            model=self._model,
            usage=usage,
        )

    def is_available(self) -> bool:
        """Проверяет, что Ollama запущена и нужная модель загружена.

        Делает GET /api/tags (список загруженных моделей).
        Таймаут — 5с, чтобы не тормозить вывод /providers.

        Returns:
            True если сервер отвечает и модель присутствует в списке.
        """
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            # Ollama хранит имена как "qwen2.5:3b" или "qwen2.5:3b:latest"
            return self._model in models or f"{self._model}:latest" in models
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        """Идентификатор провайдера для отображения в интерфейсе."""
        return f"ollama/{self._model}"
