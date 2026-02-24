"""SimpleAgent: оркестратор взаимодействий с LLM."""

from __future__ import annotations

from llm_agent.domain.models import ChatMessage
from llm_agent.domain.protocols import LLMClientProtocol


class SimpleAgent:
    """Агент приложения, делегирующий запросы LLM-клиенту.

    Хранит историю диалога между вызовами ask().
    Принимает любой объект, удовлетворяющий LLMClientProtocol (структурная типизация).
    Не знает об HTTP, эндпоинтах или аутентификации — эти детали
    относятся к инфраструктурному слою.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        system_prompt: str | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._system_prompt = system_prompt.strip() if system_prompt else None
        self._history: list[ChatMessage] = []

    def ask(self, prompt: str) -> str:
        """Отправить сообщение, накопить историю и вернуть ответ модели.

        Полный контекст разговора (системный промпт + история + новое сообщение)
        передаётся LLM-клиенту при каждом вызове. Ответ ассистента добавляется
        в историю, чтобы следующие вызовы имели полный контекст.

        Args:
            prompt: Ввод пользователя. Не должен быть пустым после strip().

        Returns:
            Ответ модели, очищенный от ведущих/завершающих пробелов.

        Raises:
            ValueError: Если prompt пустой или состоит только из пробелов.
        """
        if not prompt.strip():
            raise ValueError("Запрос не должен быть пустым")

        # Добавляем сообщение пользователя в историю до вызова LLM
        self._history.append(ChatMessage(role="user", content=prompt))

        # Собираем полный список: системный промпт (если есть) + вся история
        messages: list[ChatMessage] = []
        if self._system_prompt:
            messages.append(ChatMessage(role="system", content=self._system_prompt))
        messages.extend(self._history)

        response = self._llm_client.generate(messages)
        reply_text = response.text.strip()

        # Добавляем ответ ассистента в историю для следующего хода
        self._history.append(ChatMessage(role="assistant", content=reply_text))

        return reply_text

    def clear_history(self) -> None:
        """Сбросить историю диалога, сохранив системный промпт."""
        self._history = []
