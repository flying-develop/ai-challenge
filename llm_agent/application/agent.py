"""SimpleAgent: оркестратор взаимодействий с LLM."""

from __future__ import annotations

from llm_agent.domain.models import ChatMessage
from llm_agent.domain.protocols import ChatHistoryRepositoryProtocol, LLMClientProtocol


class SimpleAgent:
    """Агент приложения, делегирующий запросы LLM-клиенту.

    Хранит историю диалога между вызовами ask().
    Принимает любой объект, удовлетворяющий LLMClientProtocol (структурная типизация).
    Не знает об HTTP, эндпоинтах или аутентификации — эти детали
    относятся к инфраструктурному слою.

    Если передан ``history_repo`` (объект, удовлетворяющий
    ChatHistoryRepositoryProtocol), агент при инициализации загружает
    предыдущую историю из хранилища и сохраняет каждое новое сообщение
    сразу после его формирования.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        system_prompt: str | None = None,
        history_repo: ChatHistoryRepositoryProtocol | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._system_prompt = system_prompt.strip() if system_prompt else None
        self._history_repo = history_repo
        # Загружаем историю из хранилища (если оно передано), иначе начинаем с чистого листа
        self._history: list[ChatMessage] = history_repo.load() if history_repo else []

    # ------------------------------------------------------------------
    # Публичные свойства
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ChatMessage]:
        """Текущая история диалога (копия, без системного промпта)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Основная логика
    # ------------------------------------------------------------------

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

        user_msg = ChatMessage(role="user", content=prompt)
        self._history.append(user_msg)
        if self._history_repo:
            self._history_repo.append(user_msg)

        # Собираем полный список: системный промпт (если есть) + вся история
        messages: list[ChatMessage] = []
        if self._system_prompt:
            messages.append(ChatMessage(role="system", content=self._system_prompt))
        messages.extend(self._history)

        response = self._llm_client.generate(messages)
        reply_text = response.text.strip()

        assistant_msg = ChatMessage(role="assistant", content=reply_text)
        self._history.append(assistant_msg)
        if self._history_repo:
            self._history_repo.append(assistant_msg)

        return reply_text

    def clear_history(self) -> None:
        """Сбросить историю диалога (и в памяти, и в хранилище), сохранив системный промпт."""
        self._history = []
        if self._history_repo:
            self._history_repo.clear()
