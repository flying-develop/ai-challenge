"""SimpleAgent: оркестратор взаимодействий с LLM."""

from __future__ import annotations

from llm_agent.domain.models import ChatMessage, ContextLimitError, TokenUsage
from llm_agent.domain.protocols import (
    ChatHistoryRepositoryProtocol,
    LLMClientProtocol,
    TokenCounterProtocol,
)


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

    Если передан ``token_counter``, агент считает токены при каждом ask()
    и сохраняет статистику в ``last_token_usage``. При указании ``context_limit``
    агент проверяет размер контекста перед отправкой:
    - ``auto_truncate=False`` (по умолчанию): вызывает ContextLimitError
    - ``auto_truncate=True``: обрезает старую историю, чтобы влезть в лимит
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        system_prompt: str | None = None,
        history_repo: ChatHistoryRepositoryProtocol | None = None,
        token_counter: TokenCounterProtocol | None = None,
        context_limit: int | None = None,
        auto_truncate: bool = False,
    ) -> None:
        self._llm_client = llm_client
        self._system_prompt = system_prompt.strip() if system_prompt else None
        self._history_repo = history_repo
        self._token_counter = token_counter
        self._context_limit = context_limit
        self._auto_truncate = auto_truncate
        # Загружаем историю из хранилища (если оно передано), иначе начинаем с чистого листа
        self._history: list[ChatMessage] = history_repo.load() if history_repo else []
        self._last_token_usage: TokenUsage | None = None

    # ------------------------------------------------------------------
    # Публичные свойства
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ChatMessage]:
        """Текущая история диалога (копия, без системного промпта)."""
        return list(self._history)

    @property
    def last_token_usage(self) -> TokenUsage | None:
        """Статистика токенов последнего вызова ask() или None если счётчик не задан."""
        return self._last_token_usage

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
            ContextLimitError: Если контекст превышает лимит и auto_truncate=False.
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

        # --- Подсчёт токенов до отправки ---
        request_tokens = 0
        history_tokens = 0
        if self._token_counter:
            request_tokens = self._token_counter.count_tokens(prompt)
            history_tokens = self._token_counter.count_messages_tokens(messages)

            if self._context_limit and history_tokens > self._context_limit:
                if self._auto_truncate:
                    messages = self._truncate_to_fit(messages, self._context_limit)
                    history_tokens = self._token_counter.count_messages_tokens(messages)
                else:
                    raise ContextLimitError(
                        tokens=history_tokens, limit=self._context_limit
                    )

        response = self._llm_client.generate(messages)
        reply_text = response.text.strip()

        # --- Фиксируем статистику токенов ---
        if self._token_counter:
            response_tokens = self._token_counter.count_tokens(reply_text)
            # Если API вернул usage — предпочитаем его (точнее локального счётчика)
            if response.usage:
                history_tokens = response.usage.get("prompt_tokens", history_tokens)
                response_tokens = response.usage.get("completion_tokens", response_tokens)
            self._last_token_usage = TokenUsage(
                request_tokens=request_tokens,
                history_tokens=history_tokens,
                response_tokens=response_tokens,
                total_tokens=history_tokens + response_tokens,
                context_limit=self._context_limit or 0,
            )

        assistant_msg = ChatMessage(role="assistant", content=reply_text)
        self._history.append(assistant_msg)
        if self._history_repo:
            self._history_repo.append(assistant_msg)

        return reply_text

    def clear_history(self) -> None:
        """Сбросить историю диалога (и в памяти, и в хранилище), сохранив системный промпт."""
        self._history = []
        self._last_token_usage = None
        if self._history_repo:
            self._history_repo.clear()

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _truncate_to_fit(
        self, messages: list[ChatMessage], limit: int
    ) -> list[ChatMessage]:
        """Обрезать историю так, чтобы уложиться в лимит токенов.

        Всегда сохраняет системное сообщение (если есть) и последнее
        сообщение пользователя. Удаляет старые пары (user + assistant)
        из середины истории до тех пор, пока контекст не влезет в лимит.
        """
        system_msgs = [m for m in messages if m.role == "system"]
        last_msg = messages[-1]
        # Средняя часть: история без системного промпта и без текущего запроса
        middle = list(messages[len(system_msgs):-1])

        while middle and self._token_counter:
            candidate = system_msgs + middle + [last_msg]
            if self._token_counter.count_messages_tokens(candidate) <= limit:
                return candidate
            # Удаляем старейшую пару (user + assistant)
            middle = middle[2:] if len(middle) >= 2 else middle[1:]

        return system_msgs + [last_msg]
