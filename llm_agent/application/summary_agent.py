"""SummaryAgent: агент с управлением контекстом через суммаризацию.

В отличие от SimpleAgent, который накапливает полную историю диалога,
SummaryAgent периодически «сворачивает» старые сообщения в краткое резюме.
Это позволяет поддерживать длинные диалоги, не выходя за пределы контекстного
окна модели и экономя токены.

Параметры управления:
    summary_batch_size  — сколько сообщений (user + assistant) накапливать
                          перед созданием резюме. При достижении этого числа
                          выполняется суммаризация, окно очищается.

Структура запроса к LLM при каждом ходе:
    [system_prompt]          — инструкции (если заданы)
    [резюме 1..N]            — system-сообщение со всеми накопленными резюме
    [текущее окно]           — последние M сообщений (M < summary_batch_size)
"""

from __future__ import annotations

from llm_agent.application.context_manager import ContextManager
from llm_agent.domain.models import ChatMessage, TokenUsage
from llm_agent.domain.protocols import LLMClientProtocol, TokenCounterProtocol


class SummaryAgent:
    """Агент с автоматическим суммаризированием контекста.

    Parameters:
        llm_client: LLM-клиент для генерации ответов. Тот же клиент
                    используется для создания резюме.
        summary_batch_size: Количество сообщений в одном «окне».
                            После заполнения окна создаётся резюме.
        system_prompt: Системный промпт для основного диалога.
        token_counter: Счётчик токенов (опционально). При наличии
                       заполняет last_token_usage после каждого ask().
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        summary_batch_size: int = 10,
        system_prompt: str | None = None,
        token_counter: TokenCounterProtocol | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._system_prompt = system_prompt.strip() if system_prompt else None
        self._token_counter = token_counter
        self._context_manager = ContextManager(
            summary_batch_size=summary_batch_size,
            llm_client=llm_client,
        )
        self._last_token_usage: TokenUsage | None = None
        self._turn: int = 0  # счётчик ходов

    # ------------------------------------------------------------------
    # Публичные свойства
    # ------------------------------------------------------------------

    @property
    def last_token_usage(self) -> TokenUsage | None:
        """Статистика токенов последнего вызова ask() или None."""
        return self._last_token_usage

    @property
    def context_manager(self) -> ContextManager:
        """Доступ к менеджеру контекста для инспекции состояния."""
        return self._context_manager

    @property
    def turn(self) -> int:
        """Номер текущего хода (начиная с 1)."""
        return self._turn

    # ------------------------------------------------------------------
    # Основная логика
    # ------------------------------------------------------------------

    def ask(self, prompt: str) -> str:
        """Отправить сообщение и получить ответ модели.

        Поток выполнения:
            1. Добавить user-сообщение в текущее окно.
            2. Построить список сообщений: system + резюме + текущее окно.
            3. Вызвать LLM.
            4. Добавить ответ ассистента в текущее окно.
            5. Если окно заполнено (>= summary_batch_size) — суммаризировать.

        Args:
            prompt: Сообщение пользователя. Не должно быть пустым.

        Returns:
            Ответ модели, очищенный от пробелов по краям.

        Raises:
            ValueError: Если prompt пустой.
        """
        if not prompt.strip():
            raise ValueError("Запрос не должен быть пустым")

        self._turn += 1

        # 1. Добавляем сообщение пользователя в текущее окно
        user_msg = ChatMessage(role="user", content=prompt)
        self._context_manager.add_message(user_msg)

        # 2. Строим запрос: system + все резюме + текущее окно
        messages = self._context_manager.build_request_messages(self._system_prompt)

        # 3. Подсчёт токенов до запроса
        request_tokens = 0
        history_tokens = 0
        if self._token_counter:
            request_tokens = self._token_counter.count_tokens(prompt)
            history_tokens = self._token_counter.count_messages_tokens(messages)

        # 4. Вызов LLM
        response = self._llm_client.generate(messages)
        reply_text = response.text.strip()

        # 5. Фиксируем статистику токенов
        if self._token_counter:
            response_tokens = self._token_counter.count_tokens(reply_text)
            if response.usage:
                history_tokens = response.usage.get("prompt_tokens", history_tokens)
                response_tokens = response.usage.get("completion_tokens", response_tokens)
            self._last_token_usage = TokenUsage(
                request_tokens=request_tokens,
                history_tokens=history_tokens,
                response_tokens=response_tokens,
                total_tokens=history_tokens + response_tokens,
                context_limit=0,
            )

        # 6. Добавляем ответ ассистента в текущее окно
        assistant_msg = ChatMessage(role="assistant", content=reply_text)
        self._context_manager.add_message(assistant_msg)

        # 7. Суммаризация, если окно заполнено
        self._context_manager.maybe_summarize()

        return reply_text

    def clear_history(self) -> None:
        """Сбросить всю историю, резюме и счётчики."""
        self._context_manager.reset()
        self._last_token_usage = None
        self._turn = 0
