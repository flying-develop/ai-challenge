"""StrategyAgent: агент с подключаемой стратегией управления контекстом.

Позволяет использовать любую стратегию, реализующую ContextStrategyProtocol:
- SlidingWindowStrategy
- StickyFactsStrategy
- BranchingStrategy

Поддерживает переключение стратегии и LLM-провайдера на лету.
"""

from __future__ import annotations

from llm_agent.application.context_strategies import (
    ContextStrategyProtocol,
    StickyFactsStrategy,
)
from llm_agent.domain.models import ChatMessage, TokenUsage
from llm_agent.domain.protocols import LLMClientProtocol, TokenCounterProtocol


class StrategyAgent:
    """Агент с подключаемой стратегией управления контекстом.

    Parameters:
        llm_client: LLM-клиент для генерации ответов.
        strategy: Стратегия управления контекстом.
        system_prompt: Системный промпт.
        token_counter: Счётчик токенов (опционально).
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        strategy: ContextStrategyProtocol,
        system_prompt: str | None = None,
        token_counter: TokenCounterProtocol | None = None,
        provider_name: str = "qwen",
        model_name: str = "",
    ) -> None:
        self._llm_client = llm_client
        self._strategy = strategy
        self._system_prompt = system_prompt.strip() if system_prompt else None
        self._token_counter = token_counter
        self._provider_name = provider_name
        self._model_name = model_name
        self._last_token_usage: TokenUsage | None = None
        self._turn: int = 0
        self._total_tokens_used: int = 0

    # ------------------------------------------------------------------
    # Публичные свойства
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> ContextStrategyProtocol:
        """Текущая стратегия управления контекстом."""
        return self._strategy

    @property
    def strategy_name(self) -> str:
        """Название текущей стратегии."""
        return self._strategy.name

    @property
    def provider_name(self) -> str:
        """Название текущего провайдера (qwen / openai / claude)."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Название текущей модели."""
        return self._model_name

    @property
    def last_token_usage(self) -> TokenUsage | None:
        """Статистика токенов последнего вызова ask()."""
        return self._last_token_usage

    @property
    def turn(self) -> int:
        """Номер текущего хода."""
        return self._turn

    @property
    def total_tokens_used(self) -> int:
        """Суммарное количество токенов за все ходы."""
        return self._total_tokens_used

    # ------------------------------------------------------------------
    # Управление стратегией и провайдером
    # ------------------------------------------------------------------

    def switch_client(
        self,
        new_client: LLMClientProtocol,
        provider_name: str = "",
        model_name: str = "",
    ) -> str:
        """Переключить LLM-клиент (провайдер/модель) без сброса истории.

        Если текущая стратегия — StickyFactsStrategy, её клиент тоже обновляется.

        Args:
            new_client: Новый LLM-клиент.
            provider_name: Название провайдера (для отображения).
            model_name: Название модели (для отображения).

        Returns:
            Строка описания переключения.
        """
        old = f"{self._provider_name}/{self._model_name}" if self._model_name else self._provider_name
        self._llm_client = new_client
        if provider_name:
            self._provider_name = provider_name
        if model_name:
            self._model_name = model_name
        # Обновляем клиент внутри StickyFactsStrategy (для извлечения фактов)
        if isinstance(self._strategy, StickyFactsStrategy):
            self._strategy._llm_client = new_client
        new = f"{self._provider_name}/{self._model_name}" if self._model_name else self._provider_name
        return f"{old} -> {new}"

    def switch_strategy(self, new_strategy: ContextStrategyProtocol) -> str:
        """Переключить стратегию управления контекстом.

        При переключении текущая стратегия сбрасывается.

        Args:
            new_strategy: Новая стратегия.

        Returns:
            Название новой стратегии.
        """
        old_name = self._strategy.name
        self._strategy = new_strategy
        self._turn = 0
        self._last_token_usage = None
        return f"{old_name} -> {new_strategy.name}"

    # ------------------------------------------------------------------
    # Основная логика
    # ------------------------------------------------------------------

    def ask(self, prompt: str) -> str:
        """Отправить сообщение и получить ответ модели.

        Args:
            prompt: Сообщение пользователя.

        Returns:
            Ответ модели.

        Raises:
            ValueError: Если prompt пустой.
        """
        if not prompt.strip():
            raise ValueError("Запрос не должен быть пустым")

        self._turn += 1

        # 1. Добавляем сообщение пользователя в стратегию
        user_msg = ChatMessage(role="user", content=prompt)
        self._strategy.add_message(user_msg)

        # 2. Строим запрос через стратегию
        messages = self._strategy.build_messages(self._system_prompt)

        # 3. Подсчёт токенов
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
            self._total_tokens_used += history_tokens + response_tokens

        # 6. Добавляем ответ ассистента в стратегию
        assistant_msg = ChatMessage(role="assistant", content=reply_text)
        self._strategy.add_message(assistant_msg)

        # 7. Уведомляем стратегию о завершённом обмене
        self._strategy.on_response(user_msg, assistant_msg)

        return reply_text

    def clear_history(self) -> None:
        """Сбросить историю и стратегию."""
        self._strategy.reset()
        self._last_token_usage = None
        self._turn = 0

    def get_stats(self) -> dict:
        """Получить полную статистику агента и стратегии."""
        stats = {
            "provider": self._provider_name,
            "model": self._model_name,
            "turn": self._turn,
            "total_tokens_used": self._total_tokens_used,
        }
        stats.update(self._strategy.get_stats())
        return stats
