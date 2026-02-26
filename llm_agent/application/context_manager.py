"""Менеджер контекста с автоматическим суммаризированием истории диалога.

Логика работы:
    - Сообщения накапливаются в «текущем окне» (current_messages).
    - Когда окно достигает summary_batch_size сообщений, вызывается суммаризация:
        1. Текущее окно отправляется в LLM с просьбой создать резюме.
        2. Резюме сохраняется в списке summaries.
        3. Текущее окно очищается.
    - При построении запроса: [system_prompt] + [все_резюме] + [текущее_окно].
    - Таким образом длина запроса = f(кол-во резюме × размер резюме + размер окна),
      а не f(полная история).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_agent.domain.models import ChatMessage
from llm_agent.domain.protocols import LLMClientProtocol

# ---------------------------------------------------------------------------
# Системный промпт для суммаризации
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "Ты ассистент, который создаёт краткое, но информативное резюме диалога. "
    "Сохрани все ключевые факты, вопросы пользователя, ответы ассистента и важный контекст. "
    "Резюме должно позволить продолжить разговор без потери смысла. "
    "Отвечай только самим резюме — без вводных фраз и пояснений."
)


# ---------------------------------------------------------------------------
# Вспомогательная структура
# ---------------------------------------------------------------------------

@dataclass
class SummaryRecord:
    """Одно резюме порции диалога."""

    index: int                  # порядковый номер резюме (1, 2, 3, …)
    content: str                # текст резюме
    messages_count: int         # сколько сообщений суммаризировано
    tokens_spent: dict = field(default_factory=dict)  # токены на суммаризацию


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class ContextManager:
    """Управляет историей диалога с автоматическим суммаризированием.

    Parameters:
        summary_batch_size: Количество сообщений (user + assistant) в одном окне.
                            После заполнения окна создаётся резюме.
        llm_client: LLM-клиент для генерации резюме. Если None — суммаризация
                    не выполняется (полезно для тестов).
    """

    def __init__(
        self,
        summary_batch_size: int = 10,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        if summary_batch_size < 2:
            raise ValueError("summary_batch_size должен быть >= 2")
        self._batch_size = summary_batch_size
        self._llm_client = llm_client
        self._current_messages: list[ChatMessage] = []
        self._summaries: list[SummaryRecord] = []

    # ------------------------------------------------------------------
    # Свойства (read-only, возвращают копии)
    # ------------------------------------------------------------------

    @property
    def summary_batch_size(self) -> int:
        return self._batch_size

    @property
    def current_messages(self) -> list[ChatMessage]:
        """Сообщения в текущем (ещё не суммаризированном) окне."""
        return list(self._current_messages)

    @property
    def summaries(self) -> list[SummaryRecord]:
        """Все созданные резюме (в порядке создания)."""
        return list(self._summaries)

    @property
    def summary_count(self) -> int:
        return len(self._summaries)

    @property
    def current_window_size(self) -> int:
        return len(self._current_messages)

    @property
    def total_messages_processed(self) -> int:
        """Полное количество сообщений (текущее окно + суммаризованные)."""
        summarized = sum(s.messages_count for s in self._summaries)
        return summarized + len(self._current_messages)

    @property
    def summary_tokens_spent(self) -> int:
        """Суммарные токены, потраченные на все операции суммаризации."""
        return sum(
            s.tokens_spent.get("total_tokens", 0) for s in self._summaries
        )

    # ------------------------------------------------------------------
    # Основные методы
    # ------------------------------------------------------------------

    def add_message(self, message: ChatMessage) -> None:
        """Добавить сообщение в текущее окно."""
        self._current_messages.append(message)

    def maybe_summarize(self) -> bool:
        """Суммаризировать текущее окно, если оно заполнено.

        Returns:
            True, если суммаризация была выполнена в этом вызове.
        """
        if len(self._current_messages) < self._batch_size:
            return False
        if not self._llm_client:
            return False
        self._create_summary()
        return True

    def build_request_messages(
        self, system_prompt: str | None = None
    ) -> list[ChatMessage]:
        """Построить список сообщений для запроса к LLM.

        Структура:
            1. [system_prompt]         — основные инструкции (если заданы)
            2. [резюме предыдущих частей] — как system-сообщение (если есть)
            3. [текущее окно]          — последние N сообщений

        Returns:
            Список ChatMessage, готовый для передачи в LLMClientProtocol.generate().
        """
        messages: list[ChatMessage] = []

        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

        if self._summaries:
            parts = [
                f"[Резюме части {s.index} — {s.messages_count} сообщ.]\n{s.content}"
                for s in self._summaries
            ]
            combined = (
                "Ниже приведены резюме предыдущих частей нашего разговора. "
                "Используй их как контекст при ответах.\n\n"
                + "\n\n---\n\n".join(parts)
            )
            messages.append(ChatMessage(role="system", content=combined))

        messages.extend(self._current_messages)
        return messages

    def reset(self) -> None:
        """Сбросить всё состояние."""
        self._current_messages = []
        self._summaries = []

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _create_summary(self) -> None:
        """Суммаризировать текущее окно и очистить его."""
        messages_to_summarize = list(self._current_messages)

        # Формируем текстовое представление диалога для суммаризации
        dialog_text = "\n\n".join(
            f"{'Пользователь' if m.role == 'user' else 'Ассистент'}: {m.content}"
            for m in messages_to_summarize
        )

        summary_request = [
            ChatMessage(role="system", content=_SUMMARY_SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    f"Создай резюме следующего диалога "
                    f"({len(messages_to_summarize)} сообщений):\n\n{dialog_text}"
                ),
            ),
        ]

        response = self._llm_client.generate(summary_request)  # type: ignore[union-attr]
        summary_text = response.text.strip()

        tokens_spent = {}
        if response.usage:
            tokens_spent = {
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
            }

        record = SummaryRecord(
            index=len(self._summaries) + 1,
            content=summary_text,
            messages_count=len(messages_to_summarize),
            tokens_spent=tokens_spent,
        )
        self._summaries.append(record)

        # Очищаем текущее окно
        self._current_messages = []
