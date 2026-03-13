"""Состояния исследовательского оркестратора.

Образовательная концепция: явное перечисление состояний через Enum
делает флоу читаемым и предотвращает опечатки в строках.

State Machine:
  TASK_RECEIVED → SEARCH_INITIAL → FETCH_INITIAL → SUMMARIZE_INITIAL
  → SEARCH_DEEP → FETCH_DEEP → SYNTHESIZE_FINAL → DELIVERED
  Из любого состояния → FAILED (при критической ошибке)
"""

from enum import Enum


class ResearchState(str, Enum):
    """Состояния задачи в исследовательском флоу."""

    TASK_RECEIVED = "TASK_RECEIVED"
    SEARCH_INITIAL = "SEARCH_INITIAL"
    FETCH_INITIAL = "FETCH_INITIAL"
    SUMMARIZE_INITIAL = "SUMMARIZE_INITIAL"
    SEARCH_DEEP = "SEARCH_DEEP"
    FETCH_DEEP = "FETCH_DEEP"
    SYNTHESIZE_FINAL = "SYNTHESIZE_FINAL"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"

    # Определён ли это терминальный статус
    @property
    def is_terminal(self) -> bool:
        return self in (ResearchState.DELIVERED, ResearchState.FAILED)

    # Telegram-сообщение для каждого состояния
    @property
    def progress_message(self) -> str:
        return _STATE_MESSAGES.get(self, f"[{self.value}]")


_STATE_MESSAGES: dict[ResearchState, str] = {
    ResearchState.TASK_RECEIVED:     "🎬 Принял задачу, начинаю исследование...",
    ResearchState.SEARCH_INITIAL:    "🔍 Ищу информацию...",
    ResearchState.FETCH_INITIAL:     "📄 Читаю страницы...",
    ResearchState.SUMMARIZE_INITIAL: "🧠 Анализирую собранные данные...",
    ResearchState.SEARCH_DEEP:       "📚 Ищу дополнительные материалы...",
    ResearchState.FETCH_DEEP:        "📄 Читаю дополнительные страницы...",
    ResearchState.SYNTHESIZE_FINAL:  "✨ Формирую финальный ответ...",
    ResearchState.DELIVERED:         "✅ Готово!",
    ResearchState.FAILED:            "❌ Ошибка выполнения",
}

# Допустимые переходы (граф состояний)
VALID_TRANSITIONS: dict[ResearchState, list[ResearchState]] = {
    ResearchState.TASK_RECEIVED:     [ResearchState.SEARCH_INITIAL, ResearchState.FAILED],
    ResearchState.SEARCH_INITIAL:    [ResearchState.FETCH_INITIAL, ResearchState.FAILED],
    ResearchState.FETCH_INITIAL:     [ResearchState.SUMMARIZE_INITIAL, ResearchState.FAILED],
    ResearchState.SUMMARIZE_INITIAL: [ResearchState.SEARCH_DEEP, ResearchState.FAILED],
    ResearchState.SEARCH_DEEP:       [ResearchState.FETCH_DEEP, ResearchState.FAILED],
    ResearchState.FETCH_DEEP:        [ResearchState.SYNTHESIZE_FINAL, ResearchState.FAILED],
    ResearchState.SYNTHESIZE_FINAL:  [ResearchState.DELIVERED, ResearchState.FAILED],
    ResearchState.DELIVERED:         [],
    ResearchState.FAILED:            [],
}
