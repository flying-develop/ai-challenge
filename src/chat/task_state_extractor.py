"""Экстрактор task state через LLM.

Анализирует историю диалога и извлекает структурированное состояние задачи
(цель, ограничения, этап) в виде JSON.

Обновление выполняется НЕ на каждое сообщение, а по условиям:
    - Первое сообщение в диалоге
    - Прошло > CHAT_TASK_STATE_UPDATE_INTERVAL сообщений с последнего обновления
    - Сообщение содержит ключевые маркеры (ошибка, новая тема, уточнение)
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional


# Промпт для извлечения task state
TASK_STATE_PROMPT = """Проанализируй диалог и извлеки текущее состояние задачи пользователя.
Верни ТОЛЬКО валидный JSON, без пояснений, без markdown:
{
  "goal": "краткая цель пользователя (1 предложение)",
  "constraints": "выявленные ограничения/параметры (роутер, ОС, провайдер, протокол...)",
  "clarified": "что пользователь уже уточнил",
  "stage": "на каком этапе сейчас (начало / установка / настройка / отладка / решено)",
  "changed": true
}

Предыдущее состояние:
{prev_state}

Последние сообщения диалога:
{recent_messages}

Верни ТОЛЬКО JSON."""

# Маркеры, требующие немедленного обновления task state
_UPDATE_MARKERS = [
    r"ошибка", r"не работает", r"проблема", r"error",
    r"установ", r"настро", r"конфигур",
    r"роутер", r"протокол", r"провайдер",
    r"вернёмся", r"другой вопрос", r"забыл",
    r"а как", r"а что", r"а можно",
]
_UPDATE_RE = re.compile("|".join(_UPDATE_MARKERS), re.IGNORECASE)


class TaskStateExtractor:
    """Извлекает и обновляет task state через LLM.

    Args:
        llm_fn:   Функция (system_prompt, user_prompt) → str. Лёгкий LLM-вызов.
        interval: Обновлять task state каждые N сообщений.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str],
        interval: int | None = None,
    ) -> None:
        self.llm_fn = llm_fn
        self.interval = interval or _get_env_int(
            "CHAT_TASK_STATE_UPDATE_INTERVAL", 3
        )
        self._messages_since_update: int = 0

    def should_update(self, message: str, is_first: bool = False) -> bool:
        """Определить нужно ли обновлять task state.

        Args:
            message:  Последнее сообщение пользователя.
            is_first: True если это первое сообщение в диалоге.

        Returns:
            True если нужно запустить обновление.
        """
        if is_first:
            return True
        if self._messages_since_update >= self.interval:
            return True
        if _UPDATE_RE.search(message):
            return True
        return False

    def extract(
        self,
        recent_messages: list[dict],
        prev_state: Optional[dict] = None,
    ) -> dict:
        """Извлечь task state через LLM.

        Args:
            recent_messages: Последние сообщения диалога.
                             Каждое: {"role": "user"|"assistant", "content": "..."}
            prev_state:      Предыдущее состояние (для сравнения).

        Returns:
            dict с ключами: goal, constraints, clarified, stage, changed.
            При ошибке LLM — возвращает prev_state или пустой dict.
        """
        self._messages_since_update = 0

        # Форматируем историю диалога
        messages_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent_messages[-10:]
        )
        prev_state_text = json.dumps(prev_state or {}, ensure_ascii=False)

        # Используем replace вместо .format() — prev_state_text содержит
        # JSON с фигурными скобками, которые ломают str.format()
        prompt = (TASK_STATE_PROMPT
                  .replace("{prev_state}", prev_state_text)
                  .replace("{recent_messages}", messages_text))

        try:
            raw = self.llm_fn("Ты — анализатор диалогов. Отвечай ТОЛЬКО JSON.", prompt)
            # Ищем JSON-объект: ищем первый { и последний } в ответе
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                state = json.loads(raw[start:end])
                return state
            print(f"[TaskState] JSON не найден в ответе: {raw[:200]}")
        except json.JSONDecodeError as exc:
            print(f"[TaskState] Ошибка парсинга JSON: {exc} | raw={raw[:200]}")
        except Exception as exc:
            print(f"[TaskState] Ошибка LLM: {exc}")

        return prev_state or {}

    def tick(self) -> None:
        """Зафиксировать одно сообщение (для счётчика интервала)."""
        self._messages_since_update += 1


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default
