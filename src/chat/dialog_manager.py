"""DialogManager: управление диалогом одного пользователя.

Один экземпляр на пользователя. Связывает историю (short_term memory),
task state (working_memory), RAG-пайплайн и LLM.

RAG-индекс и LLM-провайдер — общие (shared между всеми пользователями).
История и task state — изолированные (per-user).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

# Добавляем корень проекта в sys.path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_agent.memory.manager import MemoryManager
from .task_state_extractor import TaskStateExtractor


# Системный промпт для чат-режима
CHAT_SYSTEM_PROMPT = """Ты — support-бот по документации проекта.
Ты помогаешь пользователям разобраться с установкой и настройкой.

КОНТЕКСТ ЗАДАЧИ (из памяти):
{task_state}

ПРАВИЛА:
1. Отвечай на основе предоставленного контекста из документации.
2. Учитывай предыдущие сообщения и уточнения пользователя.
3. Если пользователь уточнил параметры (роутер, протокол и т.д.) — используй их в ответе, не спрашивай повторно.
4. Если информации нет — честно скажи и предложи уточнить вопрос.
5. Возвращай ответ в формате [ANSWER]/[SOURCES]/[QUOTES].
6. Если пользователь сменил тему — адаптируйся, не цепляйся за старую."""

_HISTORY_LIMIT = int(os.environ.get("CHAT_HISTORY_LIMIT", "20"))


class DialogManager:
    """Управляет диалогом ОДНОГО пользователя.

    Каждый пользователь Telegram получает свой экземпляр с
    изолированной историей и task state.
    RAG-пайплайн и LLM-провайдер — общие (shared).

    Args:
        user_id:          ID пользователя (chat_id из Telegram).
        memory_manager:   Экземпляр MemoryManager (per-user).
        rag_pipeline:     Общий RAGPipeline (shared).
        llm_fn:           Функция (system, user) → str. Используется для task state.
        history_limit:    Размер sliding window истории.
    """

    def __init__(
        self,
        user_id: str,
        memory_manager: MemoryManager,
        rag_pipeline,
        llm_fn: Callable[[str, str], str],
        history_limit: int = _HISTORY_LIMIT,
    ) -> None:
        self.user_id = user_id
        self.memory = memory_manager
        self.rag = rag_pipeline
        self.llm_fn = llm_fn
        self.history_limit = history_limit
        self._session_id = f"user_{user_id}"
        self._message_count = 0
        self._last_rag_answer = None  # для /sources

        # Экстрактор task state
        self._state_extractor = TaskStateExtractor(llm_fn=llm_fn)

    def process_message(self, text: str) -> object:
        """Обработать сообщение пользователя и вернуть RAGAnswer.

        Цепочка:
            1. Сохранить в short_term
            2. Обновить task state (по условию)
            3. Обогатить запрос контекстом из task state
            4. RAG pipeline (поиск → реранкинг → LLM)
            5. Сохранить ответ в short_term

        Args:
            text: Текст сообщения пользователя.

        Returns:
            RAGAnswer с ответом, источниками и цитатами.
        """
        is_first = self._message_count == 0

        # 1. Сохранить вопрос в историю
        self.memory.add_to_short("user", text, session_id=self._session_id)

        # 2. Обновить task state
        if self._state_extractor.should_update(text, is_first=is_first):
            self._update_task_state(text)
        else:
            self._state_extractor.tick()

        self._message_count += 1

        # 3. Обогатить запрос контекстом
        enriched_query = self._enrich_query(text)

        # 4. RAG pipeline
        extra_context = self._get_task_state_context()
        rag_answer = self.rag.answer(
            question=enriched_query,
            top_k=5,
            initial_k=20,
            extra_context=extra_context,
        )
        self._last_rag_answer = rag_answer

        # 5. Сохранить ответ в историю
        answer_text = rag_answer.answer
        if rag_answer.structured:
            answer_text = rag_answer.structured.answer

        self.memory.add_to_short("assistant", answer_text, session_id=self._session_id)

        # Обрезаем историю (sliding window)
        self._trim_history()

        return rag_answer

    def get_task_state(self) -> dict:
        """Получить текущий task state из working_memory.

        Returns:
            dict с ключами goal, constraints, clarified, stage.
        """
        prefix = f"task_state_{self.user_id}:"
        entries = self.memory.get_working()
        state = {}
        for entry in entries:
            if entry.key.startswith(prefix):
                field_name = entry.key[len(prefix):]
                state[field_name] = entry.value
        return state

    def format_task_state(self) -> str:
        """Форматировать task state для отображения пользователю.

        Returns:
            Человекочитаемая строка с текущим состоянием задачи.
        """
        state = self.get_task_state()
        if not state:
            return "Задача не определена"

        lines = []
        if state.get("goal"):
            lines.append(f"Цель: {state['goal']}")
        if state.get("constraints"):
            lines.append(f"Параметры: {state['constraints']}")
        if state.get("stage"):
            lines.append(f"Этап: {state['stage']}")
        if state.get("clarified"):
            lines.append(f"Уточнено: {state['clarified']}")

        return "\n".join(lines) if lines else "Задача не определена"

    def get_history(self) -> list[dict]:
        """Получить историю диалога.

        Returns:
            Список {"role": ..., "content": ...}.
        """
        entries = self.memory.get_short_term(session_id=self._session_id)
        return [{"role": e.role, "content": e.content} for e in entries]

    def reset(self) -> None:
        """Сбросить историю и task state (новая тема)."""
        self.memory.clear_short_term(session_id=self._session_id)
        # Удаляем task state этого пользователя
        prefix = f"task_state_{self.user_id}:"
        self.memory.remove_working_by_key_prefix(prefix)
        self._message_count = 0
        self._state_extractor._messages_since_update = 0
        self._last_rag_answer = None

    def get_last_sources(self) -> list[str]:
        """Получить источники последнего ответа.

        Returns:
            Список строк "source: section_title".
        """
        if not self._last_rag_answer:
            return []

        sources = []
        if self._last_rag_answer.structured and self._last_rag_answer.structured.sources:
            # SourceRef: поля file + section
            for src in self._last_rag_answer.structured.sources:
                sources.append(f"`{src.file}` — {src.section}")
        elif self._last_rag_answer.sources:
            # RetrievalResult: поля source + section
            for src in self._last_rag_answer.sources[:5]:
                sources.append(f"`{src.source}` — {src.section}")

        return sources

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _update_task_state(self, message: str) -> None:
        """Обновить task state через LLM.

        Извлекает структурированное состояние задачи и сохраняет
        в working_memory как key-value пары.
        """
        history = self.get_history()
        prev_state = self.get_task_state()

        new_state = self._state_extractor.extract(
            recent_messages=history[-10:],
            prev_state=prev_state,
        )

        if not new_state:
            return

        # Обновляем working_memory (удаляем старые, записываем новые)
        prefix = f"task_state_{self.user_id}:"
        self.memory.remove_working_by_key_prefix(prefix)

        fields_to_save = ["goal", "constraints", "clarified", "stage"]
        for field in fields_to_save:
            value = new_state.get(field, "")
            if value:
                self.memory.add_to_working(f"{prefix}{field}", str(value))

    def _enrich_query(self, question: str) -> str:
        """Обогатить RAG-запрос ключевыми словами из task state.

        Добавляет контекст из task state к вопросу без перефразирования.
        Это даёт RAG-поиску больше сигналов для поиска релевантных чанков.

        Args:
            question: Исходный вопрос пользователя.

        Returns:
            Вопрос + ключевые слова из task state.
        """
        state = self.get_task_state()
        constraints = state.get("constraints", "")
        goal = state.get("goal", "")

        if not constraints and not goal:
            return question

        context_keywords = " ".join(filter(None, [goal, constraints]))
        return f"{question} (контекст: {context_keywords})"

    def _get_task_state_context(self) -> str:
        """Получить task state как строку для system prompt.

        Returns:
            Форматированный текст для подстановки в {task_state}.
        """
        state = self.get_task_state()
        if not state:
            return "Начало диалога, контекст не определён."

        parts = []
        if state.get("goal"):
            parts.append(f"Цель: {state['goal']}")
        if state.get("constraints"):
            parts.append(f"Ограничения: {state['constraints']}")
        if state.get("stage"):
            parts.append(f"Этап: {state['stage']}")

        return "\n".join(parts) if parts else "Контекст не определён."

    def _trim_history(self) -> None:
        """Обрезать историю до sliding window (history_limit сообщений).

        Удаляет старые записи если история превышает лимит.
        """
        entries = self.memory.get_short_term(session_id=self._session_id)
        if len(entries) > self.history_limit:
            # Удаляем все записи и перезаписываем последние N
            self.memory.clear_short_term(session_id=self._session_id)
            for entry in entries[-self.history_limit:]:
                self.memory.add_to_short(
                    entry.role, entry.content,
                    session_id=self._session_id,
                )
