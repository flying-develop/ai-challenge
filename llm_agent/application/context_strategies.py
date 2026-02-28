"""Стратегии управления контекстом диалога.

Реализованы 3 стратегии:

1. SlidingWindowStrategy — хранит только последние N сообщений, остальное отбрасывает.
2. StickyFactsStrategy — извлекает ключевые факты из диалога в блок «facts» (ключ-значение),
   отправляет facts + последние N сообщений.
3. BranchingStrategy — поддерживает checkpoint-и и ветки диалога,
   позволяет переключаться между ветками.

Все стратегии реализуют единый протокол ContextStrategyProtocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from llm_agent.domain.models import ChatMessage
from llm_agent.domain.protocols import LLMClientProtocol


# ---------------------------------------------------------------------------
# Протокол стратегии
# ---------------------------------------------------------------------------

@runtime_checkable
class ContextStrategyProtocol(Protocol):
    """Единый интерфейс для всех стратегий управления контекстом."""

    @property
    def name(self) -> str:
        """Человекочитаемое название стратегии."""
        ...

    def add_message(self, message: ChatMessage) -> None:
        """Добавить сообщение в контекст."""
        ...

    def build_messages(self, system_prompt: str | None = None) -> list[ChatMessage]:
        """Построить список сообщений для отправки в LLM."""
        ...

    def on_response(self, user_msg: ChatMessage, assistant_msg: ChatMessage) -> None:
        """Хук, вызываемый после получения ответа от LLM.

        Позволяет стратегии обновить внутреннее состояние
        (например, извлечь факты из диалога).
        """
        ...

    def reset(self) -> None:
        """Сбросить всё состояние стратегии."""
        ...

    def get_stats(self) -> dict:
        """Вернуть статистику стратегии для сравнительного анализа."""
        ...


# ===========================================================================
# Стратегия 1: Sliding Window
# ===========================================================================

class SlidingWindowStrategy:
    """Хранит только последние N сообщений, всё остальное отбрасывает.

    Параметр window_size задаёт максимальное количество сообщений в окне.
    Системный промпт не входит в счёт окна.
    """

    def __init__(self, window_size: int = 10) -> None:
        if window_size < 2:
            raise ValueError("window_size должен быть >= 2")
        self._window_size = window_size
        self._messages: list[ChatMessage] = []
        self._total_added: int = 0
        self._total_dropped: int = 0

    @property
    def name(self) -> str:
        return f"Sliding Window (N={self._window_size})"

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def add_message(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self._total_added += 1
        self._trim()

    def build_messages(self, system_prompt: str | None = None) -> list[ChatMessage]:
        result: list[ChatMessage] = []
        if system_prompt:
            result.append(ChatMessage(role="system", content=system_prompt))
        result.extend(self._messages)
        return result

    def on_response(self, user_msg: ChatMessage, assistant_msg: ChatMessage) -> None:
        pass

    def reset(self) -> None:
        self._messages = []
        self._total_added = 0
        self._total_dropped = 0

    def get_stats(self) -> dict:
        return {
            "strategy": self.name,
            "window_size": self._window_size,
            "current_messages": len(self._messages),
            "total_added": self._total_added,
            "total_dropped": self._total_dropped,
        }

    def _trim(self) -> None:
        """Обрезать историю до window_size."""
        while len(self._messages) > self._window_size:
            self._messages.pop(0)
            self._total_dropped += 1


# ===========================================================================
# Стратегия 2: Sticky Facts / Key-Value Memory
# ===========================================================================

# Системный промпт для извлечения фактов
_FACTS_EXTRACTION_PROMPT = (
    "Ты ассистент, который извлекает ключевые факты из диалога.\n"
    "На вход ты получаешь текущий набор фактов (ключ: значение) и последний обмен "
    "сообщениями (вопрос пользователя + ответ ассистента).\n\n"
    "Твоя задача:\n"
    "1. Проанализировать новый обмен.\n"
    "2. Обновить/добавить/удалить факты.\n"
    "3. Вернуть ТОЛЬКО обновлённый набор фактов в формате:\n"
    "   КЛЮЧ: значение\n"
    "   КЛЮЧ: значение\n"
    "   ...\n\n"
    "Категории фактов:\n"
    "- ЦЕЛЬ: основная цель пользователя\n"
    "- ТЕМА: тема обсуждения\n"
    "- РЕШЕНИЯ: принятые решения\n"
    "- ОГРАНИЧЕНИЯ: известные ограничения\n"
    "- ПРЕДПОЧТЕНИЯ: предпочтения пользователя\n"
    "- ТЕХНОЛОГИИ: упомянутые технологии/инструменты\n"
    "- ДОГОВОРЁННОСТИ: согласованные договорённости\n"
    "- КОНТЕКСТ: важный контекст\n\n"
    "Не добавляй пустые факты. Не добавляй комментарии. Только ключ: значение.\n"
    "Если нового ничего нет — верни старые факты без изменений."
)


@dataclass
class FactsStore:
    """Хранилище фактов (ключ-значение)."""

    facts: dict[str, str] = field(default_factory=dict)

    def to_text(self) -> str:
        """Преобразовать факты в текстовое представление."""
        if not self.facts:
            return "(пусто)"
        return "\n".join(f"{k}: {v}" for k, v in self.facts.items())

    def parse_from_text(self, text: str) -> None:
        """Обновить факты из текста LLM-ответа."""
        self.facts.clear()
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().upper()
                value = value.strip()
                if key and value:
                    self.facts[key] = value


class StickyFactsStrategy:
    """Хранит блок «facts» (ключ-значение) + последние N сообщений.

    После каждого обмена (user + assistant) вызывает LLM для обновления фактов.
    В запрос отправляет: system_prompt + facts-блок + последние N сообщений.

    Parameters:
        window_size: Количество последних сообщений (помимо facts).
        llm_client: LLM-клиент для извлечения фактов. Если None — факты не обновляются.
    """

    def __init__(
        self,
        window_size: int = 6,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size должен быть >= 2")
        self._window_size = window_size
        self._llm_client = llm_client
        self._messages: list[ChatMessage] = []
        self._facts = FactsStore()
        self._total_added: int = 0
        self._facts_update_count: int = 0
        self._facts_update_tokens: int = 0

    @property
    def name(self) -> str:
        return f"Sticky Facts (N={self._window_size})"

    @property
    def facts(self) -> dict[str, str]:
        return dict(self._facts.facts)

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def add_message(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self._total_added += 1
        self._trim()

    def build_messages(self, system_prompt: str | None = None) -> list[ChatMessage]:
        result: list[ChatMessage] = []
        if system_prompt:
            result.append(ChatMessage(role="system", content=system_prompt))

        # Добавляем блок фактов как системное сообщение
        if self._facts.facts:
            facts_text = (
                "ВАЖНЫЕ ФАКТЫ ИЗ ДИАЛОГА (используй их как контекст):\n\n"
                + self._facts.to_text()
            )
            result.append(ChatMessage(role="system", content=facts_text))

        result.extend(self._messages)
        return result

    def on_response(self, user_msg: ChatMessage, assistant_msg: ChatMessage) -> None:
        """Обновить факты после получения ответа."""
        if not self._llm_client:
            return
        self._update_facts(user_msg, assistant_msg)

    def reset(self) -> None:
        self._messages = []
        self._facts = FactsStore()
        self._total_added = 0
        self._facts_update_count = 0
        self._facts_update_tokens = 0

    def get_stats(self) -> dict:
        return {
            "strategy": self.name,
            "window_size": self._window_size,
            "current_messages": len(self._messages),
            "total_added": self._total_added,
            "facts_count": len(self._facts.facts),
            "facts": dict(self._facts.facts),
            "facts_update_count": self._facts_update_count,
            "facts_update_tokens": self._facts_update_tokens,
        }

    def _trim(self) -> None:
        while len(self._messages) > self._window_size:
            self._messages.pop(0)

    def _update_facts(self, user_msg: ChatMessage, assistant_msg: ChatMessage) -> None:
        """Вызвать LLM для обновления фактов."""
        current_facts_text = self._facts.to_text()
        exchange_text = (
            f"Пользователь: {user_msg.content}\n"
            f"Ассистент: {assistant_msg.content}"
        )
        request = [
            ChatMessage(role="system", content=_FACTS_EXTRACTION_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Текущие факты:\n{current_facts_text}\n\n"
                    f"Последний обмен:\n{exchange_text}\n\n"
                    f"Верни обновлённые факты:"
                ),
            ),
        ]
        response = self._llm_client.generate(request)  # type: ignore[union-attr]
        self._facts.parse_from_text(response.text)
        self._facts_update_count += 1
        if response.usage:
            self._facts_update_tokens += response.usage.get("total_tokens", 0)


# ===========================================================================
# Стратегия 3: Branching (ветки диалога)
# ===========================================================================

@dataclass
class DialogCheckpoint:
    """Снимок состояния диалога в определённой точке."""

    checkpoint_id: str
    messages: list[ChatMessage]
    turn: int
    description: str = ""


@dataclass
class DialogBranch:
    """Ветка диалога, начинающаяся от определённого checkpoint."""

    branch_id: str
    checkpoint_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    description: str = ""


class BranchingStrategy:
    """Поддерживает checkpoint-и и ветки диалога.

    Позволяет:
    - Сохранить checkpoint в текущей точке диалога
    - Создать ветку от checkpoint-а
    - Переключаться между ветками
    - Продолжать диалог в каждой ветке независимо

    По умолчанию работает в основной ветке «main».
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, DialogCheckpoint] = {}
        self._branches: dict[str, DialogBranch] = {}
        self._main_messages: list[ChatMessage] = []
        self._current_branch: str | None = None  # None = main
        self._turn: int = 0
        self._total_added: int = 0

    @property
    def name(self) -> str:
        branch = self._current_branch or "main"
        return f"Branching (branch={branch})"

    @property
    def current_branch_id(self) -> str:
        return self._current_branch or "main"

    @property
    def checkpoints(self) -> list[str]:
        return list(self._checkpoints.keys())

    @property
    def branches(self) -> list[str]:
        return ["main"] + list(self._branches.keys())

    @property
    def messages(self) -> list[ChatMessage]:
        """Текущие сообщения (main или активная ветка)."""
        if self._current_branch and self._current_branch in self._branches:
            branch = self._branches[self._current_branch]
            cp = self._checkpoints[branch.checkpoint_id]
            return list(cp.messages) + list(branch.messages)
        return list(self._main_messages)

    def add_message(self, message: ChatMessage) -> None:
        self._total_added += 1
        self._turn += 1
        if self._current_branch and self._current_branch in self._branches:
            self._branches[self._current_branch].messages.append(message)
        else:
            self._main_messages.append(message)

    def build_messages(self, system_prompt: str | None = None) -> list[ChatMessage]:
        result: list[ChatMessage] = []
        if system_prompt:
            result.append(ChatMessage(role="system", content=system_prompt))
        result.extend(self.messages)
        return result

    def on_response(self, user_msg: ChatMessage, assistant_msg: ChatMessage) -> None:
        pass

    def reset(self) -> None:
        self._checkpoints = {}
        self._branches = {}
        self._main_messages = []
        self._current_branch = None
        self._turn = 0
        self._total_added = 0

    def get_stats(self) -> dict:
        return {
            "strategy": self.name,
            "current_branch": self.current_branch_id,
            "checkpoints": list(self._checkpoints.keys()),
            "branches": self.branches,
            "current_messages": len(self.messages),
            "total_added": self._total_added,
            "turn": self._turn,
        }

    # ------------------------------------------------------------------
    # Методы управления ветками
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_id: str, description: str = "") -> DialogCheckpoint:
        """Сохранить текущее состояние диалога как checkpoint.

        Args:
            checkpoint_id: Уникальный идентификатор checkpoint-а.
            description: Описание checkpoint-а.

        Returns:
            Созданный DialogCheckpoint.

        Raises:
            ValueError: Если checkpoint_id уже существует или мы в ветке.
        """
        if checkpoint_id in self._checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id}' уже существует")

        # Сохраняем текущие сообщения (из main или из текущей ветки)
        current_msgs = self.messages

        cp = DialogCheckpoint(
            checkpoint_id=checkpoint_id,
            messages=copy.deepcopy(current_msgs),
            turn=self._turn,
            description=description,
        )
        self._checkpoints[checkpoint_id] = cp
        return cp

    def create_branch(
        self, branch_id: str, checkpoint_id: str, description: str = ""
    ) -> DialogBranch:
        """Создать новую ветку от указанного checkpoint-а.

        Args:
            branch_id: Уникальный идентификатор ветки.
            checkpoint_id: От какого checkpoint-а ответвляемся.
            description: Описание ветки.

        Returns:
            Созданная DialogBranch.

        Raises:
            ValueError: Если branch_id уже существует или checkpoint_id не найден.
        """
        if branch_id in self._branches or branch_id == "main":
            raise ValueError(f"Ветка '{branch_id}' уже существует")
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id}' не найден")

        branch = DialogBranch(
            branch_id=branch_id,
            checkpoint_id=checkpoint_id,
            description=description,
        )
        self._branches[branch_id] = branch
        return branch

    def switch_branch(self, branch_id: str) -> None:
        """Переключиться на указанную ветку.

        Args:
            branch_id: Идентификатор ветки («main» для основной).

        Raises:
            ValueError: Если ветка не найдена.
        """
        if branch_id == "main":
            self._current_branch = None
            return
        if branch_id not in self._branches:
            raise ValueError(
                f"Ветка '{branch_id}' не найдена. "
                f"Доступные: {', '.join(self.branches)}"
            )
        self._current_branch = branch_id

    def get_branch_info(self, branch_id: str) -> dict:
        """Получить информацию о ветке."""
        if branch_id == "main":
            return {
                "branch_id": "main",
                "messages_count": len(self._main_messages),
                "description": "Основная ветка диалога",
            }
        if branch_id not in self._branches:
            raise ValueError(f"Ветка '{branch_id}' не найдена")
        branch = self._branches[branch_id]
        cp = self._checkpoints[branch.checkpoint_id]
        return {
            "branch_id": branch_id,
            "checkpoint_id": branch.checkpoint_id,
            "checkpoint_messages": len(cp.messages),
            "branch_messages": len(branch.messages),
            "total_messages": len(cp.messages) + len(branch.messages),
            "description": branch.description,
        }
