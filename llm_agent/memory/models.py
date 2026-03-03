"""Модели данных для слоёв памяти (Memory Layers)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShortTermEntry:
    """Запись краткосрочной памяти — один ход диалога."""

    role: str
    content: str
    ts: str


@dataclass
class WorkingMemoryEntry:
    """Запись рабочей памяти — данные текущей задачи."""

    id: int
    key: str
    value: str
    created_at: str
    updated_at: str


@dataclass
class LongTermMemoryEntry:
    """Запись долговременной памяти — профиль, решения, знания."""

    id: int
    key: str
    value: str
    created_at: str
    tags: list[str] = field(default_factory=list)
