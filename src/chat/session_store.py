"""Хранилище пользовательских сессий.

Каждая сессия (chat_id) имеет:
    - Изолированную историю диалога (sliding window)
    - Изолированный task state (working memory)
    - Метаданные: username, время последнего сообщения

История хранится in-memory (session_id используется как ключ в MemoryManager).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionInfo:
    """Метаданные пользовательской сессии.

    Attributes:
        chat_id:       ID чата Telegram.
        username:      Имя пользователя.
        last_seen:     Unix timestamp последнего сообщения.
        message_count: Общее количество сообщений в сессии.
        last_sources:  Источники последнего ответа (для /sources).
    """

    chat_id: str
    username: str
    last_seen: float = field(default_factory=time.time)
    message_count: int = 0
    last_sources: list[str] = field(default_factory=list)


class SessionStore:
    """Хранилище сессий пользователей.

    Каждый chat_id получает изолированную SessionInfo.
    DialogManager создаётся фабрикой и хранится отдельно.

    Args:
        max_idle_seconds: Время неактивности до "устаревания" сессии (для /users).
    """

    def __init__(self, max_idle_seconds: int = 3600) -> None:
        self._sessions: dict[str, SessionInfo] = {}
        self.max_idle_seconds = max_idle_seconds

    def get_or_create(self, chat_id: str, username: str) -> SessionInfo:
        """Получить или создать сессию для пользователя.

        Args:
            chat_id:  Telegram chat ID.
            username: Имя пользователя.

        Returns:
            Объект SessionInfo.
        """
        if chat_id not in self._sessions:
            self._sessions[chat_id] = SessionInfo(
                chat_id=chat_id,
                username=username,
            )
        return self._sessions[chat_id]

    def touch(self, chat_id: str) -> None:
        """Обновить время последнего сообщения."""
        if chat_id in self._sessions:
            self._sessions[chat_id].last_seen = time.time()

    def increment(self, chat_id: str) -> None:
        """Увеличить счётчик сообщений."""
        if chat_id in self._sessions:
            self._sessions[chat_id].message_count += 1

    def set_last_sources(self, chat_id: str, sources: list[str]) -> None:
        """Сохранить источники последнего ответа."""
        if chat_id in self._sessions:
            self._sessions[chat_id].last_sources = sources

    def get_active_sessions(self) -> list[SessionInfo]:
        """Получить активные сессии (были активны в пределах max_idle_seconds).

        Returns:
            Список SessionInfo, отсортированных по last_seen (новые первые).
        """
        now = time.time()
        active = [
            s for s in self._sessions.values()
            if (now - s.last_seen) < self.max_idle_seconds
        ]
        return sorted(active, key=lambda s: s.last_seen, reverse=True)

    def all_sessions(self) -> list[SessionInfo]:
        """Получить все сессии."""
        return list(self._sessions.values())

    def __len__(self) -> int:
        return len(self._sessions)
