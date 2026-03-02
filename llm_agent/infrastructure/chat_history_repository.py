"""SQLite-хранилище истории чата."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from llm_agent.domain.models import ChatMessage


class SQLiteChatHistoryRepository:
    """Сохраняет историю диалогов в локальной SQLite-базе данных.

    Каждая «сессия» — это независимый диалог, идентифицируемый строковым именем.
    По умолчанию используется сессия ``"default"``.

    DB-файл создаётся автоматически по пути ``db_path`` (включая родительские
    директории). Рекомендуемое расположение: ``~/.llm-agent/history.db``.

    Поддерживает использование как контекстный менеджер (with-блок).
    """

    def __init__(self, db_path: str | Path, session_id: str = "default") -> None:
        self._db_path = Path(db_path)
        self._session_id = session_id
        # Создаём директорию, если нужно
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        # Включаем каскадное удаление через внешние ключи
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._setup_schema()

    # ------------------------------------------------------------------
    # Схема БД
    # ------------------------------------------------------------------

    def _setup_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL
                               REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages (session_id, id);
        """)
        self._conn.commit()
        # Гарантируем наличие текущей сессии
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (id) VALUES (?)",
            (self._session_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Публичный API (ChatHistoryRepositoryProtocol)
    # ------------------------------------------------------------------

    def load(self) -> list[ChatMessage]:
        """Загрузить все сообщения текущей сессии в хронологическом порядке."""
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (self._session_id,),
        ).fetchall()
        return [ChatMessage(role=row["role"], content=row["content"]) for row in rows]

    def append(self, message: ChatMessage) -> None:
        """Добавить сообщение в конец истории текущей сессии."""
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (self._session_id, message.role, message.content),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Удалить все сообщения текущей сессии (сессия остаётся в БД)."""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (self._session_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Дополнительные методы (управление сессиями)
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[str]:
        """Вернуть имена всех существующих сессий (по дате создания)."""
        rows = self._conn.execute(
            "SELECT id FROM sessions ORDER BY created_at, id"
        ).fetchall()
        return [row["id"] for row in rows]

    def delete_session(self, session_id: str) -> None:
        """Полностью удалить сессию и все её сообщения из БД."""
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    @property
    def session_id(self) -> str:
        """Идентификатор текущей сессии."""
        return self._session_id

    def message_count(self) -> int:
        """Количество сообщений в текущей сессии."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (self._session_id,),
        ).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Управление ресурсом
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Закрыть соединение с БД."""
        self._conn.close()

    def __enter__(self) -> SQLiteChatHistoryRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
