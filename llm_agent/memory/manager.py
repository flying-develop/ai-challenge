"""MemoryManager: управление тремя слоями памяти агента.

Слои:
    - SHORT-TERM  — текущий диалог (очищается при /clear)
    - WORKING     — данные текущей задачи (явное управление)
    - LONG-TERM   — профиль, решения, знания (никогда не очищается автоматически)

Хранение — SQLite, три отдельные таблицы.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from llm_agent.memory.models import (
    LongTermMemoryEntry,
    ShortTermEntry,
    WorkingMemoryEntry,
)


class MemoryManager:
    """Менеджер трёхуровневой памяти на базе SQLite.

    Parameters:
        db_path: Путь к файлу SQLite (создаётся автоматически).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._setup_schema()

    # ------------------------------------------------------------------
    # Схема БД
    # ------------------------------------------------------------------

    def _setup_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL DEFAULT 'default',
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                ts         TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_short_term_session
                ON short_term_memory (session_id, id);

            CREATE TABLE IF NOT EXISTS working_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT    NOT NULL,
                value      TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS long_term_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT    NOT NULL,
                value      TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                tags       TEXT    NOT NULL DEFAULT ''
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # SHORT-TERM
    # ------------------------------------------------------------------

    def add_to_short(
        self, role: str, content: str, session_id: str = "default",
    ) -> None:
        """Добавить запись в краткосрочную память."""
        self._conn.execute(
            "INSERT INTO short_term_memory (session_id, role, content) "
            "VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self._conn.commit()

    def get_short_term(
        self, session_id: str = "default",
    ) -> list[ShortTermEntry]:
        """Получить все записи краткосрочной памяти для сессии."""
        rows = self._conn.execute(
            "SELECT role, content, ts FROM short_term_memory "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            ShortTermEntry(role=r["role"], content=r["content"], ts=r["ts"])
            for r in rows
        ]

    def clear_short_term(self, session_id: str = "default") -> None:
        """Очистить краткосрочную память для сессии."""
        self._conn.execute(
            "DELETE FROM short_term_memory WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # WORKING MEMORY
    # ------------------------------------------------------------------

    def add_to_working(self, key: str, value: str) -> int:
        """Добавить запись в рабочую память. Возвращает id записи."""
        cur = self._conn.execute(
            "INSERT INTO working_memory (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_working(self) -> list[WorkingMemoryEntry]:
        """Получить все записи рабочей памяти."""
        rows = self._conn.execute(
            "SELECT id, key, value, created_at, updated_at "
            "FROM working_memory ORDER BY id",
        ).fetchall()
        return [
            WorkingMemoryEntry(
                id=r["id"],
                key=r["key"],
                value=r["value"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def remove_from_working(self, entry_id: int | None = None) -> int:
        """Удалить из рабочей памяти. Без id — удаляет всё. Возвращает кол-во удалённых."""
        if entry_id is not None:
            cur = self._conn.execute(
                "DELETE FROM working_memory WHERE id = ?", (entry_id,),
            )
        else:
            cur = self._conn.execute("DELETE FROM working_memory")
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # LONG-TERM MEMORY
    # ------------------------------------------------------------------

    def add_to_long(
        self, key: str, value: str, tags: list[str] | None = None,
    ) -> int:
        """Добавить запись в долговременную память. Возвращает id записи."""
        tags_str = ",".join(tags) if tags else ""
        cur = self._conn.execute(
            "INSERT INTO long_term_memory (key, value, tags) VALUES (?, ?, ?)",
            (key, value, tags_str),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_long_term(self) -> list[LongTermMemoryEntry]:
        """Получить все записи долговременной памяти."""
        rows = self._conn.execute(
            "SELECT id, key, value, created_at, tags "
            "FROM long_term_memory ORDER BY id",
        ).fetchall()
        return [
            LongTermMemoryEntry(
                id=r["id"],
                key=r["key"],
                value=r["value"],
                created_at=r["created_at"],
                tags=[t for t in r["tags"].split(",") if t],
            )
            for r in rows
        ]

    def remove_from_long(self, entry_id: int | None = None) -> int:
        """Удалить из долговременной памяти. Без id — удаляет всё. Возвращает кол-во удалённых."""
        if entry_id is not None:
            cur = self._conn.execute(
                "DELETE FROM long_term_memory WHERE id = ?", (entry_id,),
            )
        else:
            cur = self._conn.execute("DELETE FROM long_term_memory")
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # PROMOTE (working → long_term)
    # ------------------------------------------------------------------

    def promote(self, from_layer: str, entry_id: int) -> int:
        """Переместить запись из одного слоя в долговременную память.

        Сейчас поддерживается только working → long_term.

        Returns:
            id новой записи в long_term.

        Raises:
            ValueError: Если слой не поддерживается или запись не найдена.
        """
        if from_layer != "working":
            raise ValueError(
                f"Promote поддерживается только из 'working', получено: {from_layer!r}"
            )
        row = self._conn.execute(
            "SELECT id, key, value FROM working_memory WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Запись working#{entry_id} не найдена")

        new_id = self.add_to_long(key=row["key"], value=row["value"])
        self.remove_from_working(entry_id)
        return new_id

    # ------------------------------------------------------------------
    # КОНТЕКСТ ДЛЯ LLM
    # ------------------------------------------------------------------

    def get_context_for_llm(self) -> dict[str, str]:
        """Собрать текстовые блоки из working и long-term для system prompt.

        Returns:
            dict с ключами ``"working_text"`` и ``"long_term_text"``.
            Пустая строка если слой пуст.
        """
        # Long-term
        long_entries = self.get_long_term()
        if long_entries:
            lines = [f"- {e.key}: {e.value}" for e in long_entries]
            long_text = (
                "ИЗВЕСТНЫЕ ФАКТЫ О ПОЛЬЗОВАТЕЛЕ (Long-term Memory):\n"
                + "\n".join(lines)
            )
        else:
            long_text = ""

        # Working
        working_entries = self.get_working()
        if working_entries:
            lines = [f"- {e.key}: {e.value}" for e in working_entries]
            working_text = (
                "ТЕКУЩИЙ КОНТЕКСТ ЗАДАЧИ (Working Memory):\n"
                + "\n".join(lines)
            )
        else:
            working_text = ""

        return {"working_text": working_text, "long_term_text": long_text}

    # ------------------------------------------------------------------
    # СТАТИСТИКА
    # ------------------------------------------------------------------

    def stats(self, session_id: str = "default") -> dict[str, int]:
        """Статистика по количеству записей в каждом слое."""
        short_count = self._conn.execute(
            "SELECT COUNT(*) FROM short_term_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        working_count = self._conn.execute(
            "SELECT COUNT(*) FROM working_memory",
        ).fetchone()[0]
        long_count = self._conn.execute(
            "SELECT COUNT(*) FROM long_term_memory",
        ).fetchone()[0]
        return {
            "short_term_count": short_count,
            "working_count": working_count,
            "long_term_count": long_count,
        }

    # ------------------------------------------------------------------
    # Управление ресурсом
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Закрыть соединение с БД."""
        self._conn.close()

    def __enter__(self) -> MemoryManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
