"""Персистентное хранилище новостей на SQLite.

Образовательная концепция: MCP-сервер может иметь собственную базу данных
для накопления данных между вызовами. SQLite идеален для встроенного хранилища —
один файл, без отдельного процесса, полный SQL.

Три таблицы:
- headlines: заголовки новостей (дедупликация по guid)
- daily_digests: сводки дня, сгенерированные LLM
- scheduler_log: лог выполнения фоновых задач
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, date
from pathlib import Path


_CREATE_HEADLINES = """
CREATE TABLE IF NOT EXISTS headlines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guid        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    link        TEXT NOT NULL,
    pub_date    TEXT NOT NULL,
    category    TEXT,
    fetched_at  TEXT NOT NULL
)
"""

_CREATE_DAILY_DIGESTS = """
CREATE TABLE IF NOT EXISTS daily_digests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT UNIQUE NOT NULL,
    headline_count  INTEGER NOT NULL,
    digest_text     TEXT NOT NULL,
    created_at      TEXT NOT NULL
)
"""

_CREATE_SCHEDULER_LOG = """
CREATE TABLE IF NOT EXISTS scheduler_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type   TEXT NOT NULL,
    status      TEXT NOT NULL,
    details     TEXT,
    executed_at TEXT NOT NULL
)
"""


class NewsStorage:
    """SQLite-хранилище новостей и сводок.

    Образовательный паттерн: хранилище инкапсулирует всю работу с БД.
    Бизнес-логика (планировщик, MCP-инструменты) работает с этим классом,
    не зная деталей SQL.
    """

    def __init__(self, db_path: str | Path = "data/news.db") -> None:
        self._db_path = Path(db_path)
        # Создаём директорию, если не существует
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        # WAL-режим: позволяет читать БД пока идёт запись (из фонового потока)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Создать таблицы при первом запуске."""
        with self._connect() as conn:
            conn.execute(_CREATE_HEADLINES)
            conn.execute(_CREATE_DAILY_DIGESTS)
            conn.execute(_CREATE_SCHEDULER_LOG)
            conn.commit()

    # ------------------------------------------------------------------
    # Заголовки
    # ------------------------------------------------------------------

    def add_headlines(self, items: list[dict]) -> int:
        """Сохранить список заголовков. Возвращает количество НОВЫХ записей.

        Дедупликация: INSERT OR IGNORE по UNIQUE(guid).
        Дублирующиеся записи молча пропускаются.

        Args:
            items: список dict с ключами guid, title, link, pub_date, category

        Returns:
            Количество реально добавленных (новых) записей
        """
        if not items:
            return 0

        fetched_at = datetime.utcnow().isoformat()
        new_count = 0

        with self._connect() as conn:
            for item in items:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO headlines
                        (guid, title, link, pub_date, category, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.get("guid", ""),
                        item.get("title", ""),
                        item.get("link", ""),
                        item.get("pub_date", ""),
                        item.get("category", ""),
                        fetched_at,
                    ),
                )
                if cursor.rowcount > 0:
                    new_count += 1
            conn.commit()

        return new_count

    def get_headlines(self, date: str | None = None, limit: int = 50) -> list[dict]:
        """Получить заголовки с опциональной фильтрацией по дате.

        Args:
            date: Дата в формате YYYY-MM-DD. Если None — без фильтра по дате.
            limit: Максимальное количество записей (по умолчанию 50).

        Returns:
            Список dict с полями: id, guid, title, link, pub_date, category, fetched_at
        """
        limit = min(limit, 1000)

        with self._connect() as conn:
            if date:
                # Фильтр по дате публикации (ISO 8601 начинается с YYYY-MM-DD)
                rows = conn.execute(
                    """
                    SELECT id, guid, title, link, pub_date, category, fetched_at
                    FROM headlines
                    WHERE pub_date LIKE ?
                    ORDER BY pub_date DESC
                    LIMIT ?
                    """,
                    (f"{date}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, guid, title, link, pub_date, category, fetched_at
                    FROM headlines
                    ORDER BY pub_date DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]

    def get_today_headlines(self) -> list[dict]:
        """Получить все заголовки за сегодня."""
        today = date.today().isoformat()
        return self.get_headlines(date=today, limit=1000)

    def count_headlines(self) -> int:
        """Общее количество заголовков в базе."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM headlines").fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Сводки дня
    # ------------------------------------------------------------------

    def save_digest(self, date: str, text: str, count: int) -> None:
        """Сохранить (или обновить) дневную сводку.

        Args:
            date: Дата сводки (YYYY-MM-DD)
            text: Текст сводки от LLM
            count: Количество заголовков, использованных для генерации
        """
        created_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_digests (date, headline_count, digest_text, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    headline_count = excluded.headline_count,
                    digest_text    = excluded.digest_text,
                    created_at     = excluded.created_at
                """,
                (date, count, text, created_at),
            )
            conn.commit()

    def get_digest(self, date: str) -> dict | None:
        """Получить сводку за конкретную дату.

        Returns:
            dict с полями date, headline_count, digest_text, created_at
            или None если сводка не найдена
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT date, headline_count, digest_text, created_at
                FROM daily_digests
                WHERE date = ?
                """,
                (date,),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_digest(self) -> dict | None:
        """Получить самую последнюю сводку.

        Returns:
            dict с полями date, headline_count, digest_text, created_at
            или None если сводок нет
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT date, headline_count, digest_text, created_at
                FROM daily_digests
                ORDER BY date DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Лог планировщика
    # ------------------------------------------------------------------

    def log_task(self, task_type: str, status: str, details: str | None = None) -> None:
        """Записать результат выполнения задачи планировщика.

        Args:
            task_type: Тип задачи ("fetch_rss" | "make_digest")
            status: Статус ("success" | "error")
            details: Детали (количество новых записей или текст ошибки)
        """
        executed_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduler_log (task_type, status, details, executed_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_type, status, details, executed_at),
            )
            conn.commit()

    def get_scheduler_status(self) -> dict:
        """Получить статус планировщика: последние выполнения каждого типа задач.

        Returns:
            dict с ключами:
            - last_fetch: dict (task_type, status, details, executed_at) или None
            - last_digest: dict (task_type, status, details, executed_at) или None
            - total_headlines: int
        """
        with self._connect() as conn:
            # Последний сбор RSS
            last_fetch_row = conn.execute(
                """
                SELECT task_type, status, details, executed_at
                FROM scheduler_log
                WHERE task_type = 'fetch_rss'
                ORDER BY executed_at DESC
                LIMIT 1
                """
            ).fetchone()

            # Последняя генерация сводки
            last_digest_row = conn.execute(
                """
                SELECT task_type, status, details, executed_at
                FROM scheduler_log
                WHERE task_type = 'make_digest'
                ORDER BY executed_at DESC
                LIMIT 1
                """
            ).fetchone()

            total_row = conn.execute("SELECT COUNT(*) as cnt FROM headlines").fetchone()

        return {
            "last_fetch": dict(last_fetch_row) if last_fetch_row else None,
            "last_digest": dict(last_digest_row) if last_digest_row else None,
            "total_headlines": total_row["cnt"] if total_row else 0,
        }
