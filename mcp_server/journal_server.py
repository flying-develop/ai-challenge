"""MCP-сервер: аудит-журнал этапов исследования.

Образовательная концепция: отдельный сервер для аудита позволяет
видеть полную историю выполнения независимо от агента. Полезен
для отладки и демонстрации работы оркестратора.

Хранилище: SQLite (stdlib), таблица research_journal.
БД создаётся автоматически при первом запуске.

Запуск:
    python -m mcp_server.journal_server

Инструменты:
    log_stage — записать этап в журнал
    get_log   — получить полный лог задачи
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

mcp = FastMCP("Journal Server")


def _get_db_path() -> str:
    """Путь к SQLite-базе журнала."""
    default = os.path.join(os.path.expanduser("~"), ".llm-agent", "research_journal.db")
    path = os.environ.get("JOURNAL_DB_PATH", default).strip()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _init_db(db_path: str) -> None:
    """Создать таблицу журнала если не существует."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_journal (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id   TEXT NOT NULL,
                stage     TEXT NOT NULL,
                status    TEXT NOT NULL,
                details   TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_id ON research_journal(task_id)")
        conn.commit()


# Инициализируем БД при старте модуля
_DB_PATH = _get_db_path()
try:
    _init_db(_DB_PATH)
except Exception as _init_exc:
    import sys
    print(f"[journal_server] Предупреждение: не удалось инициализировать БД: {_init_exc}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------

@mcp.tool()
def log_stage(task_id: str, stage: str, status: str, details: str = "") -> str:
    """
    Записать этап выполнения задачи в журнал (SQLite).

    Args:
        task_id: Уникальный идентификатор задачи
        stage: Название этапа (TASK_RECEIVED, SEARCH_INITIAL, ...)
        status: started | complete | failed
        details: Дополнительная информация (кол-во ссылок, ошибки)

    Returns:
        Подтверждение записи с timestamp
    """
    db_path = _get_db_path()
    _init_db(db_path)

    ts = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO research_journal (task_id, stage, status, details, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, stage, status, details, ts),
            )
            conn.commit()
    except sqlite3.Error as exc:
        return f"Ошибка записи в журнал: {exc}"

    return f"[{ts[:19]}] {task_id} | {stage} | {status}"


@mcp.tool()
def get_log(task_id: str) -> str:
    """
    Получить полный лог выполнения задачи.

    Args:
        task_id: Идентификатор задачи

    Returns:
        Таблица этапов с таймингами в формате текста.
        Для пустого task_id возвращает все последние записи.
    """
    db_path = _get_db_path()
    _init_db(db_path)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            if task_id:
                rows = conn.execute(
                    "SELECT stage, status, details, timestamp "
                    "FROM research_journal WHERE task_id = ? ORDER BY id",
                    (task_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT task_id, stage, status, details, timestamp "
                    "FROM research_journal ORDER BY id DESC LIMIT 50"
                ).fetchall()
    except sqlite3.Error as exc:
        return f"Ошибка чтения журнала: {exc}"

    if not rows:
        return f"Журнал для задачи '{task_id}' пуст."

    # Форматируем таблицу с вычислением времени между этапами
    lines = []
    if task_id:
        lines.append(f"Журнал задачи: {task_id}")
        lines.append(f"{'Этап':<22} {'Статус':<10} {'Время':<8} {'Детали'}")
        lines.append("─" * 70)
        prev_ts: float | None = None
        for row in rows:
            try:
                ts_dt = datetime.fromisoformat(row["timestamp"])
                ts_unix = ts_dt.timestamp()
            except ValueError:
                ts_unix = 0.0
            elapsed = f"+{ts_unix - prev_ts:.1f}s" if prev_ts else "0.0s"
            prev_ts = ts_unix
            details_short = str(row["details"])[:30]
            lines.append(
                f"{row['stage']:<22} {row['status']:<10} {elapsed:<8} {details_short}"
            )
    else:
        lines.append("Последние записи журнала:")
        lines.append(f"{'task_id':<20} {'Этап':<22} {'Статус':<10} {'Время UTC'}")
        lines.append("─" * 80)
        for row in rows:
            ts_short = str(row["timestamp"])[:19].replace("T", " ")
            lines.append(
                f"{str(row['task_id'])[:18]:<20} {row['stage']:<22} "
                f"{row['status']:<10} {ts_short}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print(f"Journal DB: {_DB_PATH}", file=sys.stderr)
    mcp.run()
