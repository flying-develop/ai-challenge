"""
CRM MCP-сервер для системы поддержки пользователей documaker.
Транспорт: stdio (FastMCP).
База: crm.db (SQLite, в корне проекта).

Запуск:
    python mcp_server/crm_server.py
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parent.parent / "crm.db"

mcp = FastMCP("CRM Support Server")

_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_OPEN_STATUSES = ("open", "in_progress", "waiting")
_VALID_STATUSES = {"open", "in_progress", "waiting", "resolved", "closed"}
_VALID_PRIORITIES = {"low", "medium", "high", "critical"}
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "open":        {"in_progress"},
    "in_progress": {"waiting", "resolved"},
    "waiting":     {"in_progress", "resolved"},
    "resolved":    {"in_progress", "closed"},
    "closed":      {"in_progress"},
}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                plan        TEXT DEFAULT 'free',
                created_at  TEXT NOT NULL,
                last_seen   TEXT,
                metadata    TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS tickets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id),
                title        TEXT NOT NULL,
                description  TEXT NOT NULL,
                status       TEXT DEFAULT 'open',
                priority     TEXT DEFAULT 'medium',
                category     TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                resolved_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS ticket_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id   INTEGER NOT NULL REFERENCES tickets(id),
                actor       TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
            CREATE INDEX IF NOT EXISTS idx_tickets_user   ON tickets(user_id);
            CREATE INDEX IF NOT EXISTS idx_events_ticket  ON ticket_events(ticket_id);
        """)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# Инициализируем БД при запуске сервера
_init_db()


# ---------------------------------------------------------------------------
# Инструменты MCP
# ---------------------------------------------------------------------------

@mcp.tool()
def get_open_tickets(limit: int = 10, priority: str = "") -> str:
    """
    Получить открытые тикеты (open | in_progress | waiting).

    Args:
        limit: Максимальное количество тикетов (по умолчанию 10)
        priority: Фильтр по приоритету: low | medium | high | critical (опционально)

    Returns:
        JSON-список [{id, title, status, priority, category, user_name, user_plan,
        created_at, last_event_at, events_count}], отсортированных по приоритету
    """
    with _conn() as conn:
        placeholders = ",".join("?" * len(_OPEN_STATUSES))
        params: list = list(_OPEN_STATUSES)
        extra_where = ""
        if priority and priority in _VALID_PRIORITIES:
            extra_where = " AND t.priority = ?"
            params.append(priority)
        params.append(limit)

        rows = conn.execute(f"""
            SELECT
                t.id, t.title, t.status, t.priority, t.category,
                t.created_at, t.updated_at,
                u.name  AS user_name,
                u.plan  AS user_plan,
                (SELECT MAX(e.created_at)
                 FROM ticket_events e WHERE e.ticket_id = t.id) AS last_event_at,
                (SELECT COUNT(*)
                 FROM ticket_events e WHERE e.ticket_id = t.id) AS events_count
            FROM tickets t
            JOIN users u ON t.user_id = u.id
            WHERE t.status IN ({placeholders}){extra_where}
            LIMIT ?
        """, params).fetchall()

    tickets = [dict(r) for r in rows]
    tickets.sort(key=lambda x: (
        _PRIORITY_ORDER.get(x["priority"], 99),
        x["created_at"],
    ))
    return json.dumps(tickets, ensure_ascii=False)


@mcp.tool()
def get_ticket_details(ticket_id: int) -> str:
    """
    Получить полную информацию по тикету, включая историю событий.

    Args:
        ticket_id: ID тикета

    Returns:
        JSON: {ticket: {...}, user: {name, email, plan, metadata}, events: [...]}
    """
    with _conn() as conn:
        row = conn.execute("""
            SELECT t.*,
                   u.name     AS user_name,
                   u.email    AS user_email,
                   u.plan     AS user_plan,
                   u.metadata AS user_metadata
            FROM tickets t
            JOIN users u ON t.user_id = u.id
            WHERE t.id = ?
        """, (ticket_id,)).fetchone()

        if not row:
            return json.dumps({"error": f"Тикет #{ticket_id} не найден"}, ensure_ascii=False)

        events = conn.execute("""
            SELECT actor, event_type, content, created_at
            FROM ticket_events
            WHERE ticket_id = ?
            ORDER BY created_at, id
        """, (ticket_id,)).fetchall()

    t = dict(row)
    user = {
        "name":     t.pop("user_name"),
        "email":    t.pop("user_email"),
        "plan":     t.pop("user_plan"),
        "metadata": json.loads(t.pop("user_metadata") or "{}"),
    }
    return json.dumps({
        "ticket": t,
        "user":   user,
        "events": [dict(e) for e in events],
    }, ensure_ascii=False)


@mcp.tool()
def create_ticket(
    user_email: str,
    title: str,
    description: str,
    category: str = "other",
    priority: str = "medium",
) -> str:
    """
    Создать новый тикет поддержки.
    Если пользователь не найден — создаётся автоматически с plan='free'.

    Args:
        user_email: Email пользователя
        title: Заголовок тикета
        description: Описание проблемы
        category: auth | export | ui | performance | billing | other
        priority: low | medium | high | critical

    Returns:
        JSON: {ticket_id, user_id, status: 'created'}
    """
    now = _now()
    priority = priority if priority in _VALID_PRIORITIES else "medium"

    with _conn() as conn:
        user_row = conn.execute(
            "SELECT id FROM users WHERE email = ?", (user_email,)
        ).fetchone()

        if user_row:
            user_id = user_row["id"]
            conn.execute("UPDATE users SET last_seen = ? WHERE id = ?", (now, user_id))
        else:
            name = user_email.split("@")[0].replace(".", " ").title()
            cur = conn.execute(
                "INSERT INTO users (name, email, plan, created_at, last_seen) VALUES (?, ?, 'free', ?, ?)",
                (name, user_email, now, now),
            )
            user_id = cur.lastrowid

        cur = conn.execute("""
            INSERT INTO tickets
                (user_id, title, description, status, priority, category, created_at, updated_at)
            VALUES (?, ?, ?, 'open', ?, ?, ?, ?)
        """, (user_id, title, description, priority, category, now, now))
        ticket_id = cur.lastrowid

        conn.execute("""
            INSERT INTO ticket_events (ticket_id, actor, event_type, content, created_at)
            VALUES (?, 'user', 'message', ?, ?)
        """, (ticket_id, description, now))

    return json.dumps(
        {"ticket_id": ticket_id, "user_id": user_id, "status": "created"},
        ensure_ascii=False,
    )


@mcp.tool()
def add_agent_response(ticket_id: int, response: str, new_status: str = "") -> str:
    """
    Добавить ответ агента в историю тикета и опционально обновить статус.

    Args:
        ticket_id: ID тикета
        response: Текст ответа агента
        new_status: Новый статус (опционально): open|in_progress|waiting|resolved|closed

    Returns:
        JSON: {ticket_id, events_added, new_status}
    """
    now = _now()

    with _conn() as conn:
        ticket_row = conn.execute(
            "SELECT status FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        if not ticket_row:
            return json.dumps({"error": f"Тикет #{ticket_id} не найден"}, ensure_ascii=False)

        current_status = ticket_row["status"]
        events_added = 0

        # Всегда добавляем комментарий агента
        conn.execute("""
            INSERT INTO ticket_events (ticket_id, actor, event_type, content, created_at)
            VALUES (?, 'agent', 'comment', ?, ?)
        """, (ticket_id, response, now))
        events_added += 1

        applied_status = current_status
        if new_status and new_status in _VALID_STATUSES:
            allowed = _VALID_TRANSITIONS.get(current_status, set())
            if new_status in allowed:
                resolved_at = now if new_status == "resolved" else None
                conn.execute("""
                    UPDATE tickets
                    SET status = ?, updated_at = ?,
                        resolved_at = COALESCE(?, resolved_at)
                    WHERE id = ?
                """, (new_status, now, resolved_at, ticket_id))
                conn.execute("""
                    INSERT INTO ticket_events (ticket_id, actor, event_type, content, created_at)
                    VALUES (?, 'system', 'status_change', ?, ?)
                """, (ticket_id, f"{current_status} → {new_status}", now))
                events_added += 1
                applied_status = new_status
            else:
                conn.execute(
                    "UPDATE tickets SET updated_at = ? WHERE id = ?", (now, ticket_id)
                )
        else:
            conn.execute(
                "UPDATE tickets SET updated_at = ? WHERE id = ?", (now, ticket_id)
            )

    return json.dumps(
        {"ticket_id": ticket_id, "events_added": events_added, "new_status": applied_status},
        ensure_ascii=False,
    )


@mcp.tool()
def find_user(email: str = "", name: str = "") -> str:
    """
    Найти пользователя по email (точно) или имени (LIKE).

    Args:
        email: Email для поиска (опционально)
        name: Имя для поиска по подстроке (опционально)

    Returns:
        JSON: {found: true, user: {...}, ticket_stats: {status: count}} или {found: false}
    """
    if not email and not name:
        return json.dumps({"found": False, "error": "Укажите email или name"}, ensure_ascii=False)

    with _conn() as conn:
        if email:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM users WHERE name LIKE ?", (f"%{name}%",)
            ).fetchone()

        if not row:
            return json.dumps({"found": False}, ensure_ascii=False)

        user = dict(row)
        stats_rows = conn.execute("""
            SELECT status, COUNT(*) AS cnt
            FROM tickets WHERE user_id = ?
            GROUP BY status
        """, (user["id"],)).fetchall()

    return json.dumps({
        "found": True,
        "user": user,
        "ticket_stats": {r["status"]: r["cnt"] for r in stats_rows},
    }, ensure_ascii=False)


@mcp.tool()
def get_ticket_stats() -> str:
    """
    Статистика CRM для дашборда агента.

    Returns:
        JSON: {by_status, by_priority, by_category, avg_resolution_hours, oldest_open_ticket}
    """
    with _conn() as conn:
        by_status = {
            r["status"]: r["cnt"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM tickets GROUP BY status"
            ).fetchall()
        }
        by_priority = {
            r["priority"]: r["cnt"]
            for r in conn.execute(
                "SELECT priority, COUNT(*) AS cnt FROM tickets GROUP BY priority"
            ).fetchall()
        }
        by_category = {
            (r["category"] or "other"): r["cnt"]
            for r in conn.execute(
                "SELECT category, COUNT(*) AS cnt FROM tickets GROUP BY category"
            ).fetchall()
        }

        resolved = conn.execute("""
            SELECT (julianday(resolved_at) - julianday(created_at)) * 24 AS hours
            FROM tickets
            WHERE resolved_at IS NOT NULL AND status IN ('resolved', 'closed')
        """).fetchall()
        hours_list = [r["hours"] for r in resolved if r["hours"] is not None]
        avg_hours = sum(hours_list) / len(hours_list) if hours_list else 0.0

        oldest = conn.execute("""
            SELECT id, title,
                   CAST((julianday('now') - julianday(created_at)) AS INTEGER) AS days_open
            FROM tickets
            WHERE status IN ('open', 'in_progress', 'waiting')
            ORDER BY created_at ASC
            LIMIT 1
        """).fetchone()

    return json.dumps({
        "by_status":            by_status,
        "by_priority":          by_priority,
        "by_category":          by_category,
        "avg_resolution_hours": round(avg_hours, 2),
        "oldest_open_ticket":   dict(oldest) if oldest else None,
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
