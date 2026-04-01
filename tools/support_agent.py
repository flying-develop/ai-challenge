"""
Агент обработки тикетов поддержки documaker.

Режимы:
  python tools/support_agent.py              # интерактивный CLI
  python tools/support_agent.py --init       # инициализация БД с начальными данными
  python tools/support_agent.py --auto       # обработать все open-тикеты

Команды в CLI:
  /tickets                — открытые тикеты (таблица с приоритетами)
  /ticket 42              — полная информация по тикету #42
  /stats                  — статистика CRM
  /simulate               — создать 1 тикет через симулятор
  /simulate 5             — создать 5 тикетов
  /simulate auth          — создать тикет категории auth
  /process                — обработать следующий приоритетный тикет
  /process 42             — обработать конкретный тикет
  /quit                   — выход
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
_TOOLS = Path(__file__).parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TOOLS))

from llm_agent.domain.models import ChatMessage
from llm_agent.infrastructure.llm_factory import build_client
from mcp_client.client import MCPClient
from mcp_client.config import MCPServerConfig

# ── RAG (опционально) ─────────────────────────────────────────────────────────
try:
    from rag_indexer.src.storage.index_store import IndexStore
    from rag_indexer.src.embedding.provider import EmbeddingProvider
    from rag_indexer.src.retrieval.retriever import (
        VectorRetriever, BM25Retriever, HybridRetriever,
    )
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

# ── Пути ─────────────────────────────────────────────────────────────────────
DB_PATH      = _ROOT / "crm.db"
DOCS_DB_PATH = _ROOT / "project_docs.db"
PERSONA_PATH = _ROOT / "config" / "support-agent-persona.md"

# ── ANSI-цвета ────────────────────────────────────────────────────────────────
_R     = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RED   = "\033[91m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"

_PRIO_COLOR = {
    "critical": "\033[91m",
    "high":     "\033[93m",
    "medium":   "\033[96m",
    "low":      "\033[37m",
}


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_R}"


# ── CRM-клиент ────────────────────────────────────────────────────────────────

def _make_crm_client() -> MCPClient:
    return MCPClient(MCPServerConfig(
        name="crm_server",
        transport="stdio",
        description="CRM для системы поддержки",
        command=sys.executable,
        args=[str(_ROOT / "mcp_server" / "crm_server.py")],
    ))


def _crm(tool: str, args: dict):
    """Вызвать инструмент CRM MCP и вернуть десериализованный результат."""
    raw = _make_crm_client().call_tool(tool, args)
    return json.loads(raw)


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _load_persona() -> str:
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text(encoding="utf-8")
    return "Ты — опытный специалист поддержки. Отвечай вежливо и конкретно."


def _setup_rag():
    """Инициализировать hybrid RAG retriever. Возвращает retriever или None."""
    if not _RAG_AVAILABLE:
        return None
    if not DOCS_DB_PATH.exists():
        print(f"[agent] Предупреждение: {DOCS_DB_PATH.name} не найден — RAG недоступен.")
        return None
    try:
        import os
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY", "")
        store   = IndexStore(str(DOCS_DB_PATH))
        embedder = EmbeddingProvider.create("qwen", api_key=api_key)
        vector  = VectorRetriever(store, embedder)
        bm25    = BM25Retriever(store)
        return HybridRetriever(vector, bm25)
    except Exception as exc:
        print(f"[agent] Предупреждение: RAG не инициализирован: {exc}")
        return None


# ── Таблица тикетов ───────────────────────────────────────────────────────────

def _print_tickets_table(tickets: list[dict]) -> None:
    if not tickets:
        print("  Нет открытых тикетов.")
        return

    W_ID    = max(max(len(str(t["id"])) for t in tickets), 2)
    W_TITLE = max(min(max(len(t["title"]) for t in tickets), 36), 34)
    W_STAT  = 12
    W_PRIO  = 8
    W_USER  = max(min(max(len(t["user_name"]) for t in tickets), 18), 14)

    def _row(c1, c2, c3, c4, c5, left="│", mid="│", right="│"):
        return (
            f"{left} {c1:>{W_ID}} {mid} {c2:<{W_TITLE}} {mid} "
            f"{c3:<{W_STAT}} {mid} {c4:<{W_PRIO}} {mid} {c5:<{W_USER}} {right}"
        )

    def _sep(l, m, r, h="─"):
        return (
            f"{l}{h * (W_ID + 2)}{m}{h * (W_TITLE + 2)}{m}"
            f"{h * (W_STAT + 2)}{m}{h * (W_PRIO + 2)}{m}{h * (W_USER + 2)}{r}"
        )

    print(_sep("┌", "┬", "┐"))
    print(_row("ID", "Заголовок", "Статус", "Приоритет", "Пользователь"))
    print(_sep("├", "┼", "┤"))
    for t in tickets:
        pc    = _PRIO_COLOR.get(t["priority"], "")
        prio  = f"{pc}{t['priority'][:W_PRIO]:<{W_PRIO}}{_R}"
        print(_row(
            str(t["id"]),
            t["title"][:W_TITLE],
            t["status"][:W_STAT],
            prio,
            t["user_name"][:W_USER],
        ))
    print(_sep("└", "┴", "┘"))


# ── Парсинг ответа агента ──────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    thoughts:   str
    response:   str
    new_status: str


def _parse_response(raw: str) -> AgentResponse:
    def _extract(tag: str) -> str:
        m = re.search(
            rf"\[{tag}\]\s*(.*?)(?=\[(?:THOUGHTS|RESPONSE|STATUS)\]|$)",
            raw, re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    thoughts   = _extract("THOUGHTS") or "(нет внутреннего монолога)"
    response   = _extract("RESPONSE")
    status_raw = _extract("STATUS").lower().strip()

    valid = {"open", "in_progress", "waiting", "resolved", "closed"}
    new_status = status_raw if status_raw in valid else "in_progress"

    if not response:
        # Парсинг не сработал — весь текст как ответ
        response = raw.strip()

    return AgentResponse(thoughts=thoughts, response=response, new_status=new_status)


# ── Обработка тикета ──────────────────────────────────────────────────────────

def process_ticket(ticket_id: int, retriever=None) -> None:
    """RAG-поиск → LLM-ответ → сохранить в CRM."""
    print(f"\n[agent] Загружаю тикет #{ticket_id}...")

    details = _crm("get_ticket_details", {"ticket_id": ticket_id})
    if "error" in details:
        print(f"[agent] Ошибка: {details['error']}")
        return

    ticket = details["ticket"]
    user   = details["user"]
    events = details["events"]

    print(f"[agent] Пользователь: {user['name']} ({user['email']}), тариф: {user['plan']}")
    print(f"[agent] Категория: {ticket.get('category', '?')} | Приоритет: {ticket.get('priority', '?')}")

    # RAG
    query = f"{ticket['title']} {ticket['description']} {ticket.get('category', '')}"
    rag_text = ""
    if retriever:
        label = query[:70] + "..." if len(query) > 70 else query
        print(f"[agent] RAG-поиск: \"{label}\"")
        try:
            results = retriever.search(query, top_k=5)
            if results:
                print(f"[agent] Найдено {len(results)} чанков из docs/documaker/")
                rag_text = "\n\n".join(
                    f"[{i}] {r.doc_title} / {r.section}\n{r.text}"
                    for i, r in enumerate(results, 1)
                )
            else:
                print("[agent] RAG: релевантных чанков не найдено")
        except Exception as exc:
            print(f"[agent] RAG-ошибка: {exc}")
    else:
        print("[agent] RAG недоступен")

    # История
    history = "\n".join(
        f"[{ev['created_at']}] "
        f"{'Пользователь' if ev['actor'] == 'user' else 'Агент' if ev['actor'] == 'agent' else 'Система'}: "
        f"{ev['content']}"
        for ev in events
    )

    persona = _load_persona()
    prompt = f"""{persona}

КОНТЕКСТ ТИКЕТА:
Пользователь: {user['name']} ({user['email']}), тариф: {user['plan']}
Тикет #{ticket['id']}: {ticket['title']}
Категория: {ticket.get('category', 'other')}, Приоритет: {ticket.get('priority', 'medium')}
Создан: {ticket['created_at']}

История тикета:
{history}

ДОКУМЕНТАЦИЯ ПРОДУКТА (из RAG):
{rag_text if rag_text else "Документация недоступна."}

ЗАДАЧА:
Сначала напиши свой внутренний монолог (честные мысли о тикете).
Оформи его в блоке [THOUGHTS].

Потом напиши ответ пользователю в блоке [RESPONSE].
Строго следуй стилю и формату из раздела "Стиль ответа" и "Формат ответа" персоны выше.

В конце укажи новый статус в блоке [STATUS]:
open | in_progress | waiting | resolved

Формат строго:
[THOUGHTS]
...твои мысли...

[RESPONSE]
...ответ пользователю...

[STATUS]
in_progress"""

    llm = build_client("qwen")
    raw = llm.generate([ChatMessage(role="user", content=prompt)]).text
    parsed = _parse_response(raw)

    # Вывод монолога
    print(f"\n{_BOLD}━━━ ВНУТРЕННИЙ МОНОЛОГ АГЕНТА ━━━━━━━━━━━━━━━━━{_R}")
    for line in parsed.thoughts.splitlines():
        print(f"  {_DIM}{line}{_R}")
    print(f"{_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{_R}")

    # Вывод ответа
    print(f"\n{_BOLD}━━━ ОТВЕТ ПОЛЬЗОВАТЕЛЮ ━━━━━━━━━━━━━━━━━━━━━━━{_R}")
    for line in parsed.response.splitlines():
        print(f"  {line}")
    print(f"{_BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{_R}")

    # Сохраняем в CRM
    result = _crm("add_agent_response", {
        "ticket_id":  ticket_id,
        "response":   parsed.response,
        "new_status": parsed.new_status,
    })

    prev_status    = ticket["status"]
    applied_status = result.get("new_status", prev_status)

    print(f"\n[agent] ✓ Ответ добавлен в тикет #{ticket_id}")
    if applied_status != prev_status:
        print(f"[agent] ✓ Статус обновлён: {prev_status} → {_c(applied_status, _GREEN)}")
    else:
        print(f"[agent]   Статус без изменений: {prev_status}")


# ── Инициализация БД ──────────────────────────────────────────────────────────

def init_database() -> None:
    """Создать crm.db с таблицами и начальными данными."""
    print("Инициализация CRM...")

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"  Удалена старая БД: {DB_PATH.name}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

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

    t0 = "2026-03-30 09:00:00"
    t1 = "2026-03-30 11:30:00"
    t2 = "2026-03-31 14:00:00"

    # Пользователи
    for name, email, plan in [
        ("Алексей Петров",  "alex@company.ru",  "pro"),
        ("Мария Сидорова",  "maria@startup.io", "free"),
        ("Иван Козлов",     "ivan@firm.com",    "enterprise"),
        ("Ольга Новикова",  "olga@design.ru",   "free"),
        ("Дмитрий Фёдоров", "dima@agency.ru",   "pro"),
    ]:
        conn.execute(
            "INSERT INTO users (name, email, plan, created_at) VALUES (?, ?, ?, ?)",
            (name, email, plan, t0),
        )

    _INS_T = (
        "INSERT INTO tickets "
        "(user_id, title, description, status, priority, category, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    _INS_TR = (
        "INSERT INTO tickets "
        "(user_id, title, description, status, priority, category, created_at, updated_at, resolved_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    _INS_EV = (
        "INSERT INTO ticket_events (ticket_id, actor, event_type, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)"
    )

    # Тикет 1: resolved (Алексей Петров, user_id=1)
    desc1 = ("Пытаюсь экспортировать документ в PDF, но получаю просто текст — "
             "таблицы и картинки пропали. Форматирование куда-то исчезло.")
    reply1 = ("Алексей, понял — форматирование теряется при экспорте.\n\n"
              "1. Откройте Файл → Экспорт → PDF\n"
              "2. Убедитесь, что выбран режим «Полный PDF» (не «Только текст»)\n"
              "3. Включите опцию «Встроить шрифты»\n"
              "4. Попробуйте экспортировать снова\n\n"
              "Если проблема сохранится — пришлите скриншот настроек.")
    conn.execute(_INS_TR, (1, "PDF экспортируется без форматирования", desc1,
                           "resolved", "medium", "export", t0, t1, t1))
    for row in [
        (1, "user",   "message",       desc1,                                    t0),
        (1, "agent",  "comment",       reply1,                                   t1),
        (1, "system", "status_change", "open → in_progress",                     t1),
        (1, "user",   "message",       "Спасибо, помогло! Не видел этой настройки.", t1),
        (1, "system", "status_change", "in_progress → resolved",                 t1),
    ]:
        conn.execute(_INS_EV, row)

    # Тикет 2: waiting (Мария Сидорова, user_id=2)
    desc2 = ("Нажала «забыл пароль», ввела email, жду уже час — письма нет. "
             "Проверила спам, тоже нет.")
    reply2 = ("Мария, понял — письмо для сброса не доходит.\n\n"
              "Уточните:\n"
              "1. Какой email используете для входа?\n"
              "2. Пробовали подождать 5–10 минут? (иногда бывает задержка)\n"
              "3. Проверили папки «Промоакции» и «Обновления» в Gmail?\n\n"
              "Ждём вашего ответа.")
    conn.execute(_INS_T, (2, "не приходит письмо для сброса пароля", desc2,
                          "waiting", "medium", "auth", t1, t2))
    for row in [
        (2, "user",   "message",       desc2,                    t1),
        (2, "agent",  "comment",       reply2,                   t2),
        (2, "system", "status_change", "in_progress → waiting",  t2),
    ]:
        conn.execute(_INS_EV, row)

    # Тикет 3: open, critical (Иван Козлов, user_id=3)
    desc3 = ("Открываю documaker — всё просто висит. Кнопки не нажимаются, "
             "документ не открывается. Срочно, дедлайн через час!!!")
    conn.execute(_INS_T, (3, "НИЧЕГО НЕ РАБОТАЕТ помогите!!!!", desc3,
                          "open", "critical", "ui", t2, t2))
    conn.execute(_INS_EV, (3, "user", "message", desc3, t2))

    conn.commit()
    conn.close()

    print("  Создано: 5 пользователей")
    print("  Создано: 3 тикета (1 resolved, 1 waiting, 1 open/critical)")
    print(f"  БД: {DB_PATH}")
    print("Инициализация завершена!")


# ── Статистика ────────────────────────────────────────────────────────────────

def _print_stats(stats: dict) -> None:
    by_status   = stats.get("by_status", {})
    by_priority = stats.get("by_priority", {})
    by_category = stats.get("by_category", {})

    total = sum(by_status.values())
    print(f"\nСтатистика CRM:")
    print(f"  Всего тикетов: {total}")
    print(f"  По статусу:   {' | '.join(f'{k}: {v}' for k, v in by_status.items())}")
    print(f"  По приоритету: {' | '.join(f'{k}: {v}' for k, v in by_priority.items())}")
    print(f"  По категориям: {', '.join(f'{k}×{v}' for k, v in by_category.items())}")
    print(f"  Среднее время решения: {stats.get('avg_resolution_hours', 0):.1f} ч.")

    oldest = stats.get("oldest_open_ticket")
    if oldest:
        print(f"  Старейший открытый: #{oldest['id']} ({oldest['days_open']} дн.) — {oldest['title']}")


# ── Интерактивный CLI ─────────────────────────────────────────────────────────

def run_cli(retriever=None) -> None:
    stats    = _crm("get_ticket_stats", {})
    by_stat  = stats.get("by_status", {})
    total_q  = by_stat.get("open", 0) + by_stat.get("in_progress", 0) + by_stat.get("waiting", 0)
    critical = stats.get("by_priority", {}).get("critical", 0)

    crit_str = f" ({_c(str(critical) + ' critical', _RED)})" if critical else ""
    print(f"\n{_BOLD}╔══════════════════════════════════════════════╗{_R}")
    print(f"{_BOLD}║{_R}     Documaker Support Agent v1.0             {_BOLD}║{_R}")
    print(f"{_BOLD}║{_R}     Тикетов в очереди: {total_q}{crit_str:<30}{_BOLD}║{_R}")
    print(f"{_BOLD}╚══════════════════════════════════════════════╝{_R}")
    print("Команды: /tickets /ticket N /process [N] /simulate [N|кат] /stats /quit\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not line:
            continue

        parts = line.split()
        cmd   = parts[0].lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("Пока!")
            break

        elif cmd == "/tickets":
            data = _crm("get_open_tickets", {"limit": 20})
            _print_tickets_table(data if isinstance(data, list) else [])

        elif cmd == "/ticket":
            if len(parts) < 2 or not parts[1].isdigit():
                print("Используйте: /ticket <id>")
                continue
            tid  = int(parts[1])
            data = _crm("get_ticket_details", {"ticket_id": tid})
            if "error" in data:
                print(f"Ошибка: {data['error']}")
                continue
            t, u = data["ticket"], data["user"]
            print(f"\nТикет #{t['id']}: {_BOLD}{t['title']}{_R}")
            print(f"  Статус: {t['status']} | Приоритет: {t['priority']} | Категория: {t.get('category','?')}")
            print(f"  Пользователь: {u['name']} ({u['email']}), тариф: {u['plan']}")
            print(f"  Создан: {t['created_at']}\n  История:")
            for ev in data["events"]:
                a = {"user": "Пользователь", "agent": "Агент", "system": "Система"}.get(ev["actor"], ev["actor"])
                print(f"  [{ev['created_at']}] {_BOLD}{a}{_R}: {ev['content'][:200]}")

        elif cmd == "/stats":
            _print_stats(_crm("get_ticket_stats", {}))

        elif cmd == "/simulate":
            import user_simulator as _sim
            arg = parts[1] if len(parts) > 1 else ""
            if arg.isdigit():
                for i in range(int(arg)):
                    print(f"\n--- Тикет {i + 1}/{int(arg)} ---")
                    _sim.simulate_ticket()
            elif arg in _sim.TICKET_CATEGORIES:
                _sim.simulate_ticket(arg)
            else:
                _sim.simulate_ticket()

        elif cmd == "/process":
            if len(parts) > 1 and parts[1].isdigit():
                process_ticket(int(parts[1]), retriever)
            else:
                data = _crm("get_open_tickets", {"limit": 1})
                if not data:
                    print("Нет открытых тикетов!")
                else:
                    process_ticket(data[0]["id"], retriever)

        else:
            print("Неизвестная команда. /tickets /ticket N /process [N] /simulate [N|кат] /stats /quit")


# ── Автоматический режим ──────────────────────────────────────────────────────

def run_auto(retriever=None) -> None:
    print("[auto] Загружаю открытые тикеты...")
    data = _crm("get_open_tickets", {"limit": 50})
    open_tickets = [t for t in (data if isinstance(data, list) else []) if t.get("status") == "open"]

    if not open_tickets:
        print("[auto] Нет открытых тикетов.")
        return

    print(f"[auto] Найдено {len(open_tickets)} открытых тикетов.")
    for t in open_tickets:
        print(f"\n{'=' * 50}")
        process_ticket(t["id"], retriever)


# ── Точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Documaker Support Agent")
    parser.add_argument("--init", action="store_true",
                        help="Инициализировать crm.db с начальными данными")
    parser.add_argument("--auto", action="store_true",
                        help="Автоматически обработать все open-тикеты")
    args = parser.parse_args()

    if args.init:
        init_database()
        return

    retriever = _setup_rag()

    if args.auto:
        run_auto(retriever)
    else:
        run_cli(retriever)


if __name__ == "__main__":
    main()
