"""
Telegram-бот поддержки documaker.

Входящее сообщение → CRM тикет + RAG из docs/documaker/ → ответ в Telegram.
Вместо симулятора — реальные пользователи через Telegram.

Запуск:
  python tools/support_bot.py
  python tools/support_bot.py --init    # пересоздать crm.db и выйти

Переменные окружения:
  TELEGRAM_BOT_TOKEN   — обязателен
  QWEN_API_KEY         — обязателен
  DASHSCOPE_API_KEY    — для RAG-эмбеддингов (= QWEN_API_KEY если не задан)

Команды пользователя:
  /start  — приветствие
  /new    — новый тикет (текущий закрывается)
  /status — статус текущего тикета
  /help   — список команд
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from llm_agent.domain.models import ChatMessage
from llm_agent.infrastructure.llm_factory import build_client
from mcp_client.client import MCPClient
from mcp_client.config import MCPServerConfig

try:
    from rag_indexer.src.storage.index_store import IndexStore
    from rag_indexer.src.embedding.provider import EmbeddingProvider
    from rag_indexer.src.retrieval.retriever import (
        VectorRetriever, BM25Retriever, HybridRetriever,
    )
    _RAG_OK = True
except ImportError:
    _RAG_OK = False

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

DOCS_DB   = _ROOT / "project_docs.db"
TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
_TG_LIMIT = 4096

_WELCOME = """\
Привет! Я бот поддержки documaker.

Опиши свою проблему — найду ответ в документации и создам тикет.

/new    — начать новый тикет
/status — статус текущего тикета
/help   — список команд\
"""

_HELP = """\
Команды:

/start  — приветствие
/new    — закрыть текущий тикет и начать новый
/status — статус текущего тикета в CRM
/help   — этот список

Просто напиши вопрос — я отвечу по документации documaker.\
"""

PERSONA_PATH = _ROOT / "config" / "support-agent-persona.md"

# ---------------------------------------------------------------------------
# CRM
# ---------------------------------------------------------------------------

def _crm(tool: str, args: dict):
    client = MCPClient(MCPServerConfig(
        name="crm_server",
        transport="stdio",
        description="CRM",
        command=sys.executable,
        args=[str(_ROOT / "mcp_server" / "crm_server.py")],
    ))
    return json.loads(client.call_tool(tool, args))


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def _build_retriever():
    if not _RAG_OK or not DOCS_DB.exists():
        print(f"[bot] RAG недоступен (project_docs.db: {DOCS_DB.exists()})")
        return None
    try:
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY", "")
        store   = IndexStore(str(DOCS_DB))
        embedder = EmbeddingProvider.create("qwen", api_key=api_key)
        return HybridRetriever(VectorRetriever(store, embedder), BM25Retriever(store))
    except Exception as exc:
        print(f"[bot] RAG ошибка инициализации: {exc}")
        return None


# ---------------------------------------------------------------------------
# Парсинг ответа LLM
# ---------------------------------------------------------------------------

@dataclass
class _Parsed:
    thoughts: str
    response: str
    status:   str


def _parse(raw: str) -> _Parsed:
    def _get(tag: str) -> str:
        m = re.search(
            rf"\[{tag}\]\s*(.*?)(?=\[(?:THOUGHTS|RESPONSE|STATUS)\]|$)",
            raw, re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    thoughts = _get("THOUGHTS") or "(нет монолога)"
    response = _get("RESPONSE") or raw.strip()
    status_raw = _get("STATUS").lower().strip()
    valid = {"open", "in_progress", "waiting", "resolved", "closed"}
    return _Parsed(thoughts=thoughts, response=response,
                   status=status_raw if status_raw in valid else "in_progress")


# ---------------------------------------------------------------------------
# Сессия одного пользователя
# ---------------------------------------------------------------------------

@dataclass
class Session:
    chat_id:   str
    username:  str                    # Telegram @login или first_name
    ticket_id: int | None  = None
    history:   list[dict]  = field(default_factory=list)  # [{role, content}]

    @property
    def email(self) -> str:
        return f"{self.username}@telegram"

    @property
    def display_name(self) -> str:
        return f"@{self.username}"


# ---------------------------------------------------------------------------
# Обработка сообщения
# ---------------------------------------------------------------------------

def _load_persona() -> str:
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text(encoding="utf-8")
    return "Ты — опытный специалист поддержки documaker. Отвечай вежливо и конкретно."


def handle_message(session: Session, text: str, retriever, llm) -> str:
    """Обработать сообщение: создать/обновить тикет → RAG → ответ."""

    # 1. Создать тикет при первом сообщении
    if session.ticket_id is None:
        result = _crm("create_ticket", {
            "user_email":  session.email,
            "title":       text[:80],
            "description": text,
            "category":    "other",
            "priority":    "medium",
        })
        session.ticket_id = result["ticket_id"]
        # Обновить имя пользователя в CRM (find_user возвращает авто-имя из email)
        print(f"[bot] {session.display_name} → тикет #{session.ticket_id}")

    # 2. RAG
    rag_text = ""
    if retriever:
        try:
            results = retriever.search(text, top_k=5)
            if results:
                rag_text = "\n\n".join(
                    f"[{i}] {r.doc_title} / {r.section}\n{r.text}"
                    for i, r in enumerate(results, 1)
                )
        except Exception as exc:
            print(f"[bot] RAG ошибка: {exc}")

    # 3. История диалога
    session.history.append({"role": "user", "content": text})
    history_str = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Агент'}: {m['content']}"
        for m in session.history[-10:]
    )

    # 4. Промпт
    persona = _load_persona()
    prompt = f"""{persona}

КОНТЕКСТ:
Пользователь: {session.display_name} (тариф: free)
Тикет #{session.ticket_id}

История диалога:
{history_str}

ДОКУМЕНТАЦИЯ (RAG):
{rag_text if rag_text else "Документация недоступна."}

ЗАДАЧА: обработай последнее сообщение пользователя.
В блоке [RESPONSE] строго следуй стилю и формату из раздела "Стиль ответа" и "Формат ответа" выше.

[THOUGHTS]
...твои мысли...

[RESPONSE]
...ответ пользователю в точном соответствии со стилем персоны...

[STATUS]
in_progress"""

    raw = llm.generate([ChatMessage(role="user", content=prompt)]).text
    parsed = _parse(raw)

    # 5. Лог внутреннего монолога (только сервер)
    print(f"[thoughts] {parsed.thoughts[:200]}")

    # 6. Сохранить в CRM
    _crm("add_agent_response", {
        "ticket_id":  session.ticket_id,
        "response":   parsed.response,
        "new_status": parsed.status,
    })

    # 7. Обновить историю
    session.history.append({"role": "assistant", "content": parsed.response})
    if len(session.history) > 20:
        session.history = session.history[-20:]

    return parsed.response


# ---------------------------------------------------------------------------
# Telegram polling
# ---------------------------------------------------------------------------

def _poll(offset: int, timeout: int = 30) -> list[dict]:
    url = (
        f"https://api.telegram.org/bot{TOKEN}/getUpdates"
        f"?offset={offset}&timeout={timeout}"
        f"&allowed_updates=%5B%22message%22%5D"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout + 5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    if not data.get("ok"):
        return []

    updates = []
    for upd in data.get("result", []):
        msg = upd.get("message", {})
        if not msg or "text" not in msg:
            continue
        from_user = msg.get("from", {})
        updates.append({
            "update_id": upd["update_id"],
            "chat_id":   str(msg["chat"]["id"]),
            "text":      msg["text"],
            "username":  (
                from_user.get("username")          # @login (без @)
                or from_user.get("first_name", "")
                or str(from_user.get("id", "user"))
            ),
        })
    return updates


def _send(chat_id: str, text: str) -> None:
    for chunk in _split(text):
        _send_one(chat_id, chunk)


def _send_one(chat_id: str, text: str) -> None:
    url  = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text}).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"[bot] sendMessage error: {exc}")


def _split(text: str, limit: int = _TG_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, buf = [], text
    while buf:
        if len(buf) <= limit:
            parts.append(buf)
            break
        cut = buf.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        parts.append(buf[:cut])
        buf = buf[cut:].lstrip("\n")
    return parts


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run() -> None:
    if not TOKEN:
        print("[bot] TELEGRAM_BOT_TOKEN не задан — выход")
        return

    print("[bot] Инициализация RAG...")
    retriever = _build_retriever()

    print("[bot] Инициализация LLM (Qwen API)...")
    llm = build_client("qwen")

    sessions: dict[str, Session] = {}
    offset = 0

    print("[bot] Запуск long-polling...")

    while True:
        try:
            updates = _poll(offset)
        except KeyboardInterrupt:
            print("\n[bot] Остановка.")
            break
        except Exception as exc:
            print(f"[bot] Ошибка polling: {exc}")
            time.sleep(5)
            continue

        for upd in updates:
            try:
                _dispatch(upd, sessions, retriever, llm)
            except Exception as exc:
                import traceback
                print(f"[bot] Ошибка dispatch: {exc}")
                traceback.print_exc()
            offset = upd["update_id"] + 1


def _dispatch(upd: dict, sessions: dict, retriever, llm) -> None:
    chat_id  = upd["chat_id"]
    text     = upd["text"].strip()
    username = upd["username"]

    # Получить или создать сессию
    if chat_id not in sessions:
        sessions[chat_id] = Session(chat_id=chat_id, username=username)
        print(f"[bot] Новая сессия: {username} ({chat_id})")

    session = sessions[chat_id]

    # Команды
    if text in ("/start", "/help"):
        _send(chat_id, _WELCOME if text == "/start" else _HELP)
        return

    if text == "/new":
        # Закрыть текущий тикет и начать новый
        if session.ticket_id:
            _crm("add_agent_response", {
                "ticket_id":  session.ticket_id,
                "response":   "Пользователь закрыл тикет командой /new.",
                "new_status": "resolved",
            })
            print(f"[bot] Тикет #{session.ticket_id} закрыт (/{username})")
        session.ticket_id = None
        session.history   = []
        _send(chat_id, "Новый тикет будет создан с вашим следующим сообщением.")
        return

    if text == "/status":
        if not session.ticket_id:
            _send(chat_id, "Нет активного тикета. Напишите вопрос — создам.")
            return
        data = _crm("get_ticket_details", {"ticket_id": session.ticket_id})
        t = data.get("ticket", {})
        _send(chat_id,
              f"Тикет #{t.get('id')}\n"
              f"Статус: {t.get('status')}\n"
              f"Приоритет: {t.get('priority')}\n"
              f"Создан: {t.get('created_at')}")
        return

    # Обычное сообщение
    print(f"[bot] {username} ({chat_id}): {text[:80]}")
    _send(chat_id, "_Обрабатываю запрос..._")

    response = handle_message(session, text, retriever, llm)
    _send(chat_id, response)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Documaker Support Bot (Telegram)")
    parser.add_argument("--init", action="store_true",
                        help="Пересоздать crm.db с начальными данными и выйти")
    args = parser.parse_args()

    if args.init:
        from support_agent import init_database
        init_database()
        return

    run()


if __name__ == "__main__":
    main()
