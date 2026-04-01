"""
Генератор реалистичных тикетов поддержки через Qwen API.

Запуск:
  python tools/user_simulator.py --tickets 5
  python tools/user_simulator.py --tickets 1 --category auth
  python tools/user_simulator.py --interactive
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

from llm_agent.domain.models import ChatMessage
from llm_agent.infrastructure.llm_factory import build_client
from mcp_client.client import MCPClient
from mcp_client.config import MCPServerConfig

# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

SEED_USERS = [
    {"name": "Алексей Петров",  "email": "alex@company.ru",  "plan": "pro"},
    {"name": "Мария Сидорова",  "email": "maria@startup.io", "plan": "free"},
    {"name": "Иван Козлов",     "email": "ivan@firm.com",    "plan": "enterprise"},
    {"name": "Ольга Новикова",  "email": "olga@design.ru",   "plan": "free"},
    {"name": "Дмитрий Фёдоров", "email": "dima@agency.ru",   "plan": "pro"},
]

TICKET_CATEGORIES: dict[str, list[str]] = {
    "auth": [
        "не могу войти в аккаунт",
        "забыл пароль, сброс не приходит",
        "вылетает из системы каждые 5 минут",
    ],
    "export": [
        "PDF не скачивается",
        "документ экспортируется пустым",
        "шрифты в PDF не те",
    ],
    "ui": [
        "кнопки не нажимаются",
        "страница не загружается",
        "редактор завис",
    ],
    "performance": [
        "всё очень медленно",
        "браузер падает при открытии",
        "документ на 10 страниц грузится минуту",
    ],
    "billing": [
        "списали деньги дважды",
        "не активируется про-тариф после оплаты",
        "хочу возврат",
    ],
}

_SYSTEM_PROMPT = """\
Ты — пользователь программы documaker (редактор PDF-документов).
У тебя возникла проблема и ты пишешь в поддержку.

Твои особенности:
- Пишешь размыто, не указываешь детали
- Эмоционально реагируешь ("всё сломалось!", "ничего не работает")
- Не знаешь технических терминов
- Иногда описываешь симптом вместо проблемы

Категория проблемы: {category}
Твоё имя: {user_name}
Тариф: {plan}

Напиши обращение в поддержку в 2-4 предложения.
Не пиши "Здравствуйте" и вежливые формальности — сразу к проблеме.
Ответь ТОЛЬКО текстом обращения, без кавычек и пояснений."""


# ---------------------------------------------------------------------------
# CRM-клиент
# ---------------------------------------------------------------------------

def _make_crm_client() -> MCPClient:
    return MCPClient(MCPServerConfig(
        name="crm_server",
        transport="stdio",
        description="CRM для системы поддержки",
        command=sys.executable,
        args=[str(_ROOT / "mcp_server" / "crm_server.py")],
    ))


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def simulate_ticket(category: str | None = None, verbose: bool = True) -> dict:
    """Создать один тикет: LLM генерирует текст, CRM MCP сохраняет.

    Returns:
        dict с ticket_id, user, category, title, description
    """
    user = random.choice(SEED_USERS)
    if not category or category not in TICKET_CATEGORIES:
        category = random.choice(list(TICKET_CATEGORIES.keys()))
    theme = random.choice(TICKET_CATEGORIES[category])

    if verbose:
        print(f"[simulator] Пользователь: {user['name']} ({user['email']}), тариф: {user['plan']}")
        print(f"[simulator] Категория: {category}")
        print("[simulator] Генерирую обращение через Qwen API...")

    llm = build_client("qwen")
    system = _SYSTEM_PROMPT.format(
        category=category,
        user_name=user["name"],
        plan=user["plan"],
    )
    resp = llm.generate([
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=f"Тема: {theme}"),
    ])
    description = resp.text.strip().strip('"').strip("'")

    crm = _make_crm_client()
    result_raw = crm.call_tool("create_ticket", {
        "user_email": user["email"],
        "title":      theme,
        "description": description,
        "category":   category,
        "priority":   "medium",
    })
    result = json.loads(result_raw)

    if verbose:
        print(f"[simulator] Создан тикет #{result.get('ticket_id')}: \"{theme}\"")

    return {
        "ticket_id":   result.get("ticket_id"),
        "user":        user,
        "category":    category,
        "title":       theme,
        "description": description,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Симулятор пользовательских тикетов для documaker")
    parser.add_argument("--tickets", type=int, default=1, metavar="N",
                        help="Количество тикетов для генерации (по умолчанию: 1)")
    parser.add_argument("--category", type=str,
                        choices=list(TICKET_CATEGORIES.keys()),
                        help="Категория тикета (по умолчанию: случайная)")
    parser.add_argument("--interactive", action="store_true",
                        help="Интерактивный режим: генерировать тикеты по запросу")
    args = parser.parse_args()

    if args.interactive:
        cats = " | ".join(TICKET_CATEGORIES.keys())
        print(f"Симулятор тикетов. Категории: {cats}")
        print("Введите категорию (или Enter для случайной), 'quit' для выхода.\n")
        while True:
            try:
                cat = input("Категория: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cat in ("quit", "exit", "q"):
                break
            simulate_ticket(cat if cat in TICKET_CATEGORIES else None)
            print()
        return

    for i in range(args.tickets):
        if args.tickets > 1:
            print(f"\n--- Тикет {i + 1}/{args.tickets} ---")
        simulate_ticket(args.category)


if __name__ == "__main__":
    main()
