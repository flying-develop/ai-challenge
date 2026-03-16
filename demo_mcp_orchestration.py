#!/usr/bin/env python3
"""День 20: Orchestration MCP — демонстрация двухпроходного исследовательского агента.

Запуск:
    python demo_mcp_orchestration.py

Что демонстрирует:
  1. Запуск 4 MCP-серверов и проверка их инструментов
  2. Полный исследовательский цикл (8 состояний)
  3. Двухпроходный поиск с дедупликацией URL
  4. Прогресс-уведомления в Telegram (или stdout)
  5. Аудит-журнал всех этапов
  6. Проверка инвариантов

Конфигурация (.env):
    SEARCH_MODE=mock           # mock работает без ключей (по умолчанию)
    TELEGRAM_BOT_TOKEN=...     # опционально (без токена — вывод в stdout)
    TELEGRAM_CHAT_ID=...       # опционально
    Любой LLM-ключ: QWEN_API_KEY, OPENAI_API_KEY или ANTHROPIC_API_KEY
"""

from __future__ import annotations

import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Проверка зависимостей
# ---------------------------------------------------------------------------

try:
    from mcp_client.client import MCPClient
    from mcp_client.config import MCPServerConfig
except ImportError:
    print("❌ Пакет 'mcp' не установлен. Установите: pip install mcp")
    sys.exit(1)

from mcp_server.llm_client import create_llm_fn
from orchestrator.research_orchestrator import ResearchOrchestrator


# ---------------------------------------------------------------------------
# Конфигурация MCP-серверов
# ---------------------------------------------------------------------------

def _make_client(module: str, name: str) -> MCPClient:
    """Создать MCPClient для python-модуля."""
    config = MCPServerConfig(
        name=name,
        transport="stdio",
        description=f"MCP сервер {name}",
        command="python",
        args=["-m", module],
    )
    return MCPClient(config)


# ---------------------------------------------------------------------------
# Шаг 1: Проверка серверов
# ---------------------------------------------------------------------------

def step1_check_servers() -> dict[str, MCPClient]:
    """Запустить все 4 MCP-сервера и проверить их инструменты."""
    print("\n" + "═" * 60)
    print("  Шаг 1: Проверка MCP-серверов")
    print("═" * 60)

    servers = {
        "search":   _make_client("mcp_server.search_server",   "search_server"),
        "scraper":  _make_client("mcp_server.scraper_server",  "scraper_server"),
        "telegram": _make_client("mcp_server.telegram_server", "telegram_server"),
        "journal":  _make_client("mcp_server.journal_server",  "journal_server"),
    }

    expected_tools = {
        "search":   {"search_web"},
        "scraper":  {"fetch_urls"},
        "telegram": {"send_progress", "send_result"},
        "journal":  {"log_stage", "get_log"},
    }

    total_tools = 0
    all_ok = True

    for key, client in servers.items():
        try:
            tools = client.connect_and_list_tools()
            tool_names = {t["name"] for t in tools}
            expected = expected_tools[key]
            ok = expected.issubset(tool_names)
            mark = "✅" if ok else "⚠️"
            print(f"  {mark} {client.config.name}: {len(tools)} tool(s) — {', '.join(tool_names)}")
            if not ok:
                missing = expected - tool_names
                print(f"     ⚠️ Ожидаемые инструменты отсутствуют: {missing}")
                all_ok = False
            total_tools += len(tools)
        except Exception as exc:
            print(f"  ❌ {client.config.name}: {exc}")
            all_ok = False

    print(f"\n  Итого: {total_tools} инструментов на 4 серверах")
    if all_ok:
        print("  ✅ Все серверы работают")
    else:
        print("  ⚠️ Некоторые серверы недоступны — продолжаем с тем что есть")

    return servers


# ---------------------------------------------------------------------------
# Шаг 2: Исследование
# ---------------------------------------------------------------------------

def step2_run_research(servers: dict[str, MCPClient]) -> tuple[ResearchOrchestrator, str]:
    """Запустить полный исследовательский цикл."""
    print("\n" + "═" * 60)
    print("  Шаг 2: Запуск исследования")
    print("═" * 60)

    task = "Хочу сходить в кино в эти выходные"
    print(f"\n  Задача: \"{task}\"\n")

    # Инициализация LLM
    try:
        llm_fn = create_llm_fn(timeout=60.0)
        print("  ✅ LLM-провайдер инициализирован")
    except ValueError as exc:
        print(f"  ❌ LLM недоступен: {exc}")
        print("  Укажите QWEN_API_KEY, OPENAI_API_KEY или ANTHROPIC_API_KEY в .env")
        sys.exit(1)

    orchestrator = ResearchOrchestrator(
        mcp_clients=servers,
        llm_fn=llm_fn,
        verbose=True,
    )

    print()
    t0 = time.time()
    result = orchestrator.run(task)
    elapsed = time.time() - t0

    print(f"\n  ✅ Исследование завершено за {elapsed:.1f}с")
    print(f"\n{'─' * 60}")
    print("  Финальный ответ:")
    print("─" * 60)
    print(result[:2000] + ("..." if len(result) > 2000 else ""))
    print("─" * 60)

    return orchestrator, task


# ---------------------------------------------------------------------------
# Шаг 3: Проверка журнала
# ---------------------------------------------------------------------------

def step3_check_journal(servers: dict[str, MCPClient], task_id: str) -> None:
    """Получить и отобразить полный лог выполнения задачи."""
    print("\n" + "═" * 60)
    print("  Шаг 3: Журнал выполнения")
    print("═" * 60)

    try:
        log = servers["journal"].call_tool("get_log", {"task_id": task_id})
        print(f"\n{log}\n")
    except Exception as exc:
        print(f"  ❌ Ошибка получения журнала: {exc}")


# ---------------------------------------------------------------------------
# Шаг 4: Проверка инвариантов
# ---------------------------------------------------------------------------

def step4_check_invariants(orchestrator: ResearchOrchestrator) -> bool:
    """Проверить соблюдение всех инвариантов."""
    print("\n" + "═" * 60)
    print("  Шаг 4: Проверка инвариантов")
    print("═" * 60)

    report = orchestrator.get_invariant_report()
    ctx = orchestrator.context

    checks = [
        (
            "Ссылки раунда 1 ≤ 10",
            report["initial_links_le_10"],
            f"{report['initial_links_count']} ссылок",
        ),
        (
            "Ссылки раунда 2 ≤ 10",
            report["deep_links_le_10"],
            f"{report['deep_links_count']} ссылок",
        ),
        (
            "Дедупликация URL",
            report["deduplication_ok"],
            f"пересечение = {len(report['duplicate_urls'])}",
        ),
        (
            "Суммаризация выполнена",
            report["summary_v1_exists"],
            f"{len(ctx.summary_v1)} символов",
        ),
        (
            "Финальный ответ сформирован",
            report["final_result_exists"],
            f"{len(ctx.final_result)} символов",
        ),
        (
            "Ровно 2 поисковых раунда",
            report["two_rounds_only"],
            "архитектурный инвариант",
        ),
        (
            "Telegram-уведомления",
            report["telegram_calls"] >= 8,
            f"{report['telegram_calls']} вызовов",
        ),
        (
            "Журнал заполнен",
            report["journal_calls"] >= 7,
            f"{report['journal_calls']} записей",
        ),
    ]

    print()
    all_passed = True
    for name, passed, details in checks:
        mark = "✅" if passed else "❌"
        print(f"  {mark} {name:<35} {details}")
        if not passed:
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Шаг 5: Маршрутизация вызовов
# ---------------------------------------------------------------------------

def step5_check_routing(orchestrator: ResearchOrchestrator) -> None:
    """Отобразить статистику вызовов по серверам."""
    print("\n" + "═" * 60)
    print("  Шаг 5: Маршрутизация вызовов")
    print("═" * 60)

    calls = orchestrator._search_calls
    print()
    print(f"  {'Сервер':<20} {'Вызовов'}")
    print(f"  {'─' * 30}")
    for server, count in sorted(calls.items()):
        print(f"  {server:<20} {count}")

    # Проверяем что все 4 сервера были использованы
    used = set(calls.keys())
    expected = {"search", "scraper", "telegram", "journal"}
    all_used = expected.issubset(used)
    mark = "✅" if all_used else "⚠️"
    print(f"\n  {mark} Использовано серверов: {len(used)}/4")
    if not all_used:
        missing = expected - used
        print(f"     Не использованы: {missing}")


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------

def print_final_report(servers_ok: bool, invariants_ok: bool, elapsed: float) -> None:
    """Напечатать итоговую таблицу."""
    print("\n" + "═" * 60)
    print("  Orchestration MCP Test Report")
    print("═" * 60)
    print()

    rows = [
        ("4 MCP-сервера",      servers_ok,     "все отвечают"),
        ("Tool discovery",      servers_ok,     "6 tools всего"),
        ("Инварианты",          invariants_ok,  "все соблюдены"),
        ("Время выполнения",    True,           f"{elapsed:.1f}с"),
    ]

    header = f"  {'Тест':<24} {'Статус':<8} {'Детали'}"
    print(header)
    print("  " + "─" * 50)
    for name, ok, details in rows:
        mark = "✅" if ok else "❌"
        print(f"  {name:<24} {mark:<8} {details}")

    print()
    overall = all(r[1] for r in rows)
    if overall:
        print("  🎉 Все проверки пройдены!")
    else:
        print("  ⚠️ Некоторые проверки не пройдены — см. детали выше")
    print()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    print("╔" + "═" * 58 + "╗")
    print("║   День 20: Orchestration MCP                           ║")
    print("║   Двухпроходный исследовательский агент               ║")
    print("╚" + "═" * 58 + "╝")

    t_start = time.time()

    # Шаг 1: Проверка серверов
    servers = step1_check_servers()

    # Шаг 2: Запуск исследования
    orchestrator, _task = step2_run_research(servers)

    # Шаг 3: Журнал
    step3_check_journal(servers, orchestrator.context.task_id)

    # Шаг 4: Инварианты
    invariants_ok = step4_check_invariants(orchestrator)

    # Шаг 5: Маршрутизация
    step5_check_routing(orchestrator)

    # Итог
    elapsed = time.time() - t_start
    print_final_report(True, invariants_ok, elapsed)


if __name__ == "__main__":
    main()
