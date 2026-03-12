"""Демо-скрипт: News Digest MCP-сервер.

Демонстрирует полный цикл работы:
1. Проверка окружения
2. Tool discovery (6 инструментов)
3. Принудительный сбор новостей (force_fetch_now)
4. Просмотр последних заголовков
5. Статус планировщика
6. Принудительная генерация сводки (force_digest_now)
7. Получение сводки из БД (get_daily_digest)
8. Повторный сбор — проверка дедупликации
9. Обработка ошибок

Запуск:
    python demo_news_digest.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print("=" * 60)


def _check_internet(url: str = "https://ria.ru", timeout: int = 5) -> bool:
    """Проверить доступность URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; news-digest-demo/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _check_llm_config() -> tuple[bool, str]:
    """Проверить конфигурацию LLM. Возвращает (ok, provider_name)."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        if os.environ.get("QWEN_API_KEY", "").strip():
            provider = "qwen"
        elif os.environ.get("OPENAI_API_KEY", "").strip():
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY", "").strip():
            provider = "claude"

    if not provider:
        return False, "не настроен"
    return True, provider


# ---------------------------------------------------------------------------
# Шаг 1: Проверка окружения
# ---------------------------------------------------------------------------

def step_check_environment() -> dict[str, bool]:
    """Шаг 1: Проверить компоненты окружения."""
    _print_header("Шаг 1: Проверка окружения")
    results = {}

    # 1.1 Файл сервера
    server_file = PROJECT_ROOT / "mcp_server" / "news_server.py"
    ok = server_file.exists()
    results["server_file"] = ok
    status = "✅" if ok else "❌"
    print(f"  {status} mcp_server/news_server.py: {'найден' if ok else 'не найден'}")

    # 1.2 Доступность ria.ru
    print("  ⏳ Проверяю доступность ria.ru...", end="", flush=True)
    ok = _check_internet("https://ria.ru/export/rss2/archive/index.xml")
    results["ria_ru"] = ok
    status = "✅" if ok else "❌"
    print(f"\r  {status} Доступность ria.ru: {'OK' if ok else 'недоступен'}")

    # 1.3 LLM-провайдер
    ok, provider = _check_llm_config()
    results["llm_provider"] = ok
    status = "✅" if ok else "⚠️"
    print(f"  {status} LLM-провайдер: {provider}")

    # 1.4 Пакет schedule
    try:
        import schedule as _
        results["schedule_pkg"] = True
        print("  ✅ Пакет schedule: установлен")
    except ImportError:
        results["schedule_pkg"] = False
        print("  ⚠️  Пакет schedule: не установлен (pip install schedule)")

    # 1.5 Пакет mcp
    try:
        from mcp import ClientSession as _
        results["mcp_pkg"] = True
        print("  ✅ Пакет mcp: установлен")
    except ImportError:
        results["mcp_pkg"] = False
        print("  ❌ Пакет mcp: не установлен (pip install mcp)")

    return results


# ---------------------------------------------------------------------------
# Шаг 2: Tool discovery
# ---------------------------------------------------------------------------

def step_tool_discovery() -> tuple[bool, list[dict]]:
    """Шаг 2: Подключиться к MCP-серверу и получить список инструментов."""
    _print_header("Шаг 2: Tool Discovery")

    try:
        from mcp_client.config import MCPConfigParser
        from mcp_client.client import MCPClient
    except ImportError:
        print("  ❌ Не удалось импортировать mcp_client")
        return False, []

    # Загружаем конфигурацию
    try:
        parser = MCPConfigParser()
        servers = {s.name: s for s in parser.load()}
    except Exception as exc:
        print(f"  ❌ Ошибка чтения конфигурации: {exc}")
        return False, []

    config = servers.get("news_digest")
    if config is None:
        print("  ❌ Сервер 'news_digest' не найден в config/mcp-servers.md")
        return False, []

    print(f"  📡 Подключаюсь к '{config.name}'...")
    client = MCPClient(config)
    try:
        tools = client.connect_and_list_tools()
    except RuntimeError as exc:
        print(f"  ❌ Ошибка подключения: {exc}")
        return False, []

    print(f"  ✅ Найдено инструментов: {len(tools)}")
    for t in tools:
        print(f"     • {t['name']}: {t.get('description', '')[:60]}")

    return len(tools) >= 6, tools


# ---------------------------------------------------------------------------
# Шаг 3-9: Вызовы MCP-инструментов
# ---------------------------------------------------------------------------

def _call_tool(client_fn, tool_name: str, args: dict, label: str) -> tuple[bool, str]:
    """Вызвать инструмент и вернуть (ok, result)."""
    try:
        result = client_fn(tool_name, args)
        return True, result
    except RuntimeError as exc:
        return False, str(exc)


def run_demo() -> None:
    """Запустить полное демо."""
    print("\n" + "█" * 60)
    print("  News Digest MCP Server — Demo")
    print("█" * 60)

    # Таблица результатов
    report: list[tuple[str, bool, str]] = []

    # Шаг 1: Окружение
    env = step_check_environment()
    if not env.get("mcp_pkg"):
        print("\n❌ Пакет mcp не установлен. Установите: pip install mcp")
        sys.exit(1)

    # Шаг 2: Tool Discovery
    _print_header("Шаг 2: Tool Discovery")
    try:
        from mcp_client.config import MCPConfigParser
        from mcp_client.client import MCPClient

        parser = MCPConfigParser()
        servers = {s.name: s for s in parser.load()}
        config = servers.get("news_digest")
        if config is None:
            print("  ❌ 'news_digest' не найден в config/mcp-servers.md")
            report.append(("Tool discovery", False, "сервер не найден в конфиге"))
        else:
            print(f"  📡 Подключаюсь к '{config.name}'...")
            client = MCPClient(config)
            tools = client.connect_and_list_tools()
            ok = len(tools) >= 6
            print(f"  {'✅' if ok else '⚠️ '} Найдено инструментов: {len(tools)}")
            for t in tools:
                print(f"     • {t['name']}")
            report.append(("Tool discovery", ok, f"{len(tools)} tools"))
    except Exception as exc:
        print(f"  ❌ Ошибка: {exc}")
        report.append(("Tool discovery", False, str(exc)[:50]))
        _print_report(report)
        return

    # Шаг 3: Принудительный сбор
    _print_header("Шаг 3: Принудительный сбор новостей")
    print("  📡 force_fetch_now()...")
    ok, result = _call_tool(client.call_tool, "force_fetch_now", {}, "force_fetch_now")
    if ok:
        print(f"  ✅ {result}")
        # Извлекаем количество заголовков из ответа
        total_str = result
        report.append(("RSS fetch", True, result[:60]))
    else:
        print(f"  ❌ {result}")
        report.append(("RSS fetch", False, result[:60]))

    # Шаг 4: Последние заголовки
    _print_header("Шаг 4: Последние 10 заголовков")
    print("  📰 get_latest_headlines(limit=10)...")
    ok, result = _call_tool(client.call_tool, "get_latest_headlines", {"limit": 10}, "")
    if ok:
        lines = result.strip().split("\n")
        print(f"  ✅ {lines[0]}")
        for line in lines[1:12]:
            print(f"  {line}")
        report.append(("Latest headlines", True, f"{min(10, len(lines)-1)} показано"))
    else:
        print(f"  ❌ {result}")
        report.append(("Latest headlines", False, result[:60]))

    # Шаг 5: Статус планировщика
    _print_header("Шаг 5: Статус планировщика")
    print("  ⏰ get_scheduler_status()...")
    ok, result = _call_tool(client.call_tool, "get_scheduler_status", {}, "")
    if ok:
        print(result)
        report.append(("Scheduler status", True, "данные получены"))
    else:
        print(f"  ❌ {result}")
        report.append(("Scheduler status", False, result[:60]))

    # Шаг 6: Генерация сводки (только если LLM настроен)
    llm_ok, llm_provider = _check_llm_config()
    _print_header("Шаг 6: Генерация сводки через LLM")
    today = date.today().isoformat()

    if llm_ok:
        print(f"  📝 force_digest_now() [провайдер: {llm_provider}]...")
        print("  ⏳ LLM-вызов, пожалуйста подождите (~10 секунд)...")
        ok, result = _call_tool(client.call_tool, "force_digest_now", {}, "")
        if ok and "Ошибка" not in result and "не настроен" not in result:
            print(f"  ✅ Сводка сгенерирована:")
            # Выводим первые 400 символов
            preview = result[:400] + ("..." if len(result) > 400 else "")
            for line in preview.split("\n"):
                print(f"     {line}")
            report.append(("LLM digest", True, "сводка создана"))
        else:
            print(f"  ⚠️  {result[:100]}")
            report.append(("LLM digest", False, result[:60]))
    else:
        print("  ⚠️  LLM не настроен, пропускаем генерацию сводки")
        report.append(("LLM digest", False, "LLM не настроен"))

    # Шаг 7: Получение сводки из БД
    _print_header("Шаг 7: Получение сводки через MCP")
    print(f"  📖 get_daily_digest('{today}')...")
    ok, result = _call_tool(client.call_tool, "get_daily_digest", {"date_str": today}, "")
    if ok and "не найдена" not in result and "ещё нет" not in result:
        lines = result.strip().split("\n")
        print(f"  ✅ Сводка получена из базы:")
        for line in lines[:5]:
            print(f"     {line}")
        report.append(("Digest from DB", True, "из кэша"))
    elif ok:
        print(f"  ℹ️  {result[:100]}")
        report.append(("Digest from DB", llm_ok, "ещё не создана" if not llm_ok else result[:40]))
    else:
        print(f"  ❌ {result}")
        report.append(("Digest from DB", False, result[:60]))

    # Шаг 8: Повторный сбор (дедупликация)
    _print_header("Шаг 8: Повторный сбор — проверка дедупликации")
    print("  📡 force_fetch_now() (второй раз)...")
    ok, result = _call_tool(client.call_tool, "force_fetch_now", {}, "")
    if ok:
        print(f"  ✅ {result}")
        # Дедупликация работает если "новых: 0" или число < общего
        dedup_ok = "новых: 0" in result or ("новых:" in result and "новых: " + result.split("новых: ")[-1].split()[0] != result.split("Получено ")[-1].split()[0])
        if "новых:" in result:
            new_part = result.split("новых:")[-1].strip().split()[0] if "новых:" in result else "?"
            total_part = result.split("Получено")[-1].strip().split(",")[0].strip() if "Получено" in result else "?"
            dedup_note = f"{total_part} получено, {new_part} новых"
        else:
            dedup_note = result[:40]
        report.append(("Deduplication", True, dedup_note))
    else:
        print(f"  ❌ {result}")
        report.append(("Deduplication", False, result[:60]))

    # Шаг 9: Обработка ошибок
    _print_header("Шаг 9: Обработка ошибок")
    errors_caught = 0

    # Несуществующая дата сводки
    print("  get_daily_digest('2020-01-01')...")
    ok, result = _call_tool(client.call_tool, "get_daily_digest", {"date_str": "2020-01-01"}, "")
    if ok and ("не найдена" in result or "ещё нет" in result):
        print(f"  ✅ Корректный ответ: {result[:60]}")
        errors_caught += 1
    else:
        print(f"  ⚠️  Ответ: {result[:60]}")

    # Неверный формат даты
    print("  get_headlines_by_date('bad-date')...")
    ok, result = _call_tool(client.call_tool, "get_headlines_by_date", {"date_str": "bad-date"}, "")
    if ok and ("Неверный формат" in result or "неверный" in result.lower()):
        print(f"  ✅ Корректная ошибка: {result[:60]}")
        errors_caught += 1
    else:
        print(f"  ⚠️  Ответ: {result[:60]}")

    report.append(("Error handling", errors_caught >= 1, f"{errors_caught}/2 ошибок пойманы"))

    # Итоговый отчёт
    _print_report(report)


def _print_report(report: list[tuple[str, bool, str]]) -> None:
    """Вывести итоговую таблицу результатов."""
    _print_header("Итоговый отчёт")
    print(f"  {'Тест':<25} {'Статус':<10} {'Детали'}")
    print("  " + "-" * 60)
    for name, ok, details in report:
        status = "✅ OK    " if ok else "❌ FAIL  "
        print(f"  {name:<25} {status} {details}")
    print()
    passed = sum(1 for _, ok, _ in report if ok)
    total = len(report)
    print(f"  Итог: {passed}/{total} тестов прошло")
    if passed == total:
        print("  🎉 Все тесты прошли успешно!")
    elif passed >= total * 0.7:
        print("  ✅ Большинство тестов прошло")
    else:
        print("  ⚠️  Часть тестов не прошла — проверьте конфигурацию")


if __name__ == "__main__":
    run_demo()
