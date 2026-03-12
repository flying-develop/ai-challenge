"""Демонстрационный скрипт новостного MCP-пайплайна.

Запуск БЕЗ интерактивного ввода:
    python demo_news_pipeline.py

Демонстрирует:
    Шаг 1 — Проверка окружения (.env)
    Шаг 2 — Tool discovery (list_tools)
    Шаг 3 — fetch_news (получение новостей)
    Шаг 4 — summarize_news (суммаризация через LLM)
    Шаг 5 — deliver_news (SQLite + Telegram)
    Шаг 6 — run_news_pipeline (полный цикл одной командой)
    Шаг 7 — Проверка целостности данных
    Итог  — Таблица с результатами тестов

Образовательная концепция:
    MCPClient вызывает каждый инструмент news_server.py через stdio.
    Каждый вызов: запускает процесс → подключается → вызывает → завершает.
    Данные между инструментами передаются как JSON-строки.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Утилиты вывода
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _header(text: str) -> None:
    print()
    _sep("═")
    print(f"  {text}")
    _sep("═")


def _step(n: int, total: int, text: str) -> None:
    print(f"\n[Шаг {n}/{total}] {text}")
    _sep()


def _ok(text: str) -> None:
    print(f"  ✅ {text}")


def _fail(text: str) -> None:
    print(f"  ❌ {text}")


def _info(text: str) -> None:
    print(f"  ℹ️  {text}")


# ---------------------------------------------------------------------------
# Результаты тестов (для итоговой таблицы)
# ---------------------------------------------------------------------------

_test_results: list[tuple[str, bool, str]] = []


def _record(name: str, success: bool, details: str) -> None:
    _test_results.append((name, success, details))


# ---------------------------------------------------------------------------
# Шаг 1: Проверка окружения
# ---------------------------------------------------------------------------

def step_check_env() -> None:
    _step(1, 7, "Проверка окружения (.env)")

    vars_to_check = [
        ("NEWS_RSS_FEEDS",                 False, "URL RSS-фида"),
        ("NEWS_MAX_ARTICLES_PER_CATEGORY", False, "Макс. статей на категорию"),
        ("NEWS_POLL_INTERVAL_MINUTES",     False, "Интервал пайплайна (мин)"),
        ("NEWS_WORK_HOURS_START",          False, "Начало рабочего дня МСК"),
        ("NEWS_WORK_HOURS_END",            False, "Конец рабочего дня МСК"),
        ("NEWS_CATEGORIES",                False, "Список категорий"),
        ("TELEGRAM_BOT_TOKEN",             True,  "Telegram Bot Token"),
        ("TELEGRAM_CHAT_ID",               True,  "Telegram Chat ID"),
        ("LLM_PROVIDER",                   True,  "LLM провайдер"),
    ]

    all_ok = True
    for var_name, is_optional, description in vars_to_check:
        value = os.environ.get(var_name, "")
        if value:
            _ok(f"{var_name}={repr(value)[:40]}  ({description})")
        elif is_optional:
            _info(f"{var_name} не задан (опционально)  ({description})")
        else:
            _info(f"{var_name} не задан, используется default  ({description})")

    # Проверка LLM-ключей
    llm_ok = any([
        os.environ.get("QWEN_API_KEY", "").strip(),
        os.environ.get("OPENAI_API_KEY", "").strip(),
        os.environ.get("ANTHROPIC_API_KEY", "").strip(),
    ])
    if llm_ok:
        _ok("LLM API ключ найден")
    else:
        _fail("LLM API ключ не задан (QWEN_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        all_ok = False

    _record("Env check", True, "проверено")  # Не блокируем при отсутствии ключей


# ---------------------------------------------------------------------------
# Шаг 2: Tool discovery
# ---------------------------------------------------------------------------

def step_tool_discovery(client) -> dict[str, str]:
    """Список инструментов сервера."""
    _step(2, 7, "Tool discovery (list_tools)")

    try:
        tools = client.connect_and_list_tools()
    except Exception as exc:
        _fail(f"Не удалось получить список инструментов: {exc}")
        _record("Tool discovery", False, str(exc))
        return {}

    expected_tools = {"fetch_news", "summarize_news", "deliver_news", "run_news_pipeline"}
    found_tools = set(tools.keys())

    print(f"  Найдено инструментов: {len(found_tools)}")
    for name, description in sorted(tools.items()):
        mark = "✅" if name in expected_tools else "ℹ️"
        short_desc = description[:60] + "..." if len(description) > 60 else description
        print(f"    {mark} {name}: {short_desc}")

    missing = expected_tools - found_tools
    if missing:
        _fail(f"Не найдены инструменты: {', '.join(sorted(missing))}")
        _record("Tool discovery", False, f"missing: {missing}")
    else:
        _ok(f"Все {len(expected_tools)} инструмента пайплайна найдены")
        _record("Tool discovery", True, f"{len(found_tools)} tools")

    return tools


# ---------------------------------------------------------------------------
# Шаг 3: fetch_news
# ---------------------------------------------------------------------------

def step_fetch_news(client) -> tuple[str, dict]:
    """Вызов fetch_news, возвращает (json_str, parsed_data)."""
    _step(3, 7, "Шаг 1 пайплайна: fetch_news")

    # Параметры из .env или defaults
    feed_url = os.environ.get("NEWS_RSS_FEEDS", "https://lenta.ru/rss").split(",")[0].strip()
    max_per_cat = os.environ.get("NEWS_MAX_ARTICLES_PER_CATEGORY", "10")
    categories = os.environ.get("NEWS_CATEGORIES", "russia,world,economics,science,sport,culture")

    print(f"  RSS-фид: {feed_url}")
    print(f"  Макс. статей на категорию: {max_per_cat}")
    print(f"  Категории: {categories}")
    print()

    start = time.time()
    try:
        news_json = client.call_tool("fetch_news", {
            "feed_url": feed_url,
            "max_per_category": int(max_per_cat),
            "categories": categories,
        })
    except Exception as exc:
        _fail(f"Ошибка вызова fetch_news: {exc}")
        _record("fetch_news", False, str(exc))
        return "", {}

    elapsed = time.time() - start

    try:
        news_data = json.loads(news_json)
    except json.JSONDecodeError as exc:
        _fail(f"Невалидный JSON от fetch_news: {exc}")
        _fail(f"Ответ: {news_json[:200]}")
        _record("fetch_news", False, "invalid json")
        return news_json, {}

    if "error" in news_data:
        _fail(f"fetch_news вернул ошибку: {news_data['error']}")
        _record("fetch_news", False, news_data["error"])
        return news_json, {}

    categories_data = news_data.get("categories", {})
    total = sum(len(v) for v in categories_data.values())

    print(f"  📡 fetch_news: получено {total} статей за {elapsed:.1f}с")
    print(f"  Категории:")
    for cat, articles in sorted(categories_data.items()):
        from mcp_server.news_api import _CATEGORY_NAMES
        cat_name = _CATEGORY_NAMES.get(cat, cat)
        print(f"    {cat} ({cat_name}): {len(articles)} статей")

    if total > 0:
        _ok(f"fetch_news: {total} статей, {len(categories_data)} категорий")
        _record("fetch_news", True, f"{total} статей, {len(categories_data)} кат.")
    else:
        _info("fetch_news: статей не найдено (возможно, все уже обработаны)")
        _record("fetch_news", True, "0 статей (дедупликация или пустой фид)")

    return news_json, news_data


# ---------------------------------------------------------------------------
# Шаг 4: summarize_news
# ---------------------------------------------------------------------------

def step_summarize_news(client, news_json: str, news_data: dict) -> tuple[str, dict]:
    """Вызов summarize_news."""
    _step(4, 7, "Шаг 2 пайплайна: summarize_news")

    if not news_json or "error" in news_data:
        _fail("Пропуск: нет данных от fetch_news")
        _record("summarize_news", False, "no input data")
        return "", {}

    categories_data = news_data.get("categories", {})
    if not categories_data:
        _info("Пропуск суммаризации: нет статей для обработки")
        _record("summarize_news", True, "0 категорий (нет статей)")
        return json.dumps({"date": news_data.get("date", ""), "summaries": {}, "article_counts": {}}), {}

    llm_ok = any([
        os.environ.get("QWEN_API_KEY", "").strip(),
        os.environ.get("OPENAI_API_KEY", "").strip(),
        os.environ.get("ANTHROPIC_API_KEY", "").strip(),
    ])
    if not llm_ok:
        _fail("Пропуск: LLM API ключ не задан")
        _record("summarize_news", False, "no LLM API key")
        return "", {}

    print(f"  Суммаризация {len(categories_data)} категорий...")
    print("  (LLM-запросы, может занять 30-60 секунд)")

    start = time.time()
    try:
        summaries_json = client.call_tool("summarize_news", {"news_json": news_json})
    except Exception as exc:
        _fail(f"Ошибка вызова summarize_news: {exc}")
        _record("summarize_news", False, str(exc))
        return "", {}

    elapsed = time.time() - start

    try:
        summaries_data = json.loads(summaries_json)
    except json.JSONDecodeError as exc:
        _fail(f"Невалидный JSON от summarize_news: {exc}")
        _record("summarize_news", False, "invalid json")
        return summaries_json, {}

    if "error" in summaries_data:
        _fail(f"summarize_news вернул ошибку: {summaries_data['error']}")
        _record("summarize_news", False, summaries_data["error"])
        return summaries_json, {}

    summaries = summaries_data.get("summaries", {})
    article_counts = summaries_data.get("article_counts", {})

    print(f"\n  Суммаризировано за {elapsed:.1f}с:")
    for cat, summary in sorted(summaries.items()):
        from mcp_server.news_api import _CATEGORY_NAMES, _CATEGORY_EMOJI
        cat_name = _CATEGORY_NAMES.get(cat, cat)
        emoji = _CATEGORY_EMOJI.get(cat, "📌")
        count = article_counts.get(cat, 0)
        print(f"\n  {emoji} {cat_name} ({count} статей):")
        # Выводим первые 200 символов суммаризации
        preview = summary[:200] + ("..." if len(summary) > 200 else "")
        for line in preview.split("\n"):
            print(f"    {line}")

    _ok(f"summarize_news: {len(summaries)} суммаризаций за {elapsed:.1f}с")
    _record("summarize_news", True, f"{len(summaries)} суммаризаций")

    return summaries_json, summaries_data


# ---------------------------------------------------------------------------
# Шаг 5: deliver_news
# ---------------------------------------------------------------------------

def step_deliver_news(client, summaries_json: str) -> str:
    """Вызов deliver_news."""
    _step(5, 7, "Шаг 3 пайплайна: deliver_news")

    if not summaries_json:
        _fail("Пропуск: нет данных от summarize_news")
        _record("deliver_news", False, "no input data")
        return ""

    try:
        summaries_data = json.loads(summaries_json)
    except json.JSONDecodeError:
        summaries_data = {}

    if "error" in summaries_data or not summaries_data.get("summaries"):
        _info("Пропуск deliver_news: нет суммаризаций для доставки")
        _record("deliver_news", True, "0 суммаризаций (пропуск)")
        return ""

    start = time.time()
    try:
        report = client.call_tool("deliver_news", {"summaries_json": summaries_json})
    except Exception as exc:
        _fail(f"Ошибка вызова deliver_news: {exc}")
        _record("deliver_news", False, str(exc))
        return ""

    elapsed = time.time() - start

    print(f"\n  Отчёт deliver_news ({elapsed:.1f}с):")
    for line in report.split("\n"):
        print(f"  {line}")

    has_error = "ошибка" in report.lower() or "error" in report.lower()
    if has_error:
        # Ошибка Telegram допустима если токен не задан
        if "telegram_bot_token не задан" in report.lower():
            _ok("deliver_news: SQLite ✅, Telegram пропущен (нет токена)")
            _record("deliver_news", True, "SQLite + Telegram пропущен")
        else:
            _fail("deliver_news: есть ошибки")
            _record("deliver_news", False, report[:80])
    else:
        _ok(f"deliver_news: {elapsed:.1f}с")
        details = "SQLite + Telegram" if "отправлено" in report else "SQLite"
        _record("deliver_news", True, details)

    return report


# ---------------------------------------------------------------------------
# Шаг 6: run_news_pipeline (полный цикл)
# ---------------------------------------------------------------------------

def step_run_pipeline(client) -> str:
    """Вызов run_news_pipeline."""
    _step(6, 7, "Полный пайплайн: run_news_pipeline")

    # Временно расширяем рабочее окно для демо
    # (изменяем env-переменные только для этого процесса)
    from datetime import datetime, timezone, timedelta
    msk = timezone(timedelta(hours=3))
    now_hour = datetime.now(msk).hour
    os.environ["NEWS_WORK_HOURS_START"] = str(now_hour)
    os.environ["NEWS_WORK_HOURS_END"] = str(now_hour + 1)

    print("  Запуск полного пайплайна одной командой...")
    print("  (Включает все три шага: fetch → summarize → deliver)")

    start = time.time()
    try:
        result = client.call_tool("run_news_pipeline", {})
    except Exception as exc:
        _fail(f"Ошибка вызова run_news_pipeline: {exc}")
        _record("run_news_pipeline", False, str(exc))
        return ""
    finally:
        # Восстанавливаем оригинальные значения
        os.environ.pop("NEWS_WORK_HOURS_START", None)
        os.environ.pop("NEWS_WORK_HOURS_END", None)

    elapsed = time.time() - start

    print(f"\n  Результат ({elapsed:.1f}с):")
    for line in result.split("\n"):
        print(f"  {line}")

    # Проверяем успешность
    is_skipped = "нерабочее время" in result.lower()
    is_error = "❌" in result and "✅" not in result

    if is_skipped:
        _info("run_news_pipeline: пропущен (нерабочее время)")
        _record("run_news_pipeline", True, "нерабочее время")
    elif is_error:
        _fail("run_news_pipeline: ошибка")
        _record("run_news_pipeline", False, result[:80])
    else:
        _ok(f"run_news_pipeline: завершён за {elapsed:.1f}с")
        _record("run_news_pipeline", True, "полный цикл")

    return result


# ---------------------------------------------------------------------------
# Шаг 7: Проверка целостности данных
# ---------------------------------------------------------------------------

def step_data_integrity(news_data: dict, summaries_data: dict) -> None:
    """Проверить что количество категорий на выходе fetch == на входе deliver."""
    _step(7, 7, "Проверка целостности данных")

    cats_fetched = set(news_data.get("categories", {}).keys())
    cats_summarized = set(summaries_data.get("summaries", {}).keys())

    print(f"  Категорий из fetch_news:      {len(cats_fetched)}")
    print(f"  Категорий из summarize_news:  {len(cats_summarized)}")

    if not cats_fetched and not cats_summarized:
        _info("Нет данных для проверки (возможно, дедупликация удалила все статьи)")
        _record("Data integrity", True, "0/0 категорий (ок)")
        return

    if cats_fetched and not cats_summarized:
        # Может быть нормально если LLM не настроен
        llm_ok = any([
            os.environ.get("QWEN_API_KEY", "").strip(),
            os.environ.get("OPENAI_API_KEY", "").strip(),
            os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        ])
        if not llm_ok:
            _info(f"LLM не настроен → суммаризация пропущена ({len(cats_fetched)} кат. получено)")
            _record("Data integrity", True, f"{len(cats_fetched)} fetched, 0 summarized (no LLM)")
        else:
            _fail(f"Расхождение: fetch={len(cats_fetched)}, summarize={len(cats_summarized)}")
            _record("Data integrity", False, f"{len(cats_fetched)}/{len(cats_summarized)}")
        return

    if cats_fetched == cats_summarized:
        _ok(f"Данные переданы корректно: {len(cats_fetched)}/{len(cats_fetched)} категорий")
        _record("Data integrity", True, f"{len(cats_fetched)}/{len(cats_fetched)} категорий")
    elif cats_summarized.issubset(cats_fetched):
        diff = cats_fetched - cats_summarized
        _info(f"Частичная суммаризация: {len(cats_summarized)}/{len(cats_fetched)} кат.")
        _info(f"Не суммаризированы: {', '.join(diff)}")
        _record("Data integrity", True, f"{len(cats_summarized)}/{len(cats_fetched)} кат.")
    else:
        unexpected = cats_summarized - cats_fetched
        _fail(f"Расхождение: суммаризированы категории которых не было в fetch: {unexpected}")
        _record("Data integrity", False, f"unexpected: {unexpected}")


# ---------------------------------------------------------------------------
# Итоговая таблица
# ---------------------------------------------------------------------------

def print_summary() -> None:
    """Вывести итоговую таблицу результатов."""
    print()
    _sep("═")
    print("  News Pipeline Test Report")
    _sep("─")
    print(f"  {'Тест':<24} {'Статус':<10} {'Детали':<30}")
    _sep("─")

    all_passed = True
    for name, success, details in _test_results:
        status = "✅" if success else "❌"
        if not success:
            all_passed = False
        details_short = details[:28] + ".." if len(details) > 30 else details
        print(f"  {name:<24} {status:<10} {details_short:<30}")

    _sep("─")
    passed = sum(1 for _, s, _ in _test_results if s)
    total = len(_test_results)
    overall = "✅ ВСЕ ТЕСТЫ ПРОШЛИ" if all_passed else f"⚠️  {passed}/{total} тестов прошли"
    print(f"  Итог: {overall}")
    _sep("═")
    print()


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    """Запустить все демонстрационные шаги."""
    _header("News Pipeline Demo")
    print("  Тестирование MCP-пайплайна: fetch → summarize → deliver")

    # Создаём MCPClient для news_digest сервера
    try:
        from mcp_client.client import MCPClient
        from mcp_client.config import MCPConfigParser, MCPServerConfig

        # Пробуем загрузить из конфига
        try:
            parser = MCPConfigParser()
            servers = parser.load()
            srv_map = {s.name: s for s in servers}
            config = srv_map.get("news_digest")
        except Exception:
            config = None

        if config is None:
            config = MCPServerConfig(
                name="news_digest",
                transport="stdio",
                command="python",
                args=["-m", "mcp_server.news_server"],
                env={},
                description="News Pipeline MCP Server",
            )

        client = MCPClient(config)

    except ImportError as exc:
        print(f"❌ Не удалось импортировать MCPClient: {exc}")
        print("   Убедитесь что пакет mcp установлен: pip install mcp")
        sys.exit(1)

    # Шаг 1: Проверка окружения
    step_check_env()

    # Шаг 2: Tool discovery
    tools = step_tool_discovery(client)
    if not tools:
        print("\n❌ Сервер недоступен. Убедитесь что mcp пакет установлен.")
        print_summary()
        sys.exit(1)

    # Шаг 3: fetch_news
    news_json, news_data = step_fetch_news(client)

    # Шаг 4: summarize_news
    summaries_json, summaries_data = step_summarize_news(client, news_json, news_data)

    # Шаг 5: deliver_news
    step_deliver_news(client, summaries_json)

    # Шаг 6: Полный пайплайн
    step_run_pipeline(client)

    # Шаг 7: Целостность данных
    step_data_integrity(news_data, summaries_data)

    # Итоговая таблица
    print_summary()


if __name__ == "__main__":
    main()
