"""MCP-пайплайн новостей: получение → суммаризация → доставка.

Образовательные концепции:
1. MCP-инструменты как этапы пайплайна (chain of tools)
2. Передача данных между инструментами через JSON-строки
3. Самодостаточный инструмент (summarize_news сам вызывает LLM)
4. Оркестратор пайплайна (run_news_pipeline вызывает три инструмента цепочкой)
5. Рабочие часы (09:00-17:00 МСК) для автоматического запуска

Архитектура пайплайна:
  fetch_news → summarize_news → deliver_news
       ↓             ↓               ↓
   JSON (статьи)  JSON (сводки)   Отчёт (SQLite + Telegram)

Все три инструмента доступны для ручного вызова по отдельности.
run_news_pipeline выполняет цепочку автоматически.

Запуск:
    python -m mcp_server.news_server

Инструменты:
    fetch_news          — получить и сгруппировать новости из RSS
    summarize_news      — суммаризировать категории через LLM
    deliver_news        — сохранить в SQLite и отправить в Telegram
    run_news_pipeline   — полный пайплайн одной командой
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta

from mcp.server.fastmcp import FastMCP

from mcp_server.news_api import (
    fetch_rss_by_categories,
    format_telegram_message,
    get_db_path,
    get_existing_links,
    get_pipeline_status,
    get_summaries,
    init_db,
    mark_links_processed,
    mark_telegram_sent,
    save_summaries,
    send_telegram_message,
)
from mcp_server.llm_client import create_llm_fn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

mcp = FastMCP("News Pipeline Server")

# Московское время
_MSK = timezone(timedelta(hours=3))


def _get_env_int(key: str, default: int) -> int:
    """Прочитать целочисленную переменную окружения."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_list(key: str, default: str = "") -> list[str]:
    """Прочитать список через запятую из .env."""
    raw = os.environ.get(key, default).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Инструмент 1: fetch_news
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_news(
    feed_url: str = "",
    max_per_category: int = 0,
    categories: str = "",
) -> str:
    """
    Получить новости из RSS-фида и сгруппировать по категориям.

    Шаг 1/3 пайплайна. Результат передаётся в summarize_news.

    Args:
        feed_url: URL RSS-фида (по умолчанию из NEWS_RSS_FEEDS в .env).
        max_per_category: Макс. статей на категорию (0 = из NEWS_MAX_ARTICLES_PER_CATEGORY).
        categories: Категории через запятую (пустое = из NEWS_CATEGORIES в .env).

    Returns:
        JSON-строка:
        {
          "date": "2026-03-12",
          "categories": {
            "economics": [{"title": ..., "description": ..., "link": ..., "pubDate": ...}, ...],
            ...
          }
        }
    """
    # Читаем .env если параметры не заданы
    if not feed_url:
        feeds_raw = os.environ.get("NEWS_RSS_FEEDS", "https://lenta.ru/rss").strip()
        # Берём первый URL если несколько
        feed_url = feeds_raw.split(",")[0].strip()

    if max_per_category <= 0:
        max_per_category = _get_env_int("NEWS_MAX_ARTICLES_PER_CATEGORY", 10)

    if not categories:
        cats_list = _get_env_list("NEWS_CATEGORIES", "russia,world,economics,science,sport,culture")
    else:
        cats_list = [c.strip() for c in categories.split(",") if c.strip()]

    # Дедупликация: исключаем уже обработанные статьи
    db_path = get_db_path()
    existing_links = get_existing_links(db_path)

    try:
        grouped = fetch_rss_by_categories(
            feed_url=feed_url,
            max_per_category=max_per_category,
            categories=cats_list,
            existing_links=existing_links,
        )
    except (ConnectionError, ValueError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    today = datetime.now(_MSK).strftime("%Y-%m-%d")

    result = {
        "date": today,
        "categories": grouped,
    }
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Инструмент 2: summarize_news
# ---------------------------------------------------------------------------

@mcp.tool()
def summarize_news(news_json: str) -> str:
    """
    Суммаризировать новости по категориям через LLM.

    Шаг 2/3 пайплайна. Получает JSON из fetch_news, возвращает JSON для deliver_news.

    Образовательная концепция: MCP-инструмент вызывает LLM самостоятельно
    через urllib + API-ключ из .env. Это делает инструмент самодостаточным.

    Args:
        news_json: JSON-строка из fetch_news.

    Returns:
        JSON-строка:
        {
          "date": "2026-03-12",
          "summaries": {"economics": "Краткое содержание...", "science": "...", ...},
          "article_counts": {"economics": 10, "science": 7, ...}
        }
    """
    # Парсим входной JSON
    try:
        data = json.loads(news_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Невалидный JSON: {exc}"}, ensure_ascii=False)

    if "error" in data:
        return json.dumps({"error": data["error"]}, ensure_ascii=False)

    date_str = data.get("date", datetime.now(_MSK).strftime("%Y-%m-%d"))
    categories_data: dict[str, list[dict]] = data.get("categories", {})

    if not categories_data:
        return json.dumps(
            {"date": date_str, "summaries": {}, "article_counts": {}},
            ensure_ascii=False,
        )

    # Инициализация LLM
    try:
        llm_fn = create_llm_fn(timeout=90.0)
    except ValueError as exc:
        return json.dumps(
            {"error": f"LLM не настроен: {exc}"},
            ensure_ascii=False,
        )

    # Суммаризация каждой категории
    from mcp_server.news_api import _CATEGORY_NAMES
    summaries: dict[str, str] = {}
    article_counts: dict[str, int] = {}

    for category, articles in categories_data.items():
        if not articles:
            continue

        article_counts[category] = len(articles)
        cat_name = _CATEGORY_NAMES.get(category, category.capitalize())

        # Формируем промпт
        articles_text = "\n".join(
            f"- {a['title']}"
            + (f": {a['description']}" if a.get("description") else "")
            for a in articles
        )
        prompt = (
            f"Сделай краткую сводку новостей категории «{cat_name}» "
            f"на русском языке, 3-5 предложений. Выдели главные события.\n\n"
            f"Статьи ({len(articles)} шт.):\n{articles_text}"
        )

        try:
            summary = llm_fn(prompt)
            summaries[category] = summary.strip()
        except Exception as exc:
            summaries[category] = f"[Ошибка суммаризации: {exc}]"

    result = {
        "date": date_str,
        "summaries": summaries,
        "article_counts": article_counts,
    }
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Инструмент 3: deliver_news
# ---------------------------------------------------------------------------

@mcp.tool()
def deliver_news(summaries_json: str) -> str:
    """
    Сохранить суммаризации в SQLite и отправить в Telegram.

    Шаг 3/3 пайплайна. Финальный этап доставки.

    Args:
        summaries_json: JSON-строка из summarize_news.

    Returns:
        Отчёт о доставке: сколько категорий сохранено, отправлено ли в Telegram.
    """
    # Парсим входной JSON
    try:
        data = json.loads(summaries_json)
    except json.JSONDecodeError as exc:
        return f"Ошибка: невалидный JSON: {exc}"

    if "error" in data:
        return f"Ошибка от предыдущего шага: {data['error']}"

    date_str = data.get("date", datetime.now(_MSK).strftime("%Y-%m-%d"))
    summaries: dict[str, str] = data.get("summaries", {})
    article_counts: dict[str, int] = data.get("article_counts", {})

    if not summaries:
        return "Нет суммаризаций для сохранения."

    # Сохраняем в SQLite
    db_path = get_db_path()
    init_db(db_path)

    saved_count = save_summaries(
        date=date_str,
        summaries=summaries,
        article_counts=article_counts,
        telegram_sent=False,
        db_path=db_path,
    )

    # Помечаем ссылки как обработанные (дедупликация для следующего запуска)
    # Получаем ссылки из оригинального fetch_news — они уже не нужны здесь,
    # так как пометка произошла бы в fetch_news при следующем запуске
    report_lines = [f"💾 Сохранено в SQLite: {saved_count} категорий"]

    # Отправка в Telegram
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if token and chat_id:
        message = format_telegram_message(date_str, summaries)
        success = send_telegram_message(message, token, chat_id)
        if success:
            mark_telegram_sent(date_str, db_path)
            report_lines.append(f"📲 Telegram {chat_id}: отправлено")
        else:
            report_lines.append(f"📲 Telegram {chat_id}: ошибка отправки")
    else:
        report_lines.append("📲 Telegram: TELEGRAM_BOT_TOKEN не задан, пропуск")

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Инструмент 4: run_news_pipeline (оркестратор)
# ---------------------------------------------------------------------------

@mcp.tool()
def run_news_pipeline() -> str:
    """
    Запустить полный пайплайн: получение → суммаризация → доставка.

    Использует настройки из .env:
    - NEWS_RSS_FEEDS, NEWS_MAX_ARTICLES_PER_CATEGORY, NEWS_CATEGORIES
    - NEWS_WORK_HOURS_START / NEWS_WORK_HOURS_END (рабочие часы МСК)
    - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    Рабочие часы: пайплайн не запускается за пределами заданного окна.
    Оркестратор вызывает fetch_news, summarize_news, deliver_news
    как обычные Python-функции (все в одном процессе).

    Returns:
        Полный отчёт о выполнении пайплайна.
    """
    # Проверяем рабочие часы (МСК)
    now_msk = datetime.now(_MSK)
    start_hour = _get_env_int("NEWS_WORK_HOURS_START", 9)
    end_hour = _get_env_int("NEWS_WORK_HOURS_END", 17)

    if not (start_hour <= now_msk.hour < end_hour):
        time_str = now_msk.strftime("%H:%M")
        return (
            f"Пайплайн не запущен: нерабочее время ({time_str} МСК). "
            f"Рабочее окно: {start_hour:02d}:00–{end_hour:02d}:00 МСК."
        )

    report_lines = [
        f"🚀 Запуск пайплайна [{now_msk.strftime('%H:%M')} МСК]",
        "",
    ]

    # Шаг 1: Получение новостей
    report_lines.append("Шаг 1/3: получение новостей...")
    news_json = fetch_news()

    try:
        news_data = json.loads(news_json)
    except json.JSONDecodeError:
        news_data = {}

    if "error" in news_data:
        report_lines.append(f"❌ Ошибка на шаге 1: {news_data['error']}")
        return "\n".join(report_lines)

    cats = news_data.get("categories", {})
    total_articles = sum(len(v) for v in cats.values())
    report_lines.append(
        f"✅ Шаг 1/3: получено {total_articles} статей в {len(cats)} категориях"
    )
    for cat, articles in cats.items():
        from mcp_server.news_api import _CATEGORY_NAMES
        cat_name = _CATEGORY_NAMES.get(cat, cat)
        report_lines.append(f"   {cat_name}: {len(articles)} статей")

    # Пометить ссылки как обработанные (дедупликация)
    db_path = get_db_path()
    for cat, articles in cats.items():
        links = [a["link"] for a in articles if a.get("link")]
        mark_links_processed(links, category=cat, db_path=db_path)

    report_lines.append("")

    # Шаг 2: Суммаризация
    report_lines.append("Шаг 2/3: суммаризация через LLM...")
    summaries_json = summarize_news(news_json)

    try:
        summaries_data = json.loads(summaries_json)
    except json.JSONDecodeError:
        summaries_data = {}

    if "error" in summaries_data:
        report_lines.append(f"❌ Ошибка на шаге 2: {summaries_data['error']}")
        return "\n".join(report_lines)

    summaries = summaries_data.get("summaries", {})
    report_lines.append(f"✅ Шаг 2/3: суммаризировано {len(summaries)} категорий")
    report_lines.append("")

    # Шаг 3: Доставка
    report_lines.append("Шаг 3/3: сохранение и отправка...")
    delivery_report = deliver_news(summaries_json)
    report_lines.append(f"✅ {delivery_report}")
    report_lines.append("")

    # Итог
    report_lines.append(
        f"🎉 Пайплайн завершён: {total_articles} статей → "
        f"{len(summaries)} суммаризаций → доставлено"
    )
    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Инициализируем БД при старте
    db_path = get_db_path()
    try:
        init_db(db_path)
        print(f"БД инициализирована: {db_path}", file=sys.stderr)
    except Exception as exc:
        print(f"Предупреждение: не удалось инициализировать БД: {exc}", file=sys.stderr)

    # Запуск MCP-сервера в stdio-режиме
    mcp.run()
