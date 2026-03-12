"""MCP-сервер "News Digest": сбор новостей и генерация дневных сводок.

Образовательные концепции:
1. Фоновые задачи в MCP-сервере (threading + schedule)
2. Персистентное хранение (SQLite) внутри MCP-сервера
3. Интеграция MCP-сервера с LLM (сервер сам вызывает LLM)
4. Агрегация и суммаризация данных по расписанию

Архитектура:
  - FastMCP: регистрирует инструменты, обрабатывает stdio транспорт
  - NewsStorage: SQLite БД (headlines, daily_digests, scheduler_log)
  - NewsScheduler: фоновый поток — каждый час RSS, каждый день сводка
  - llm_fn: вызов LLM для генерации сводок (читает провайдер из .env)

Запуск:
    python -m mcp_server.news_server

Инструменты (6 штук):
    get_latest_headlines   — последние N заголовков
    get_headlines_by_date  — заголовки за дату
    get_daily_digest       — сводка дня (от LLM)
    get_scheduler_status   — статус планировщика
    force_fetch_now        — принудительный сбор
    force_digest_now       — принудительная генерация сводки
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_server.news_storage import NewsStorage
from mcp_server.scheduler import NewsScheduler

# ---------------------------------------------------------------------------
# Глобальные объекты (инициализируются в if __name__ == "__main__")
# ---------------------------------------------------------------------------
# FastMCP регистрирует инструменты через декораторы на уровне модуля,
# поэтому storage и scheduler объявлены как глобальные переменные.
# При запуске через python -m они инициализируются в блоке __main__.

mcp = FastMCP("News Digest Server")

# Будут инициализированы при запуске
_storage: NewsStorage | None = None
_scheduler: NewsScheduler | None = None


def _get_storage() -> NewsStorage:
    """Вернуть глобальное хранилище. Инициализирует с путём по умолчанию если нужно."""
    global _storage
    if _storage is None:
        # Fallback: создать хранилище в текущей директории
        # (нужно только при импорте модуля без __main__)
        _storage = NewsStorage("data/news.db")
    return _storage


def _get_scheduler() -> NewsScheduler | None:
    """Вернуть глобальный планировщик."""
    return _scheduler


# ---------------------------------------------------------------------------
# Инструмент 1: Последние заголовки
# ---------------------------------------------------------------------------

@mcp.tool()
def get_latest_headlines(limit: int = 20) -> str:
    """
    Получить последние заголовки новостей.

    Args:
        limit: Количество заголовков (по умолчанию 20, максимум 100)

    Returns:
        Список последних заголовков с датами публикации
    """
    limit = max(1, min(limit, 100))
    storage = _get_storage()
    headlines = storage.get_headlines(limit=limit)

    if not headlines:
        return "Заголовки не найдены. Используйте force_fetch_now() для сбора."

    lines = [f"Последние {len(headlines)} заголовков новостей:\n"]
    for i, h in enumerate(headlines, 1):
        # Форматируем дату: берём только время из ISO 8601
        pub_date = h.get("pub_date", "")
        time_str = ""
        if pub_date and "T" in pub_date:
            time_part = pub_date.split("T")[1]
            # Берём HH:MM из "HH:MM:SS+03:00"
            time_str = time_part[:5]
        elif pub_date:
            time_str = pub_date[:10]

        title = h.get("title", "")
        link = h.get("link", "")
        if time_str:
            lines.append(f"  {i}. [{time_str}] {title}")
        else:
            lines.append(f"  {i}. {title}")
        if link:
            lines.append(f"     {link}")

    total = storage.count_headlines()
    lines.append(f"\nВсего в базе: {total} заголовков")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Инструмент 2: Заголовки за дату
# ---------------------------------------------------------------------------

@mcp.tool()
def get_headlines_by_date(date_str: str) -> str:
    """
    Получить заголовки новостей за конкретную дату.

    Args:
        date_str: Дата в формате YYYY-MM-DD (например, 2026-03-11)

    Returns:
        Все собранные заголовки за указанную дату
    """
    # Валидация формата даты
    try:
        from datetime import datetime
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return f"Неверный формат даты '{date_str}'. Используйте YYYY-MM-DD"

    storage = _get_storage()
    headlines = storage.get_headlines(date=date_str, limit=500)

    if not headlines:
        return f"Заголовки за {date_str} не найдены."

    lines = [f"Заголовки за {date_str} ({len(headlines)} шт.):\n"]
    for i, h in enumerate(headlines, 1):
        pub_date = h.get("pub_date", "")
        time_str = ""
        if pub_date and "T" in pub_date:
            time_part = pub_date.split("T")[1]
            time_str = time_part[:5]

        title = h.get("title", "")
        if time_str:
            lines.append(f"  {i}. [{time_str}] {title}")
        else:
            lines.append(f"  {i}. {title}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Инструмент 3: Дневная сводка
# ---------------------------------------------------------------------------

@mcp.tool()
def get_daily_digest(date_str: str = "") -> str:
    """
    Получить сводку новостей за день (сгенерированную LLM).

    Args:
        date_str: Дата YYYY-MM-DD (по умолчанию — последняя доступная сводка)

    Returns:
        Текст дневной сводки или сообщение что сводка ещё не готова
    """
    storage = _get_storage()

    if date_str:
        # Валидация формата
        try:
            from datetime import datetime
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return f"Неверный формат даты '{date_str}'. Используйте YYYY-MM-DD"
        digest = storage.get_digest(date_str)
        if not digest:
            return f"Сводка за {date_str} не найдена. Используйте force_digest_now() для генерации."
    else:
        digest = storage.get_latest_digest()
        if not digest:
            return (
                "Сводки ещё нет. Используйте force_digest_now() для генерации "
                "или дождитесь 23:00 (автоматическая генерация)."
            )

    date_label = digest.get("date", "")
    count = digest.get("headline_count", 0)
    text = digest.get("digest_text", "")
    created_at = digest.get("created_at", "")

    lines = [
        f"Сводка дня за {date_label} (на основе {count} заголовков):",
        f"Создана: {created_at[:19].replace('T', ' ')} UTC",
        "",
        text,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Инструмент 4: Статус планировщика
# ---------------------------------------------------------------------------

@mcp.tool()
def get_scheduler_status() -> str:
    """
    Показать статус планировщика: когда последний раз выполнялись задачи.

    Returns:
        Время последнего сбора RSS, время последней сводки,
        общее количество заголовков в базе
    """
    storage = _get_storage()
    status = storage.get_scheduler_status()

    lines = ["Статус планировщика новостей:\n"]

    # Последний сбор RSS
    last_fetch = status.get("last_fetch")
    if last_fetch:
        ts = last_fetch.get("executed_at", "")[:19].replace("T", " ")
        stat = "✅" if last_fetch.get("status") == "success" else "❌"
        details = last_fetch.get("details", "")
        lines.append(f"  Последний сбор RSS: {ts} UTC  {stat}")
        if details:
            lines.append(f"    Детали: {details}")
    else:
        lines.append("  Последний сбор RSS: ещё не выполнялся")

    # Последняя сводка
    last_digest = status.get("last_digest")
    if last_digest:
        ts = last_digest.get("executed_at", "")[:19].replace("T", " ")
        stat = "✅" if last_digest.get("status") == "success" else "❌"
        details = last_digest.get("details", "")
        lines.append(f"  Последняя сводка:   {ts} UTC  {stat}")
        if details:
            lines.append(f"    Детали: {details}")
    else:
        lines.append("  Последняя сводка:   ещё не создана")

    # Общая статистика
    total = status.get("total_headlines", 0)
    today = date.today().isoformat()
    today_headlines = storage.get_headlines(date=today, limit=1000)
    lines.append(f"\n  Всего заголовков в базе: {total}")
    lines.append(f"  Заголовков за сегодня ({today}): {len(today_headlines)}")
    lines.append("\n  Расписание: сбор каждый час, сводка каждый день в 23:00")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Инструмент 5: Принудительный сбор
# ---------------------------------------------------------------------------

@mcp.tool()
def force_fetch_now() -> str:
    """
    Принудительно собрать новости прямо сейчас (не дожидаясь расписания).

    Returns:
        Количество новых заголовков и общее полученных
    """
    scheduler = _get_scheduler()
    if scheduler is None:
        # Если планировщик не инициализирован — выполняем сбор напрямую
        storage = _get_storage()
        try:
            from mcp_server.rss_parser import fetch_rss
            items = fetch_rss()
            new_count = storage.add_headlines(items)
            storage.log_task(
                "fetch_rss",
                "success",
                f"Получено {len(items)}, новых {new_count}",
            )
            return f"Собрано {len(items)} заголовков, из них новых: {new_count}"
        except Exception as exc:
            storage.log_task("fetch_rss", "error", str(exc))
            return f"Ошибка при сборе: {exc}"

    result = scheduler.force_fetch()
    return result


# ---------------------------------------------------------------------------
# Инструмент 6: Принудительная генерация сводки
# ---------------------------------------------------------------------------

@mcp.tool()
def force_digest_now() -> str:
    """
    Принудительно сгенерировать сводку дня прямо сейчас.

    Returns:
        Текст сгенерированной сводки или сообщение об ошибке
    """
    scheduler = _get_scheduler()
    if scheduler is None:
        return (
            "Планировщик не инициализирован. "
            "Запустите сервер через python -m mcp_server.news_server"
        )

    if scheduler.llm_fn is None:
        return (
            "LLM не настроен. Проверьте .env: "
            "QWEN_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY"
        )

    result = scheduler.force_digest()
    return result


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Определяем путь к БД
    db_path = Path("data/news.db")

    # Инициализация хранилища
    _storage = NewsStorage(db_path)

    # Создание LLM-функции (из .env)
    llm_fn = None
    try:
        from mcp_server.llm_client import create_llm_fn
        llm_fn = create_llm_fn()
    except ValueError as exc:
        # LLM не настроен — сервер запускается без генерации сводок
        print(
            f"Предупреждение: LLM не настроен ({exc}). "
            "Генерация сводок будет недоступна.",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"Предупреждение: не удалось инициализировать LLM: {exc}", file=sys.stderr)

    # Инициализация и запуск планировщика
    _scheduler = NewsScheduler(_storage, llm_fn=llm_fn)
    _scheduler.start()

    # Запуск MCP-сервера в stdio-режиме (основной поток, блокирующий)
    mcp.run()
