"""Вспомогательные функции для MCP-пайплайна новостей.

Образовательные концепции:
1. RSS-парсинг через xml.etree.ElementTree (stdlib)
2. HTTP-запросы через urllib (stdlib, инвариант проекта)
3. SQLite через sqlite3 (stdlib)
4. Telegram Bot API через urllib (stdlib)

Разделение ответственности:
- fetch_rss_by_categories — получение и группировка новостей
- save_summaries / get_summaries — персистентное хранение
- send_telegram_message — отправка уведомлений
- get_existing_links — дедупликация статей

Схема БД (таблица news_summaries):
  id, date, category, article_count, summary, telegram_sent, created_at
"""

from __future__ import annotations

import email.utils
import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT = 30

# Категории Lenta.ru: slug → человекочитаемое название
_CATEGORY_NAMES: dict[str, str] = {
    "russia":    "Россия",
    "world":     "Мир",
    "economics": "Экономика",
    "science":   "Наука и техника",
    "sport":     "Спорт",
    "culture":   "Культура",
}

# Эмодзи для Telegram-сообщений
_CATEGORY_EMOJI: dict[str, str] = {
    "russia":    "🇷🇺",
    "world":     "🌍",
    "economics": "📊",
    "science":   "🔬",
    "sport":     "⚽",
    "culture":   "🎭",
}

_CREATE_NEWS_SUMMARIES = """
CREATE TABLE IF NOT EXISTS news_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    category        TEXT NOT NULL,
    article_count   INTEGER,
    summary         TEXT NOT NULL,
    telegram_sent   INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    UNIQUE (date, category)
)
"""

_CREATE_PROCESSED_ARTICLES = """
CREATE TABLE IF NOT EXISTS processed_articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    link            TEXT UNIQUE NOT NULL,
    category        TEXT,
    processed_at    TEXT NOT NULL
)
"""


# ---------------------------------------------------------------------------
# SQLite: инициализация и путь к БД
# ---------------------------------------------------------------------------

def get_db_path() -> str:
    """Вернуть путь к SQLite-базе из .env или по умолчанию."""
    return os.environ.get("NEWS_DB_PATH", "data/news_pipeline.db")


def _has_news_summaries_unique_key(conn: sqlite3.Connection) -> bool:
    """Проверить наличие UNIQUE-ограничения/индекса на (date, category)."""
    for row in conn.execute("PRAGMA index_list('news_summaries')"):
        # row: seq, name, unique, origin, partial
        if len(row) < 3 or not row[2]:
            continue
        index_name = row[1]
        columns = [
            info[2]
            for info in conn.execute(f"PRAGMA index_info('{index_name}')")
        ]
        if columns == ["date", "category"]:
            return True
    return False


def _ensure_news_summaries_schema(conn: sqlite3.Connection) -> None:
    """Довести legacy-таблицу news_summaries до схемы с UNIQUE(date, category)."""
    if _has_news_summaries_unique_key(conn):
        return

    try:
        conn.execute(
            """
            CREATE UNIQUE INDEX uq_news_summaries_date_category
            ON news_summaries(date, category)
            """
        )
        return
    except sqlite3.IntegrityError:
        # В legacy-базе могли накопиться дубли по одной дате и категории.
        # Оставляем самую позднюю запись по id и затем добавляем уникальный индекс.
        conn.execute(
            """
            DELETE FROM news_summaries
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM news_summaries
                GROUP BY date, category
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX uq_news_summaries_date_category
            ON news_summaries(date, category)
            """
        )


def init_db(db_path: str | None = None) -> str:
    """Создать таблицы в SQLite. Вернуть путь к файлу БД.

    Args:
        db_path: Путь к файлу БД. Если None — из get_db_path().

    Returns:
        Абсолютный путь к файлу БД.
    """
    path = Path(db_path or get_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_CREATE_NEWS_SUMMARIES)
        conn.execute(_CREATE_PROCESSED_ARTICLES)
        _ensure_news_summaries_schema(conn)
        conn.commit()
    finally:
        conn.close()

    return str(path)


# ---------------------------------------------------------------------------
# SQLite: обработанные статьи (дедупликация)
# ---------------------------------------------------------------------------

def get_existing_links(db_path: str | None = None) -> set[str]:
    """Вернуть множество ссылок, уже обработанных ранее.

    Args:
        db_path: Путь к БД. Если None — get_db_path().

    Returns:
        set[str] — ссылки статей, которые уже суммаризировались.
    """
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT link FROM processed_articles").fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def mark_links_processed(
    links: list[str],
    category: str = "",
    db_path: str | None = None,
) -> None:
    """Пометить ссылки как обработанные для дедупликации.

    Args:
        links: Список URL статей.
        category: Категория статей.
        db_path: Путь к БД.
    """
    if not links:
        return
    path = init_db(db_path)
    processed_at = datetime.utcnow().isoformat()
    conn = sqlite3.connect(path)
    try:
        for link in links:
            conn.execute(
                "INSERT OR IGNORE INTO processed_articles (link, category, processed_at) VALUES (?, ?, ?)",
                (link, category, processed_at),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SQLite: сводки по категориям
# ---------------------------------------------------------------------------

def save_summaries(
    date: str,
    summaries: dict[str, str],
    article_counts: dict[str, int],
    telegram_sent: bool = False,
    db_path: str | None = None,
) -> int:
    """Сохранить суммаризации в news_summaries.

    Если запись для (date, category) уже существует — обновить.

    Args:
        date: Дата в формате YYYY-MM-DD.
        summaries: dict {category → текст суммаризации}.
        article_counts: dict {category → количество статей}.
        telegram_sent: Флаг отправки в Telegram.
        db_path: Путь к БД.

    Returns:
        Количество сохранённых категорий.
    """
    if not summaries:
        return 0

    path = init_db(db_path)
    created_at = datetime.utcnow().isoformat()
    sent_flag = 1 if telegram_sent else 0
    count = 0

    conn = sqlite3.connect(path)
    try:
        for category, summary_text in summaries.items():
            article_count = article_counts.get(category, 0)
            conn.execute(
                """
                INSERT INTO news_summaries
                    (date, category, article_count, summary, telegram_sent, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, category) DO UPDATE SET
                    article_count = excluded.article_count,
                    summary = excluded.summary,
                    telegram_sent = excluded.telegram_sent,
                    created_at = excluded.created_at
                """,
                (date, category, article_count, summary_text, sent_flag, created_at),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()

    return count


def mark_telegram_sent(date: str, db_path: str | None = None) -> None:
    """Поставить флаг telegram_sent=1 для всех записей за дату.

    Args:
        date: Дата в формате YYYY-MM-DD.
        db_path: Путь к БД.
    """
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE news_summaries SET telegram_sent=1 WHERE date=?",
            (date,),
        )
        conn.commit()
    finally:
        conn.close()


def get_summaries(
    date: str | None = None,
    limit: int = 10,
    db_path: str | None = None,
) -> list[dict]:
    """Получить суммаризации из БД.

    Args:
        date: Фильтр по дате (YYYY-MM-DD). Если None — последние записи.
        limit: Максимальное количество записей.
        db_path: Путь к БД.

    Returns:
        Список dict: id, date, category, article_count, summary, telegram_sent, created_at.
    """
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        if date:
            rows = conn.execute(
                """
                SELECT id, date, category, article_count, summary, telegram_sent, created_at
                FROM news_summaries
                WHERE date=?
                ORDER BY category
                LIMIT ?
                """,
                (date, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, date, category, article_count, summary, telegram_sent, created_at
                FROM news_summaries
                ORDER BY date DESC, category
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def get_pipeline_status(db_path: str | None = None) -> dict:
    """Вернуть статус пайплайна: последний запуск, количество суммаризаций.

    Returns:
        dict с ключами last_date, total_summaries, last_summaries.
    """
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        total_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM news_summaries"
        ).fetchone()
        last_row = conn.execute(
            "SELECT date, created_at FROM news_summaries ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        last_summaries = conn.execute(
            """
            SELECT category, article_count, telegram_sent, created_at
            FROM news_summaries
            ORDER BY created_at DESC
            LIMIT 10
            """
        ).fetchall()
    finally:
        conn.close()

    return {
        "total_summaries": total_row["cnt"] if total_row else 0,
        "last_date": last_row["date"] if last_row else None,
        "last_run_at": last_row["created_at"] if last_row else None,
        "last_summaries": [dict(r) for r in last_summaries],
    }


# ---------------------------------------------------------------------------
# RSS-парсинг Lenta.ru
# ---------------------------------------------------------------------------

def _parse_pub_date(pub_date_str: str) -> str:
    """Конвертировать RFC 2822 дату из RSS в ISO 8601.

    Args:
        pub_date_str: "Wed, 12 Mar 2026 10:30:00 +0300"

    Returns:
        "2026-03-12T10:30:00+03:00" или исходная строка при ошибке.
    """
    if not pub_date_str:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(pub_date_str)
        return dt.isoformat()
    except Exception:
        return pub_date_str


def _extract_category_from_url(link: str) -> str:
    """Определить категорию из URL статьи Lenta.ru.

    Примеры:
        https://lenta.ru/news/economics/2026/03/12/... → "economics"
        https://lenta.ru/sport/2026/03/12/...         → "sport"
        https://lenta.ru/news/russia/...               → "russia"

    Returns:
        Slug категории (строчные буквы) или "" если не определена.
    """
    try:
        parsed = urllib.parse.urlparse(link)
        parts = [p for p in parsed.path.split("/") if p]
        # Пробуем найти известную категорию в сегментах пути
        for part in parts:
            if part in _CATEGORY_NAMES:
                return part
        # Lenta.ru: /news/<category>/...
        if len(parts) >= 2 and parts[0] == "news":
            return parts[1]
    except Exception:
        pass
    return ""


def _normalize_category(raw: str) -> str:
    """Нормализовать сырое значение категории к известному slug.

    Lenta.ru возвращает категории на русском, например «Экономика».
    Нам нужен slug для группировки.

    Args:
        raw: Значение из тега <category> или URL.

    Returns:
        Slug категории или нормализованный lowercase вариант.
    """
    if not raw:
        return "other"
    raw_lower = raw.strip().lower()

    # Прямое совпадение со slug
    if raw_lower in _CATEGORY_NAMES:
        return raw_lower

    # Русские названия → slug
    ru_to_slug = {
        "россия":         "russia",
        "мир":            "world",
        "экономика":      "economics",
        "наука":          "science",
        "наука и техника":"science",
        "спорт":          "sport",
        "культура":       "culture",
    }
    for ru_name, slug in ru_to_slug.items():
        if ru_name in raw_lower:
            return slug

    # Английские полные названия
    en_to_slug = {
        "russia":    "russia",
        "world":     "world",
        "economics": "economics",
        "science":   "science",
        "sport":     "sport",
        "sports":    "sport",
        "culture":   "culture",
    }
    for en_name, slug in en_to_slug.items():
        if en_name in raw_lower:
            return slug

    # Fallback: используем первое слово
    return raw_lower.split()[0] if raw_lower else "other"


def fetch_rss_by_categories(
    feed_url: str,
    max_per_category: int,
    categories: list[str],
    existing_links: set[str] | None = None,
) -> dict:
    """Загрузить RSS и сгруппировать статьи по категориям.

    Args:
        feed_url: URL RSS-фида.
        max_per_category: Максимум статей на категорию (0 = без ограничений).
        categories: Список категорий для фильтрации (пустой = все).
        existing_links: Ссылки уже обработанных статей (для дедупликации).

    Returns:
        dict {category → [{"title": ..., "description": ..., "link": ..., "pubDate": ...}, ...]}

    Raises:
        ConnectionError: Если не удалось загрузить фид.
        ValueError: Если XML невалиден.
    """
    # Загрузка XML
    try:
        req = urllib.request.Request(
            feed_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; news-pipeline-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Не удалось загрузить RSS {feed_url}: {exc}") from exc

    # Парсинг XML
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Ошибка парсинга RSS XML: {exc}") from exc

    channel = root.find("channel")
    elements = channel.findall("item") if channel is not None else root.findall("item")

    # Категории для фильтрации (slug → True)
    filter_cats = set(categories) if categories else set()
    # Ссылки уже обработанных статей
    seen = existing_links or set()

    # Группировка по категориям
    grouped: dict[str, list[dict]] = {}

    for elem in elements:
        title = (elem.findtext("title") or "").strip()
        link = (elem.findtext("link") or "").strip()
        description = (elem.findtext("description") or "").strip()
        pub_date_raw = (elem.findtext("pubDate") or "").strip()
        raw_category = (elem.findtext("category") or "").strip()

        if not title:
            continue

        # Дедупликация
        if link and link in seen:
            continue

        # Определяем категорию
        if raw_category:
            category = _normalize_category(raw_category)
        elif link:
            category = _extract_category_from_url(link)
        else:
            category = "other"

        # Фильтр по категориям
        if filter_cats and category not in filter_cats:
            continue

        pub_date = _parse_pub_date(pub_date_raw) if pub_date_raw else ""

        item = {
            "title": title,
            "description": description,
            "link": link,
            "pubDate": pub_date,
        }

        if category not in grouped:
            grouped[category] = []

        # Ограничение max_per_category
        if max_per_category > 0 and len(grouped[category]) >= max_per_category:
            continue

        grouped[category].append(item)

    return grouped


# ---------------------------------------------------------------------------
# Telegram Bot API
# ---------------------------------------------------------------------------

def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Разбить длинное сообщение на части по max_len символов.

    Разбивает по абзацам (двойной перенос) или по одиночному переносу.

    Args:
        text: Исходный текст.
        max_len: Максимальная длина части.

    Returns:
        Список частей.
    """
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    current = ""

    for line in text.split("\n"):
        candidate = (current + "\n" + line).lstrip("\n") if current else line
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                parts.append(current)
            # Если одна строка больше лимита — режем принудительно
            while len(line) > max_len:
                parts.append(line[:max_len])
                line = line[max_len:]
            current = line

    if current:
        parts.append(current)

    return parts or [text[:max_len]]


def send_telegram_message(
    text: str,
    token: str,
    chat_id: str,
) -> bool:
    """Отправить сообщение в Telegram через Bot API.

    Длинные сообщения автоматически разбиваются на части (лимит 4096 символов).
    Использует stdlib urllib (инвариант проекта).

    Args:
        text: Текст сообщения (Markdown).
        token: Telegram Bot Token (из @BotFather).
        chat_id: ID чата или @username канала.

    Returns:
        True — успешно отправлено, False — ошибка.
    """
    if not token or not chat_id:
        return False

    parts = _split_message(text, max_len=4000)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    success = True

    for part in parts:
        payload = {
            "chat_id": chat_id,
            "text": part,
            "parse_mode": "Markdown",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                if not resp_data.get("ok"):
                    success = False
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            # parse_mode=Markdown может вызвать ошибку из-за спецсимволов
            # Пробуем отправить без форматирования
            try:
                plain_payload = {
                    "chat_id": chat_id,
                    "text": part,
                }
                plain_data = json.dumps(plain_payload).encode("utf-8")
                plain_req = urllib.request.Request(
                    url,
                    data=plain_data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(plain_req, timeout=_HTTP_TIMEOUT):
                    pass
            except Exception:
                success = False
        except Exception:
            success = False

    return success


# ---------------------------------------------------------------------------
# Telegram: получить свой chat_id
# ---------------------------------------------------------------------------

def get_my_chat_id(token: str) -> list[dict]:
    """Получить chat_id(ы) из последних входящих сообщений бота.

    Образовательная концепция: Telegram Bot API → getUpdates.
    Используется один раз при настройке — чтобы узнать свой числовой chat_id
    и записать его в .env как TELEGRAM_CHAT_ID.

    Порядок действий:
        1. Создать бота у @BotFather → получить токен
        2. Написать боту /start в личных сообщениях
        3. Вызвать get_my_chat_id(token) → получить свой chat_id
        4. Записать в .env: TELEGRAM_CHAT_ID=<chat_id>

    Args:
        token: Telegram Bot Token.

    Returns:
        Список dict: [{"chat_id": 123456789, "username": "ivan", "first_name": "Ivan", "type": "private"}, ...]
        Пустой список если ни одного сообщения нет.

    Raises:
        RuntimeError: Если токен неверный или API недоступен.
    """
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ошибка Telegram API {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ошибка сети при вызове Telegram: {exc}") from exc

    if not data.get("ok"):
        raise RuntimeError(f"Telegram API вернул ошибку: {data}")

    results = data.get("result", [])
    seen_chats: set[int] = set()
    chats: list[dict] = []

    for update in results:
        # Ищем chat во всех возможных полях
        msg = (
            update.get("message")
            or update.get("channel_post")
            or update.get("edited_message")
            or update.get("callback_query", {}).get("message")
        )
        if not msg:
            continue
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        if chat_id and chat_id not in seen_chats:
            seen_chats.add(chat_id)
            chats.append({
                "chat_id": chat_id,
                "username": chat.get("username", ""),
                "first_name": chat.get("first_name", ""),
                "last_name": chat.get("last_name", ""),
                "type": chat.get("type", ""),  # private / group / channel
            })

    return chats


# ---------------------------------------------------------------------------
# Формирование Telegram-сообщения
# ---------------------------------------------------------------------------

def format_telegram_message(date: str, summaries: dict[str, str]) -> str:
    """Сформировать Telegram-сообщение из суммаризаций.

    Формат:
        📰 Новости за 12.03.2026

        📊 Экономика:
        {summary}

        🌍 Мир:
        {summary}
        ...

    Args:
        date: Дата в формате YYYY-MM-DD.
        summaries: dict {category_slug → текст суммаризации}.

    Returns:
        Отформатированный текст.
    """
    try:
        d = datetime.strptime(date, "%Y-%m-%d")
        date_formatted = d.strftime("%d.%m.%Y")
    except ValueError:
        date_formatted = date

    lines = [f"📰 *Новости за {date_formatted}*\n"]

    # Сортируем категории в предсказуемом порядке
    category_order = ["russia", "world", "economics", "science", "sport", "culture"]
    sorted_cats = sorted(
        summaries.keys(),
        key=lambda c: category_order.index(c) if c in category_order else 999,
    )

    for cat in sorted_cats:
        summary = summaries[cat]
        emoji = _CATEGORY_EMOJI.get(cat, "📌")
        name = _CATEGORY_NAMES.get(cat, cat.capitalize())
        lines.append(f"{emoji} *{name}:*")
        lines.append(summary)
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Утилита настройки: python -m mcp_server.news_api
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Утилита для получения Telegram chat_id.

    Запуск:
        python -m mcp_server.news_api

    Выводит chat_id всех пользователей/каналов, которые писали боту.
    После запуска запишите нужный chat_id в .env:
        TELEGRAM_CHAT_ID=123456789
    """
    import sys

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN не задан в .env")
        print("   Создайте бота у @BotFather и добавьте токен в .env")
        sys.exit(1)

    print(f"🤖 Токен: {token[:10]}...{token[-4:]}")
    print("   Проверяю входящие сообщения бота (getUpdates)...")
    print()

    try:
        chats = get_my_chat_id(token)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if not chats:
        print("⚠️  Нет входящих сообщений.")
        print()
        print("   Что нужно сделать:")
        print("   1. Найдите своего бота в Telegram")
        print("   2. Нажмите START или напишите /start")
        print("   3. Запустите этот скрипт снова")
        sys.exit(0)

    print(f"✅ Найдено чатов: {len(chats)}")
    print()
    for ch in chats:
        chat_type = ch["type"]
        name_parts = [ch.get("first_name", ""), ch.get("last_name", "")]
        name = " ".join(p for p in name_parts if p)
        username = f"@{ch['username']}" if ch.get("username") else ""
        label = f"{name} {username}".strip() or f"[{chat_type}]"
        print(f"   {'👤' if chat_type == 'private' else '👥'} {label}")
        print(f"      chat_id: {ch['chat_id']}")
        print(f"      type:    {chat_type}")
        print()

    print("─" * 50)
    print("Добавьте в .env:")
    if chats:
        first = chats[0]
        print(f"   TELEGRAM_CHAT_ID={first['chat_id']}")
    print()
    print("Проверить отправку:")
    print("   python demo_news_pipeline.py")
