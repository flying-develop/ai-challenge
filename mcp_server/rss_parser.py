"""RSS-парсер для фида РИА Новостей.

Образовательная концепция: парсинг внешнего источника данных с помощью stdlib.
- urllib.request: HTTP-запрос (без внешних зависимостей)
- xml.etree.ElementTree: разбор XML
- email.utils.parsedate_to_datetime: парсинг RFC 2822 дат из RSS

Формат даты в RSS: "Sun, 01 Mar 2026 05:45:11 +0300"
Конвертируем в ISO 8601: "2026-03-01T05:45:11+03:00"

RSS-фид РИА содержит ТОЛЬКО заголовки (без description).
Это нормально — для LLM-сводки заголовков достаточно.
"""

from __future__ import annotations

import email.utils
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timezone

# URL RSS-фида РИА Новостей (архив, обновляется каждые несколько минут)
_DEFAULT_RSS_URL = "https://ria.ru/export/rss2/archive/index.xml"

# Таймаут HTTP-запроса (секунды)
_HTTP_TIMEOUT = 30


def _parse_pub_date(pub_date_str: str) -> str:
    """Конвертировать RFC 2822 дату из RSS в ISO 8601.

    Args:
        pub_date_str: Строка вида "Sun, 01 Mar 2026 05:45:11 +0300"

    Returns:
        ISO 8601 строка вида "2026-03-01T05:45:11+03:00"
        При ошибке парсинга возвращает исходную строку.
    """
    if not pub_date_str:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(pub_date_str)
        # Конвертируем в aware datetime с явным UTC offset
        return dt.isoformat()
    except Exception:
        # Fallback: вернуть как есть, не ломать обработку
        return pub_date_str


def _fetch_rss_bytes(url: str) -> bytes:
    """Загрузить RSS-фид по URL, вернуть сырые байты.

    Raises:
        ConnectionError: если не удалось подключиться
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                # Корректный User-Agent: некоторые серверы блокируют Python/urllib
                "User-Agent": "Mozilla/5.0 (compatible; news-digest-bot/1.0)"
            },
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Ошибка: не удалось загрузить RSS-фид {url}: {exc}"
        ) from exc
    except OSError as exc:
        raise ConnectionError(
            f"Ошибка: не удалось загрузить RSS-фид {url}: {exc}"
        ) from exc


def parse_rss_from_bytes(content: bytes) -> list[dict]:
    """Разобрать RSS XML из байтов.

    Используется для тестирования (можно передать mock-данные).

    Args:
        content: Байты RSS XML (UTF-8 или другая кодировка)

    Returns:
        Список dict: {guid, title, link, pub_date, category}

    Raises:
        ValueError: если XML не валидный
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Ошибка парсинга RSS XML: {exc}") from exc

    items: list[dict] = []

    # RSS 2.0: /rss/channel/item
    channel = root.find("channel")
    if channel is None:
        # Иногда корень сам является channel (атом или нестандартный RSS)
        elements = root.findall("item")
    else:
        elements = channel.findall("item")

    for elem in elements:
        title = (elem.findtext("title") or "").strip()
        link = (elem.findtext("link") or "").strip()
        guid = (elem.findtext("guid") or link).strip()
        pub_date_raw = (elem.findtext("pubDate") or "").strip()
        category = (elem.findtext("category") or "").strip()

        # Пропускаем элементы без заголовка или guid
        if not title or not guid:
            continue

        pub_date = _parse_pub_date(pub_date_raw) if pub_date_raw else ""

        items.append({
            "guid": guid,
            "title": title,
            "link": link,
            "pub_date": pub_date,
            "category": category,
        })

    return items


def fetch_rss(url: str = _DEFAULT_RSS_URL) -> list[dict]:
    """Загрузить и разобрать RSS-фид РИА Новостей.

    Образовательная концепция: две ответственности разделены:
    - _fetch_rss_bytes: сетевой IO (сложно тестировать)
    - parse_rss_from_bytes: чистая функция (легко тестировать с mock-данными)

    Args:
        url: URL RSS-фида (по умолчанию — РИА Новости)

    Returns:
        Список dict: {guid, title, link, pub_date, category}
        Поля pub_date в ISO 8601, category может быть пустой строкой.

    Raises:
        ConnectionError: если не удалось загрузить фид
        ValueError: если фид содержит невалидный XML
    """
    content = _fetch_rss_bytes(url)
    return parse_rss_from_bytes(content)
