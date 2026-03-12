"""Тесты для News Digest MCP-сервера.

Покрывают:
- парсинг RSS (mock XML-строка)
- дедупликация (повторная вставка)
- форматирование заголовков
- хранение и получение сводки
- scheduler_log записи
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mcp_server.rss_parser import parse_rss_from_bytes, _parse_pub_date
from mcp_server.news_storage import NewsStorage


# ---------------------------------------------------------------------------
# Тесты rss_parser
# ---------------------------------------------------------------------------

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>RIA Novosti</title>
    <item>
      <title>Pervaya novost dnya</title>
      <link>https://ria.ru/20260311/article1.html</link>
      <guid>https://ria.ru/20260311/article1.html</guid>
      <pubDate>Wed, 11 Mar 2026 10:00:00 +0300</pubDate>
      <category>Lenta novostej</category>
    </item>
    <item>
      <title>Vtoraya novost dnya</title>
      <link>https://ria.ru/20260311/article2.html</link>
      <guid>https://ria.ru/20260311/article2.html</guid>
      <pubDate>Wed, 11 Mar 2026 09:30:00 +0300</pubDate>
      <category>Ekonomika</category>
    </item>
    <item>
      <title>Novost bez kategorii</title>
      <link>https://ria.ru/20260311/article3.html</link>
      <guid>https://ria.ru/20260311/article3.html</guid>
      <pubDate>Wed, 11 Mar 2026 08:00:00 +0300</pubDate>
    </item>
  </channel>
</rss>
""".encode("utf-8")

SAMPLE_RSS_NO_CHANNEL = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <item>
    <title>Only one item</title>
    <link>https://example.com/item1</link>
    <guid>https://example.com/item1</guid>
    <pubDate>Wed, 11 Mar 2026 10:00:00 +0000</pubDate>
  </item>
</rss>
""".encode("utf-8")

SAMPLE_RSS_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty feed</title>
  </channel>
</rss>
""".encode("utf-8")


class TestRssParser:
    """Тесты парсера RSS."""

    def test_parse_standard_feed(self):
        """Parsing standard RSS feed."""
        items = parse_rss_from_bytes(SAMPLE_RSS)
        assert len(items) == 3

    def test_parse_titles(self):
        """Titles are correctly extracted."""
        items = parse_rss_from_bytes(SAMPLE_RSS)
        titles = [i["title"] for i in items]
        assert "Pervaya novost dnya" in titles
        assert "Vtoraya novost dnya" in titles
        assert "Novost bez kategorii" in titles

    def test_parse_guids(self):
        """GUID (URL) is correctly extracted."""
        items = parse_rss_from_bytes(SAMPLE_RSS)
        guids = {i["guid"] for i in items}
        assert "https://ria.ru/20260311/article1.html" in guids

    def test_parse_category(self):
        """Categories are correctly extracted."""
        items = parse_rss_from_bytes(SAMPLE_RSS)
        categories = {i["title"]: i["category"] for i in items}
        assert categories["Pervaya novost dnya"] == "Lenta novostej"
        assert categories["Vtoraya novost dnya"] == "Ekonomika"
        # Element without category → empty string
        assert categories["Novost bez kategorii"] == ""

    def test_parse_pub_date_iso8601(self):
        """pubDate is converted to ISO 8601."""
        items = parse_rss_from_bytes(SAMPLE_RSS)
        first = next(i for i in items if i["title"] == "Pervaya novost dnya")
        # Result should be ISO 8601
        assert "T" in first["pub_date"], f"Expected ISO 8601, got: {first['pub_date']}"
        assert "2026" in first["pub_date"]

    def test_parse_empty_feed(self):
        """Пустой фид возвращает пустой список."""
        items = parse_rss_from_bytes(SAMPLE_RSS_EMPTY)
        assert items == []

    def test_parse_feed_without_channel(self):
        """Фид без <channel> элемента (нестандартный)."""
        # Может вернуть элементы или пустой список — главное не упасть
        items = parse_rss_from_bytes(SAMPLE_RSS_NO_CHANNEL)
        # Структура без channel не имеет items в корне в стандарте,
        # но парсер не должен крашиться
        assert isinstance(items, list)

    def test_parse_invalid_xml_raises(self):
        """Невалидный XML вызывает ValueError."""
        with pytest.raises(ValueError, match="Ошибка парсинга"):
            parse_rss_from_bytes(b"<not>valid</xml>")

    def test_parse_pub_date_helper(self):
        """_parse_pub_date конвертирует RFC 2822 в ISO 8601."""
        result = _parse_pub_date("Wed, 11 Mar 2026 10:00:00 +0300")
        assert "2026" in result
        assert "T" in result

    def test_parse_pub_date_empty(self):
        """_parse_pub_date с пустой строкой возвращает пустую строку."""
        assert _parse_pub_date("") == ""

    def test_parse_pub_date_invalid_fallback(self):
        """_parse_pub_date с невалидной строкой возвращает исходную строку."""
        result = _parse_pub_date("not a date at all")
        assert result == "not a date at all"


# ---------------------------------------------------------------------------
# Тесты NewsStorage
# ---------------------------------------------------------------------------

@pytest.fixture
def storage(tmp_path: Path) -> NewsStorage:
    """Создать временное хранилище для каждого теста."""
    db_path = tmp_path / "test_news.db"
    return NewsStorage(db_path)


SAMPLE_ITEMS = [
    {
        "guid": "https://ria.ru/article1",
        "title": "Новость первая",
        "link": "https://ria.ru/article1",
        "pub_date": "2026-03-11T10:00:00+03:00",
        "category": "Политика",
    },
    {
        "guid": "https://ria.ru/article2",
        "title": "Новость вторая",
        "link": "https://ria.ru/article2",
        "pub_date": "2026-03-11T09:00:00+03:00",
        "category": "Экономика",
    },
]


class TestNewsStorage:
    """Тесты SQLite-хранилища."""

    def test_add_headlines_returns_count(self, storage: NewsStorage):
        """add_headlines возвращает количество НОВЫХ записей."""
        count = storage.add_headlines(SAMPLE_ITEMS)
        assert count == 2

    def test_deduplication_on_second_insert(self, storage: NewsStorage):
        """Повторная вставка тех же guid не создаёт дублей."""
        first = storage.add_headlines(SAMPLE_ITEMS)
        second = storage.add_headlines(SAMPLE_ITEMS)
        assert first == 2
        assert second == 0  # Все уже существуют

    def test_partial_deduplication(self, storage: NewsStorage):
        """Частичная дедупликация: часть новых, часть дублей."""
        storage.add_headlines(SAMPLE_ITEMS[:1])  # добавляем первую
        new_count = storage.add_headlines(SAMPLE_ITEMS)  # обе — вторая новая
        assert new_count == 1

    def test_get_headlines_returns_all(self, storage: NewsStorage):
        """get_headlines возвращает все добавленные заголовки."""
        storage.add_headlines(SAMPLE_ITEMS)
        headlines = storage.get_headlines(limit=10)
        assert len(headlines) == 2

    def test_get_headlines_by_date(self, storage: NewsStorage):
        """get_headlines фильтрует по дате."""
        storage.add_headlines(SAMPLE_ITEMS)
        headlines = storage.get_headlines(date="2026-03-11")
        assert len(headlines) == 2

    def test_get_headlines_wrong_date(self, storage: NewsStorage):
        """get_headlines с несуществующей датой возвращает пустой список."""
        storage.add_headlines(SAMPLE_ITEMS)
        headlines = storage.get_headlines(date="2020-01-01")
        assert headlines == []

    def test_get_headlines_limit(self, storage: NewsStorage):
        """get_headlines уважает лимит."""
        storage.add_headlines(SAMPLE_ITEMS)
        headlines = storage.get_headlines(limit=1)
        assert len(headlines) == 1

    def test_get_headlines_empty(self, storage: NewsStorage):
        """get_headlines на пустой базе возвращает пустой список."""
        assert storage.get_headlines() == []

    def test_add_empty_list(self, storage: NewsStorage):
        """add_headlines с пустым списком возвращает 0."""
        assert storage.add_headlines([]) == 0

    def test_count_headlines(self, storage: NewsStorage):
        """count_headlines возвращает общее количество."""
        assert storage.count_headlines() == 0
        storage.add_headlines(SAMPLE_ITEMS)
        assert storage.count_headlines() == 2

    def test_save_and_get_digest(self, storage: NewsStorage):
        """Сохранённая сводка доступна через get_digest."""
        storage.save_digest("2026-03-11", "Краткая сводка дня.", 10)
        digest = storage.get_digest("2026-03-11")
        assert digest is not None
        assert digest["date"] == "2026-03-11"
        assert digest["digest_text"] == "Краткая сводка дня."
        assert digest["headline_count"] == 10

    def test_get_digest_not_found(self, storage: NewsStorage):
        """get_digest возвращает None для несуществующей даты."""
        assert storage.get_digest("2020-01-01") is None

    def test_get_latest_digest(self, storage: NewsStorage):
        """get_latest_digest возвращает самую последнюю сводку."""
        storage.save_digest("2026-03-10", "Сводка 10 марта.", 5)
        storage.save_digest("2026-03-11", "Сводка 11 марта.", 10)
        latest = storage.get_latest_digest()
        assert latest is not None
        assert latest["date"] == "2026-03-11"

    def test_get_latest_digest_empty(self, storage: NewsStorage):
        """get_latest_digest на пустой базе возвращает None."""
        assert storage.get_latest_digest() is None

    def test_save_digest_upsert(self, storage: NewsStorage):
        """Повторный save_digest обновляет существующую запись."""
        storage.save_digest("2026-03-11", "Первая версия.", 5)
        storage.save_digest("2026-03-11", "Обновлённая версия.", 15)
        digest = storage.get_digest("2026-03-11")
        assert digest["digest_text"] == "Обновлённая версия."
        assert digest["headline_count"] == 15

    def test_log_task_success(self, storage: NewsStorage):
        """log_task записывает успешное выполнение."""
        storage.log_task("fetch_rss", "success", "Получено 45, новых 10")
        status = storage.get_scheduler_status()
        last_fetch = status["last_fetch"]
        assert last_fetch is not None
        assert last_fetch["status"] == "success"
        assert "45" in last_fetch["details"]

    def test_log_task_error(self, storage: NewsStorage):
        """log_task записывает ошибку."""
        storage.log_task("make_digest", "error", "ConnectionError: timeout")
        status = storage.get_scheduler_status()
        last_digest = status["last_digest"]
        assert last_digest is not None
        assert last_digest["status"] == "error"

    def test_scheduler_status_empty(self, storage: NewsStorage):
        """get_scheduler_status на пустой базе возвращает None для задач."""
        status = storage.get_scheduler_status()
        assert status["last_fetch"] is None
        assert status["last_digest"] is None
        assert status["total_headlines"] == 0

    def test_scheduler_status_with_data(self, storage: NewsStorage):
        """get_scheduler_status возвращает актуальные данные."""
        storage.add_headlines(SAMPLE_ITEMS)
        storage.log_task("fetch_rss", "success", "Получено 2, новых 2")
        storage.log_task("make_digest", "success", "2 заголовка → сводка")

        status = storage.get_scheduler_status()
        assert status["total_headlines"] == 2
        assert status["last_fetch"]["status"] == "success"
        assert status["last_digest"]["status"] == "success"

    def test_headline_fields_present(self, storage: NewsStorage):
        """Поля заголовка корректно сохраняются и возвращаются."""
        storage.add_headlines(SAMPLE_ITEMS[:1])
        headlines = storage.get_headlines()
        h = headlines[0]
        assert h["title"] == "Новость первая"
        assert h["guid"] == "https://ria.ru/article1"
        assert h["link"] == "https://ria.ru/article1"
        assert h["pub_date"] == "2026-03-11T10:00:00+03:00"
        assert h["category"] == "Политика"
        assert "fetched_at" in h

    def test_persistence_across_instances(self, tmp_path: Path):
        """Данные сохраняются между созданиями нового экземпляра NewsStorage."""
        db_path = tmp_path / "persist_test.db"
        s1 = NewsStorage(db_path)
        s1.add_headlines(SAMPLE_ITEMS)
        s1.save_digest("2026-03-11", "Тестовая сводка.", 2)

        # Создаём новый экземпляр — должны увидеть данные
        s2 = NewsStorage(db_path)
        assert s2.count_headlines() == 2
        digest = s2.get_digest("2026-03-11")
        assert digest is not None
        assert digest["digest_text"] == "Тестовая сводка."
