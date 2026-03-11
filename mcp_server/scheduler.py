"""Фоновый планировщик задач для MCP-сервера новостей.

Образовательная концепция: MCP-сервер может не только отвечать на запросы,
но и выполнять работу проактивно — собирать данные, агрегировать,
готовить результаты заранее.

Архитектура:
- Планировщик работает в ОТДЕЛЬНОМ ПОТОКЕ (threading.Thread, daemon=True)
- MCP-сервер (FastMCP) работает в основном потоке
- daemon=True: поток останавливается автоматически при завершении основного процесса
- schedule: минималистичная библиотека для периодических задач

Задачи:
1. fetch_rss — каждый час: скачать RSS, сохранить новые заголовки
2. make_digest — каждый день в 23:00: сгенерировать LLM-сводку дня
"""

from __future__ import annotations

import threading
import time
from datetime import date
from typing import Callable

try:
    import schedule
    _SCHEDULE_AVAILABLE = True
except ImportError:
    _SCHEDULE_AVAILABLE = False

from mcp_server.rss_parser import fetch_rss
from mcp_server.news_storage import NewsStorage


class NewsScheduler:
    """Фоновый планировщик задач.

    Образовательный паттерн: планировщик получает storage и llm_fn
    через конструктор (Dependency Injection). Это позволяет:
    - тестировать без реального LLM (передать mock)
    - тестировать без сети (mock fetch_rss)
    - использовать любой LLM-провайдер
    """

    def __init__(
        self,
        storage: NewsStorage,
        llm_fn: Callable[[str], str] | None = None,
        rss_url: str | None = None,
    ) -> None:
        """
        Args:
            storage: Хранилище новостей (SQLite)
            llm_fn: Функция для генерации сводки (prompt: str) -> str.
                    Если None — генерация сводок отключена.
            rss_url: URL RSS-фида (по умолчанию — РИА Новостей)
        """
        self.storage = storage
        self.llm_fn = llm_fn
        self.rss_url = rss_url  # None → fetch_rss использует default URL
        self._running = False
        self._thread: threading.Thread | None = None
        # Защита от одновременного запуска нескольких сборов/сводок
        self._fetch_lock = threading.Lock()
        self._digest_lock = threading.Lock()

    def start(self) -> None:
        """Запустить планировщик в фоновом потоке.

        Первый сбор RSS выполняется немедленно при старте.
        Далее — по расписанию в фоновом потоке.
        """
        if self._running:
            return

        if _SCHEDULE_AVAILABLE:
            # Используем библиотеку schedule
            schedule.every(1).hours.do(self._fetch_task)
            schedule.every().day.at("23:00").do(self._digest_task)
        # Если schedule не установлен — задачи только по принудительному вызову

        # Первый сбор — сразу при старте (до запуска потока, синхронно)
        self._fetch_task()

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Остановить планировщик."""
        self._running = False

    def _run_loop(self) -> None:
        """Основной цикл планировщика (в фоновом потоке).

        Каждые 30 секунд проверяет, нужно ли выполнить какую-то задачу.
        Небольшой интервал проверки = быстрая реакция на расписание.
        """
        while self._running:
            if _SCHEDULE_AVAILABLE:
                schedule.run_pending()
            time.sleep(30)

    def _fetch_task(self) -> None:
        """Задача: собрать новые заголовки из RSS-фида."""
        if not self._fetch_lock.acquire(blocking=False):
            # Уже выполняется — пропускаем
            return
        try:
            kwargs = {}
            if self.rss_url:
                kwargs["url"] = self.rss_url
            items = fetch_rss(**kwargs)
            new_count = self.storage.add_headlines(items)
            self.storage.log_task(
                "fetch_rss",
                "success",
                f"Получено {len(items)}, новых {new_count}",
            )
        except Exception as exc:
            self.storage.log_task("fetch_rss", "error", str(exc))
        finally:
            self._fetch_lock.release()

    def _digest_task(self) -> None:
        """Задача: сгенерировать дневную сводку через LLM."""
        if self.llm_fn is None:
            self.storage.log_task(
                "make_digest", "error", "LLM не настроен (llm_fn=None)"
            )
            return

        if not self._digest_lock.acquire(blocking=False):
            return
        try:
            today = date.today().isoformat()
            headlines = self.storage.get_headlines(date=today, limit=500)

            if not headlines:
                self.storage.log_task(
                    "make_digest", "success", "Нет новостей за день"
                )
                return

            # Формируем prompt для LLM
            titles = "\n".join(f"- {h['title']}" for h in headlines)
            prompt = (
                f"Вот заголовки новостей за {today} ({len(headlines)} шт.):\n\n"
                f"{titles}\n\n"
                "Составь краткую сводку дня (5-7 предложений): "
                "выдели главные темы, ключевые события, общий тон новостей."
            )

            digest_text = self.llm_fn(prompt)
            self.storage.save_digest(today, digest_text, len(headlines))
            self.storage.log_task(
                "make_digest",
                "success",
                f"{len(headlines)} заголовков → сводка создана",
            )
        except Exception as exc:
            self.storage.log_task("make_digest", "error", str(exc))
        finally:
            self._digest_lock.release()

    def force_fetch(self) -> str:
        """Принудительный сбор (для MCP-инструмента force_fetch_now).

        Returns:
            Строка с результатом: "Получено N, новых M"
        """
        self._fetch_task()
        # Читаем результат из лога
        status = self.storage.get_scheduler_status()
        last = status.get("last_fetch")
        if last:
            return last.get("details", "Сбор выполнен")
        return "Сбор выполнен"

    def force_digest(self) -> str:
        """Принудительная генерация сводки (для MCP-инструмента force_digest_now).

        Returns:
            Текст сгенерированной сводки или сообщение об ошибке
        """
        self._digest_task()
        # Читаем результат из лога
        status = self.storage.get_scheduler_status()
        last = status.get("last_digest")
        if last:
            if last.get("status") == "error":
                return f"Ошибка генерации сводки: {last.get('details', '')}"
            # Возвращаем свежесозданную сводку
            today = date.today().isoformat()
            digest = self.storage.get_digest(today)
            if digest:
                return digest["digest_text"]
            return last.get("details", "Сводка создана")
        return "Сводка создана"
