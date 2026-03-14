"""Двухпроходный исследовательский оркестратор.

Образовательная концепция: оркестратор координирует несколько
MCP-серверов, каждый из которых отвечает за свою часть работы.
Агент (LLM) принимает решения на ключевых этапах:
  - какие поисковые запросы сформировать
  - какие сущности требуют доп. исследования
  - как синтезировать финальный ответ

Схема флоу:
  TASK_RECEIVED → SEARCH_INITIAL → FETCH_INITIAL → SUMMARIZE_INITIAL
  → SEARCH_DEEP → FETCH_DEEP → SYNTHESIZE_FINAL → DELIVERED
  Из любого состояния → FAILED (при критической ошибке)

Использование:
    from orchestrator import ResearchOrchestrator, ResearchContext
    from mcp_client.client import MCPClient
    from mcp_server.llm_client import create_llm_fn

    orchestrator = ResearchOrchestrator(
        mcp_clients={
            "search": search_client,
            "scraper": scraper_client,
            "telegram": telegram_client,
            "journal": journal_client,
        },
        llm_fn=create_llm_fn(),
    )
    orchestrator.run("Хочу сходить в кино в эти выходные", chat_id="@flying_dev")
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable

from orchestrator.research_context import ResearchContext
from orchestrator.research_states import ResearchState, VALID_TRANSITIONS


# ---------------------------------------------------------------------------
# ResearchOrchestrator
# ---------------------------------------------------------------------------

class ResearchOrchestrator:
    """
    Двухпроходный исследовательский оркестратор.

    Принимает MCPClient для каждого из 4 серверов и callable-LLM.
    Выполняет полный цикл исследования с уведомлением в Telegram
    и аудит-логом в SQLite (через journal_server).

    Инварианты (из config/invariants/research.md):
      - Не более 10 ссылок на один поисковый раунд
      - Ровно два поисковых раунда
      - Ссылки не дублируются между раундами
      - Финальный ответ нельзя формировать без первой суммаризации
    """

    # Жёсткий лимит на кол-во ссылок за раунд (инвариант)
    MAX_LINKS_PER_ROUND = 10

    def __init__(
        self,
        mcp_clients: dict,
        llm_fn: Callable[[str], str],
        verbose: bool = True,
    ) -> None:
        """
        Args:
            mcp_clients: Словарь {"search": MCPClient, "scraper": ...,
                                   "telegram": ..., "journal": ...}
            llm_fn:      Callable (prompt: str) -> str — LLM-функция
            verbose:     Печатать прогресс в stdout (для отладки)
        """
        self.search = mcp_clients["search"]
        self.scraper = mcp_clients["scraper"]
        self.telegram = mcp_clients["telegram"]
        self.journal = mcp_clients["journal"]
        self.llm = llm_fn
        self.verbose = verbose

        self.state = ResearchState.TASK_RECEIVED
        self.context = ResearchContext()

        # Счётчики для проверки инвариантов
        self._telegram_calls = 0
        self._journal_calls = 0
        self._search_calls: dict[str, int] = {}  # server: count

    # -----------------------------------------------------------------------
    # Публичный API
    # -----------------------------------------------------------------------

    def run(self, task: str, chat_id: str = "") -> str:
        """
        Запустить полный цикл исследования.

        Args:
            task:    Запрос пользователя (например, "хочу сходить в кино")
            chat_id: Telegram chat_id или @username для уведомлений

        Returns:
            Финальный текст ответа (или сообщение об ошибке)
        """
        self.context = ResearchContext()
        self.context.task = task
        self.context.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

        try:
            # === ЭТАП 1: Приём задачи ===
            self._transition(ResearchState.TASK_RECEIVED)
            self._notify(ResearchState.TASK_RECEIVED.progress_message)
            self._log(ResearchState.TASK_RECEIVED, "complete", task[:100])

            # === ЭТАП 2: Первичный поиск ===
            self._transition(ResearchState.SEARCH_INITIAL)
            self._notify(ResearchState.SEARCH_INITIAL.progress_message)
            search_queries = self._agent_plan_search(task)
            self._print(f"[SEARCH_INITIAL] Запросы: {search_queries}")
            initial_links = self._search(search_queries)
            self.context.initial_links = initial_links
            self._print(f"[SEARCH_INITIAL] Найдено {len(initial_links)} ссылок")
            self._log(ResearchState.SEARCH_INITIAL, "complete", f"{len(initial_links)} ссылок")

            # === ЭТАП 3: Fetch страниц ===
            self._transition(ResearchState.FETCH_INITIAL)
            self._notify(f"📄 Читаю {len(initial_links)} страниц...")
            initial_docs = self._fetch(initial_links)
            self.context.initial_docs = initial_docs
            errors = len(initial_links) - len(initial_docs)
            self._print(f"[FETCH_INITIAL] Получено {len(initial_docs)} документов ({errors} ошибок)")
            self._log(ResearchState.FETCH_INITIAL, "complete", f"{len(initial_docs)} документов")

            # === ЭТАП 4: Первая суммаризация ===
            self._transition(ResearchState.SUMMARIZE_INITIAL)
            self._notify(ResearchState.SUMMARIZE_INITIAL.progress_message)
            summary_v1 = self._agent_summarize(initial_docs, task)
            self.context.summary_v1 = summary_v1
            self._print(f"[SUMMARIZE_INITIAL] Суммаризация готова ({len(summary_v1)} символов)")
            self._log(ResearchState.SUMMARIZE_INITIAL, "complete")

            # === ЭТАП 5: Уточняющий поиск ===
            self._transition(ResearchState.SEARCH_DEEP)
            self._notify(ResearchState.SEARCH_DEEP.progress_message)
            deep_queries = self._agent_plan_deep_search(summary_v1)
            self._print(f"[SEARCH_DEEP] Уточняющие запросы: {deep_queries}")
            all_deep = self._search(deep_queries)
            # Инвариант: дедупликация — убираем ссылки из первого прохода
            deep_links = [l for l in all_deep if l not in self.context.seen_urls]
            self.context.deep_links = deep_links
            self._print(f"[SEARCH_DEEP] Найдено {len(deep_links)} новых ссылок")
            self._log(ResearchState.SEARCH_DEEP, "complete", f"{len(deep_links)} новых ссылок")

            # === ЭТАП 6: Fetch доп. страниц ===
            self._transition(ResearchState.FETCH_DEEP)
            self._notify(f"📄 Читаю ещё {len(deep_links)} страниц...")
            deep_docs = self._fetch(deep_links)
            self.context.deep_docs = deep_docs
            self._print(f"[FETCH_DEEP] Получено {len(deep_docs)} документов")
            self._log(ResearchState.FETCH_DEEP, "complete", f"{len(deep_docs)} документов")

            # === ЭТАП 7: Финальный синтез ===
            # Инвариант: нельзя синтезировать без первой суммаризации
            if not self.context.summary_v1:
                raise RuntimeError("Инвариант нарушен: финальный синтез без суммаризации")
            self._transition(ResearchState.SYNTHESIZE_FINAL)
            self._notify(ResearchState.SYNTHESIZE_FINAL.progress_message)
            final_result = self._agent_synthesize(summary_v1, deep_docs, task)
            self.context.final_result = final_result
            self._print(f"[SYNTHESIZE_FINAL] Финальный ответ: {len(final_result)} символов")
            self._log(ResearchState.SYNTHESIZE_FINAL, "complete")

            # === ЭТАП 8: Доставка ===
            self._transition(ResearchState.DELIVERED)
            self._send_result(final_result)
            self._log(ResearchState.DELIVERED, "complete")
            self._print("[DELIVERED] ✅ Отправлено в Telegram")

            return final_result

        except Exception as exc:
            self.state = ResearchState.FAILED
            error_msg = f"❌ Ошибка: {exc}"
            self._print(f"[FAILED] {error_msg}")
            try:
                self._notify(error_msg)
                self._log(ResearchState.FAILED, "failed", str(exc))
            except Exception:
                pass
            return error_msg

    # -----------------------------------------------------------------------
    # Вспомогательные методы
    # -----------------------------------------------------------------------

    def _search(self, queries: list[str]) -> list[str]:
        """Вызывает search_server для каждого запроса, собирает URL (дедупликация).

        Сохраняет сниппеты в context.search_snippets как резервный контент на случай,
        если страница не сможет быть скачана скрапером.
        """
        all_links: list[str] = []
        for q in queries:
            try:
                raw = self.search.call_tool("search_web", {"query": q, "limit": 10})
                self._count_call("search")
                data = json.loads(raw)
                if isinstance(data, dict) and "error" in data:
                    self._print(f"  search error: {data['error']}")
                    continue
                for item in data:
                    url = item.get("url") if isinstance(item, dict) else str(item)
                    if url and url not in self.context.seen_urls:
                        self.context.seen_urls.add(url)
                        all_links.append(url)
                        # Сохраняем сниппет для использования как запасного контента
                        if isinstance(item, dict):
                            self.context.search_snippets[url] = {
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                            }
            except Exception as exc:
                self._print(f"  search exception: {exc}")

        # Инвариант: не более MAX_LINKS_PER_ROUND ссылок
        return all_links[:self.MAX_LINKS_PER_ROUND]

    def _fetch(self, urls: list[str]) -> list[dict]:
        """Вызывает scraper_server для списка URL, возвращает документы.

        Если скачать страницу не удалось, подставляет сниппет из поискового результата
        как запасной контент — это позволяет LLM работать с данными даже при недоступных URL.
        """
        if not urls:
            return []
        try:
            raw = self.scraper.call_tool("fetch_urls", {"urls": urls})
            self._count_call("scraper")
            docs = json.loads(raw)
        except Exception as exc:
            self._print(f"  fetch exception: {exc}")
            docs = []

        result = []
        fetched_urls = {d.get("url") for d in docs if isinstance(d, dict)}

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            url = doc.get("url", "")
            if doc.get("error") is None and doc.get("content"):
                result.append(doc)
            elif url in self.context.search_snippets:
                # Страница недоступна — используем сниппет из поиска как запасной контент
                meta = self.context.search_snippets[url]
                snippet = meta.get("snippet", "")
                if snippet:
                    self._print(f"  fetch fallback to snippet for {url}")
                    result.append({
                        "url": url,
                        "title": meta.get("title", ""),
                        "content": snippet,
                        "error": None,
                    })

        # Добавляем запасные документы для URL, которые scraper вообще не вернул
        for url in urls:
            if url not in fetched_urls and url in self.context.search_snippets:
                meta = self.context.search_snippets[url]
                snippet = meta.get("snippet", "")
                if snippet:
                    self._print(f"  fetch fallback to snippet for {url} (not fetched)")
                    result.append({
                        "url": url,
                        "title": meta.get("title", ""),
                        "content": snippet,
                        "error": None,
                    })

        return result

    def _agent_plan_search(self, task: str) -> list[str]:
        """LLM формирует 2-4 поисковых запроса по задаче."""
        max_q = int(os.environ.get("RESEARCH_MAX_SEARCH_QUERIES", "4"))
        prompt = (
            f"Задача пользователя: \"{task}\"\n\n"
            f"Сформируй {min(max_q, 4)} поисковых запроса для сбора информации.\n"
            f"Верни только список запросов, по одному на строку, без нумерации, "
            f"без пояснений."
        )
        response = self.llm(prompt)
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        return queries[:max_q] or [task]

    def _agent_plan_deep_search(self, summary: str) -> list[str]:
        """LLM формирует уточняющие запросы по результатам первого прохода."""
        prompt = (
            f"Вот первичная суммаризация результатов исследования:\n{summary}\n\n"
            f"Какие данные стоит дособрать? Сформируй 2-3 уточняющих поисковых запроса "
            f"для получения отзывов, конкретных фактов, деталей.\n"
            f"Верни только список запросов, по одному на строку, без нумерации."
        )
        response = self.llm(prompt)
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        return queries[:3] or ["дополнительная информация " + self.context.task[:50]]

    def _agent_summarize(self, documents: list[dict], task: str) -> str:
        """LLM делает первичную суммаризацию документов."""
        if not documents:
            return f"Документы для суммаризации не найдены. Задача: {task}"
        texts = "\n\n---\n\n".join(
            f"## {d.get('title', 'Без заголовка')}\n{d.get('content', '')[:1500]}"
            for d in documents[:8]
        )
        prompt = (
            f"Задача пользователя: \"{task}\"\n\n"
            f"Вот собранные материалы:\n\n{texts}\n\n"
            f"Проанализируй материалы применительно к задаче пользователя.\n"
            f"Выдели ключевые факты, данные и сведения, прямо отвечающие на вопрос.\n"
            f"Укажи, какой информации не хватает для полного ответа."
        )
        return self.llm(prompt)

    def _agent_synthesize(self, summary_v1: str, deep_docs: list[dict], task: str) -> str:
        """LLM формирует финальный ответ из всех данных."""
        deep_texts = ""
        if deep_docs:
            deep_texts = "\n\n---\n\n".join(
                f"## {d.get('title', '')}\n{d.get('content', '')[:1000]}"
                for d in deep_docs[:6]
            )
        else:
            deep_texts = "Дополнительных данных нет."

        prompt = (
            f"Задача пользователя: \"{task}\"\n\n"
            f"Первичный анализ:\n{summary_v1}\n\n"
            f"Дополнительные данные:\n{deep_texts}\n\n"
            f"Сформируй финальный ответ для пользователя на русском языке.\n\n"
            f"ПРАВИЛА:\n"
            f"1. Отвечай строго на поставленную задачу — используй только факты из собранных данных.\n"
            f"2. Структурируй ответ логично: ключевые выводы, детали, источники.\n"
            f"3. Если данных недостаточно — честно скажи об этом и изложи то, что удалось найти.\n"
            f"4. Формат: Markdown, кратко и по делу."
        )
        return self.llm(prompt)

    def _notify(self, message: str) -> None:
        """Отправить прогресс-уведомление в Telegram."""
        try:
            self.telegram.call_tool("send_progress", {
                "chat_id": self.context.chat_id,
                "stage": self.state.value,
                "message": message,
            })
            self._count_call("telegram")
            self._telegram_calls += 1
        except Exception as exc:
            self._print(f"  telegram notify error: {exc}")

    def _send_result(self, text: str) -> None:
        """Отправить финальный результат в Telegram."""
        try:
            self.telegram.call_tool("send_result", {
                "chat_id": self.context.chat_id,
                "text": text,
            })
            self._count_call("telegram")
            self._telegram_calls += 1
        except Exception as exc:
            self._print(f"  telegram send_result error: {exc}")

    def _log(self, stage: ResearchState, status: str, details: str = "") -> None:
        """Записать этап в журнал."""
        try:
            self.journal.call_tool("log_stage", {
                "task_id": self.context.task_id,
                "stage": stage.value,
                "status": status,
                "details": details,
            })
            self._count_call("journal")
            self._journal_calls += 1
        except Exception as exc:
            self._print(f"  journal error: {exc}")

    def _transition(self, new_state: ResearchState) -> None:
        """Переход в новое состояние (с проверкой допустимости)."""
        allowed = VALID_TRANSITIONS.get(self.state, [])
        if new_state not in allowed and self.state != new_state:
            # При TASK_RECEIVED допускаем переход в себя (первый вызов)
            if not (self.state == ResearchState.TASK_RECEIVED and new_state == ResearchState.TASK_RECEIVED):
                self._print(
                    f"  [transition] {self.state} → {new_state} "
                    f"(не в списке допустимых: {allowed})"
                )
        self.state = new_state

    def _count_call(self, server: str) -> None:
        """Считаем вызовы по серверам (для проверки инвариантов)."""
        self._search_calls[server] = self._search_calls.get(server, 0) + 1

    def _print(self, msg: str) -> None:
        """Вывод в stdout если verbose=True."""
        if self.verbose:
            print(msg, flush=True)

    # -----------------------------------------------------------------------
    # Отчёт об инвариантах
    # -----------------------------------------------------------------------

    def get_invariant_report(self) -> dict:
        """Вернуть отчёт о соблюдении инвариантов."""
        initial_set = set(self.context.initial_links)
        deep_set = set(self.context.deep_links)
        intersection = initial_set & deep_set

        return {
            "initial_links_count": len(self.context.initial_links),
            "deep_links_count": len(self.context.deep_links),
            "initial_links_le_10": len(self.context.initial_links) <= self.MAX_LINKS_PER_ROUND,
            "deep_links_le_10": len(self.context.deep_links) <= self.MAX_LINKS_PER_ROUND,
            "deduplication_ok": len(intersection) == 0,
            "duplicate_urls": list(intersection),
            "summary_v1_exists": bool(self.context.summary_v1),
            "final_result_exists": bool(self.context.final_result),
            "telegram_calls": self._telegram_calls,
            "journal_calls": self._journal_calls,
            "calls_by_server": dict(self._search_calls),
            "two_rounds_only": True,  # архитектурный инвариант: ровно 2 раунда
        }
