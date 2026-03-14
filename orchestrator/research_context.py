"""Контекст задачи для исследовательского оркестратора.

ResearchContext хранит всё состояние одной исследовательской задачи:
входной запрос, собранные URL, документы, промежуточные суммаризации.

Образовательная концепция: явный dataclass лучше dict[str, Any] —
типизация документирует структуру и делает код читаемым.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ResearchContext:
    """Полный контекст одной исследовательской задачи."""

    # Входные данные
    task: str = ""
    chat_id: str = ""

    # Уникальный ID — генерируется автоматически
    task_id: str = field(default_factory=lambda: f"research_{int(time.time())}_{uuid.uuid4().hex[:6]}")

    # Дедупликация URL между раундами поиска
    seen_urls: set = field(default_factory=set)

    # Сниппеты поисковой выдачи: url → {title, snippet}
    # Используются как запасной контент, если страница не удалась
    search_snippets: dict = field(default_factory=dict)

    # Ссылки из первого и второго раундов
    initial_links: list = field(default_factory=list)
    deep_links: list = field(default_factory=list)

    # Документы (fetch-результаты с content)
    initial_docs: list = field(default_factory=list)
    deep_docs: list = field(default_factory=list)

    # Промежуточные результаты LLM
    summary_v1: str = ""
    final_result: str = ""

    # Статистика выполнения
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        """Секунд с начала задачи."""
        return time.time() - self.start_time

    @property
    def total_docs(self) -> int:
        return len(self.initial_docs) + len(self.deep_docs)

    @property
    def total_links(self) -> int:
        return len(self.seen_urls)
