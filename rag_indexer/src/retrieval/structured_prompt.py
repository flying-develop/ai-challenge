"""Структурированный промпт для RAG с обязательными цитатами и источниками.

Заменяет RAGQueryBuilder для режима структурированных ответов.
LLM обязана вернуть ответ в формате [ANSWER]/[SOURCES]/[QUOTES].
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .retriever import RetrievalResult


@dataclass
class StructuredPrompt:
    """Готовый промпт для передачи в LLM."""

    system: str
    user: str


class StructuredRAGPrompt:
    """Формирует промпт, который ОБЯЗЫВАЕТ LLM вернуть структурированный
    ответ с цитатами и источниками.

    Поддерживает три уровня уверенности (confidence):
        HIGH   (≥ 0.7) — обычный ответ
        MEDIUM (≥ 0.4) — ответ с предупреждением о неполноте
        LOW    (< 0.4) — усиленная инструкция сказать "не знаю"
    """

    LOW_CONFIDENCE_THRESHOLD: float = float(
        os.environ.get("RAG_CONFIDENCE_THRESHOLD", "0.4")
    )

    SYSTEM_PROMPT = """Ты — ассистент по документации проекта podkop.
Ты отвечаешь ТОЛЬКО на основе предоставленного контекста.

ПРАВИЛА (обязательны к выполнению):

1. ОТВЕЧАЙ только на основе контекста. Если информации нет — скажи об этом.
2. ЦИТИРУЙ: каждое утверждение подкрепляй цитатой из контекста.
3. УКАЗЫВАЙ ИСТОЧНИКИ: для каждой цитаты указывай файл и раздел.
4. НЕ ДОДУМЫВАЙ: если в контексте чего-то нет — не придумывай.

ФОРМАТ ОТВЕТА (строго соблюдай):

[ANSWER]
Твой ответ на вопрос. Каждое утверждение сопровождай ссылкой
на цитату в формате [N], где N — номер цитаты.

[SOURCES]
1. файл.md | Раздел: название_раздела
2. файл.md | Раздел: название_раздела
...

[QUOTES]
1. "точная цитата из контекста" — файл.md
2. "точная цитата из контекста" — файл.md
...

Если контекст НЕ содержит информации для ответа, верни:

[ANSWER]
К сожалению, в документации не нашлось информации по вашему вопросу.
Попробуйте переформулировать запрос или уточнить, что именно вас интересует.

[SOURCES]
(нет релевантных источников)

[QUOTES]
(нет релевантных цитат)"""

    CONTEXT_TEMPLATE = """Контекст из документации:

{chunks}

---
Вопрос пользователя: {question}

Ответь в указанном формате [ANSWER], [SOURCES], [QUOTES]."""

    def build(
        self,
        question: str,
        results: list[RetrievalResult],
        confidence: float,
    ) -> StructuredPrompt:
        """Формирует промпт для LLM.

        Args:
            question:   Вопрос пользователя.
            results:    Найденные и отфильтрованные чанки.
            confidence: Средний score контекста (из ConfidenceScorer).

        Returns:
            StructuredPrompt с system и user строками.
        """
        chunks_text = "\n\n".join(
            f"[Источник {i + 1}: {r.source}, раздел: {r.section or 'нет раздела'}]\n{r.text}"
            for i, r in enumerate(results)
        )

        system = self.SYSTEM_PROMPT
        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            system += (
                "\n\nВНИМАНИЕ: релевантность контекста НИЗКАЯ. "
                "Если не уверен в ответе — ОБЯЗАТЕЛЬНО скажи, "
                "что информации недостаточно."
            )

        user = self.CONTEXT_TEMPLATE.format(
            chunks=chunks_text,
            question=question,
        )

        return StructuredPrompt(system=system, user=user)
