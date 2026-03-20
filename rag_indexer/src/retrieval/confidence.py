"""Оценка уверенности в контексте перед отправкой в LLM.

Три уровня:
    HIGH   (score ≥ 0.7) — обычный ответ
    MEDIUM (0.4 ≤ score < 0.7) — ответ с предупреждением о неполноте
    LOW    (score < 0.4) — отказ: "не нашлось информации"
"""
from __future__ import annotations

import os
from enum import Enum

from .retriever import RetrievalResult


class ConfidenceLevel(str, Enum):
    """Уровень уверенности в релевантности контекста."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceScorer:
    """Оценивает, достаточно ли релевантен контекст для уверенного ответа.

    Если уверенность LOW — пайплайн должен переключить LLM в режим "не знаю"
    через усиленную инструкцию в промпте (StructuredRAGPrompt).
    """

    # Порог из .env; если не задан — 0.4
    DEFAULT_THRESHOLD: float = float(
        os.environ.get("RAG_CONFIDENCE_THRESHOLD", "0.4")
    )

    HIGH_THRESHOLD: float = 0.7

    def score(self, results: list[RetrievalResult]) -> float:
        """Вычислить средний score top-K результатов.

        Args:
            results: Список чанков после реранкинга.

        Returns:
            Средний score в диапазоне [0.0, 1.0].
            0.0 если результатов нет.
        """
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def level(
        self,
        results: list[RetrievalResult],
        threshold: float | None = None,
    ) -> ConfidenceLevel:
        """Определить уровень уверенности по результатам поиска.

        Args:
            results:   Список чанков.
            threshold: Порог LOW/MEDIUM (по умолчанию DEFAULT_THRESHOLD).

        Returns:
            ConfidenceLevel.HIGH / MEDIUM / LOW.
        """
        low_threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        avg = self.score(results)

        if avg >= self.HIGH_THRESHOLD:
            return ConfidenceLevel.HIGH
        if avg >= low_threshold:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def should_refuse(
        self,
        results: list[RetrievalResult],
        threshold: float | None = None,
    ) -> bool:
        """True если контекст слишком слаб для уверенного ответа.

        Критерии отказа (хотя бы одно):
        - средний score < threshold
        - ИЛИ top-1 score < threshold * 1.5
        - ИЛИ менее 2 результатов

        Args:
            results:   Список чанков после реранкинга.
            threshold: Порог (по умолчанию DEFAULT_THRESHOLD).

        Returns:
            True если рекомендуется режим "не знаю".
        """
        thr = threshold if threshold is not None else self.DEFAULT_THRESHOLD

        if len(results) < 2:
            return True

        avg = self.score(results)
        if avg < thr:
            return True

        top_score = results[0].score
        if top_score < thr * 1.5:
            return True

        return False
