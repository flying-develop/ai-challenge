"""CLI-форматирование структурированного ответа RAG.

Выводит блоки ANSWER / SOURCES / QUOTES с иконками и верификацией цитат.
"""
from __future__ import annotations

from .response_parser import StructuredResponse
from .confidence import ConfidenceLevel, ConfidenceScorer


def format_structured_response(
    response: StructuredResponse,
    show_verification: bool = True,
) -> str:
    """Отформатировать StructuredResponse для вывода в CLI.

    Args:
        response:          Разобранный ответ LLM.
        show_verification: Показывать ли строку "Цитаты верифицированы: N/M".

    Returns:
        Многострочная строка для print().
    """
    lines: list[str] = []

    # Предупреждение о низкой релевантности
    if response.confidence < ConfidenceScorer.DEFAULT_THRESHOLD:
        lines.append(f"\u26a0\ufe0f  Низкая релевантность контекста ({response.confidence:.2f})\n")

    # ── ANSWER ───────────────────────────────────────────────────────────────
    lines.append("\U0001f4cb Ответ:")
    if response.answer:
        for text_line in response.answer.splitlines():
            lines.append(f"  {text_line}")
    else:
        lines.append("  (ответ отсутствует)")

    # ── SOURCES ──────────────────────────────────────────────────────────────
    lines.append("\n\U0001f4da Источники:")
    if response.sources:
        for src in response.sources:
            lines.append(f'  {src.index}. {src.file} | Раздел: "{src.section}"')
    else:
        lines.append("  (нет релевантных источников)")

    # ── QUOTES ───────────────────────────────────────────────────────────────
    lines.append("\n\U0001f4ac Цитаты:")
    if response.quotes:
        for q in response.quotes:
            check = "\u2705" if q.verified else "\u274c"
            lines.append(f"  [{q.index}] {check} \"{q.text}\"")
            lines.append(f"      — {q.source_file}")
    else:
        lines.append("  (нет релевантных цитат)")

    # ── Статус верификации ────────────────────────────────────────────────────
    if show_verification and response.quotes:
        total = len(response.quotes)
        verified = len(response.verified_quotes)
        pct = int(response.verified_ratio * 100)
        icon = "\u2705" if verified == total else "\u26a0\ufe0f"
        lines.append(f"\n{icon} Цитаты верифицированы: {verified}/{total} ({pct}%)")

    return "\n".join(lines)


def format_refusal(response: StructuredResponse) -> str:
    """Форматировать ответ-отказ (is_refusal=True).

    Args:
        response: StructuredResponse с is_refusal=True.

    Returns:
        Строка для вывода в CLI.
    """
    lines: list[str] = []

    conf_str = f"{response.confidence:.2f}"
    lines.append(f"\u26a0\ufe0f  Низкая релевантность контекста ({conf_str})\n")

    lines.append("\U0001f4cb Ответ:")
    for text_line in response.answer.splitlines():
        lines.append(f"  {text_line}")

    lines.append("\n\U0001f4da Источники:")
    lines.append("  (нет релевантных источников)")

    lines.append("\n\U0001f4ac Цитаты:")
    lines.append("  (нет релевантных цитат)")

    return "\n".join(lines)


def format_confidence_level(level: ConfidenceLevel, score: float) -> str:
    """Вернуть строку с описанием уровня уверенности.

    Args:
        level: ConfidenceLevel.HIGH / MEDIUM / LOW.
        score: Числовое значение (0.0–1.0).

    Returns:
        Человекочитаемая строка.
    """
    labels = {
        ConfidenceLevel.HIGH: "\U0001f7e2 ВЫСОКИЙ",
        ConfidenceLevel.MEDIUM: "\U0001f7e1 СРЕДНИЙ",
        ConfidenceLevel.LOW: "\U0001f534 НИЗКИЙ",
    }
    label = labels.get(level, str(level))
    return f"Уровень уверенности: {label} ({score:.2f})"
