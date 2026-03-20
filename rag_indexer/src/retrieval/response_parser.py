"""Парсер структурированного ответа LLM в формате [ANSWER]/[SOURCES]/[QUOTES].

Основной анти-галлюцинационный механизм: верификация цитат по текстам чанков
без внешних библиотек — посимвольное скользящее окно + подстрочный поиск.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retriever import RetrievalResult


@dataclass
class SourceRef:
    """Ссылка на источник из блока [SOURCES]."""

    index: int      # порядковый номер (1, 2, ...)
    file: str       # имя/путь файла
    section: str    # раздел документа


@dataclass
class Quote:
    """Цитата из блока [QUOTES]."""

    index: int          # порядковый номер [1], [2], ...
    text: str           # текст цитаты
    source_file: str    # имя файла-источника
    verified: bool = False  # True если цитата найдена в контексте


@dataclass
class StructuredResponse:
    """Разобранный структурированный ответ LLM."""

    answer: str                  # текст из блока [ANSWER]
    sources: list[SourceRef]     # разобранные источники
    quotes: list[Quote]          # разобранные цитаты
    is_refusal: bool             # True если модель сказала "не знаю"
    raw_response: str            # исходный ответ LLM (для отладки)
    confidence: float            # средняя релевантность чанков

    @property
    def verified_quotes(self) -> list[Quote]:
        """Возвращает только верифицированные цитаты."""
        return [q for q in self.quotes if q.verified]

    @property
    def verified_ratio(self) -> float:
        """Доля верифицированных цитат (0.0–1.0)."""
        if not self.quotes:
            return 1.0  # нет цитат → нет нарушений
        return len(self.verified_quotes) / len(self.quotes)

    @property
    def has_sources(self) -> bool:
        """True если есть хотя бы один не-пустой источник."""
        return len(self.sources) > 0

    @property
    def has_quotes(self) -> bool:
        """True если есть хотя бы одна цитата."""
        return len(self.quotes) > 0


# Маркеры для обнаружения отказа
_REFUSAL_PATTERNS = [
    "не нашлось информации",
    "недостаточно информации",
    "не знаю",
    "нет информации",
    "не могу ответить",
    "информация отсутствует",
    "не содержит информации",
]

_EMPTY_SOURCES_PATTERN = re.compile(
    r"\(нет\s+релевантных\s+источников?\)", re.IGNORECASE
)
_EMPTY_QUOTES_PATTERN = re.compile(
    r"\(нет\s+релевантных\s+цитат?\)", re.IGNORECASE
)


class ResponseParser:
    """Парсит ответ LLM в формате [ANSWER]/[SOURCES]/[QUOTES].

    Если ответ не содержит маркеров — создаёт fallback-ответ,
    сохраняя сырой текст в поле answer.
    """

    # regex для разбивки на блоки
    _BLOCK_RE = re.compile(
        r"\[(ANSWER|SOURCES|QUOTES)\]\s*\n?(.*?)(?=\[(?:ANSWER|SOURCES|QUOTES)\]|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # regex для строки источника: "1. file.md | Раздел: ..."
    _SOURCE_LINE_RE = re.compile(
        r"^\s*(\d+)\.\s+(.+?)\s*\|\s*Раздел:\s*(.+?)\s*$",
        re.IGNORECASE,
    )

    # regex для строки цитаты: '1. "текст цитаты" — file.md'
    _QUOTE_LINE_RE = re.compile(
        r"""^\s*(\d+)\.\s+"(.+?)"\s*[—–-]+\s*(.+?)\s*$""",
        re.DOTALL,
    )

    def parse(
        self,
        raw: str,
        search_results: list[RetrievalResult],
        confidence: float = 0.0,
    ) -> StructuredResponse:
        """Разобрать сырой ответ LLM в StructuredResponse.

        Args:
            raw:            Текст ответа от LLM.
            search_results: Чанки, отправленные в контекст (для верификации цитат).
            confidence:     Средний score контекста.

        Returns:
            Заполненный StructuredResponse.
        """
        blocks = self._extract_blocks(raw)

        answer_text = blocks.get("ANSWER", "").strip()
        sources_text = blocks.get("SOURCES", "").strip()
        quotes_text = blocks.get("QUOTES", "").strip()

        # Если структуры нет — fallback
        if not blocks:
            return StructuredResponse(
                answer=raw.strip(),
                sources=[],
                quotes=[],
                is_refusal=False,
                raw_response=raw,
                confidence=confidence,
            )

        sources = self._parse_sources(sources_text)
        quotes = self._parse_quotes(quotes_text)

        # Верификация цитат по текстам чанков
        chunk_texts = [r.text for r in search_results]
        for quote in quotes:
            quote.verified = self._verify_quote(quote.text, chunk_texts)

        is_refusal = self._detect_refusal(answer_text, sources_text, quotes_text)

        return StructuredResponse(
            answer=answer_text,
            sources=sources,
            quotes=quotes,
            is_refusal=is_refusal,
            raw_response=raw,
            confidence=confidence,
        )

    def _extract_blocks(self, raw: str) -> dict[str, str]:
        """Извлекает блоки [ANSWER], [SOURCES], [QUOTES] из сырого ответа."""
        result: dict[str, str] = {}
        for m in self._BLOCK_RE.finditer(raw):
            block_name = m.group(1).upper()
            block_content = m.group(2)
            result[block_name] = block_content
        return result

    def _parse_sources(self, text: str) -> list[SourceRef]:
        """Разобрать блок [SOURCES] в список SourceRef."""
        if not text or _EMPTY_SOURCES_PATTERN.search(text):
            return []

        sources: list[SourceRef] = []
        for line in text.splitlines():
            m = self._SOURCE_LINE_RE.match(line)
            if m:
                sources.append(
                    SourceRef(
                        index=int(m.group(1)),
                        file=m.group(2).strip(),
                        section=m.group(3).strip(),
                    )
                )
        return sources

    def _parse_quotes(self, text: str) -> list[Quote]:
        """Разобрать блок [QUOTES] в список Quote."""
        if not text or _EMPTY_QUOTES_PATTERN.search(text):
            return []

        quotes: list[Quote] = []
        for line in text.splitlines():
            m = self._QUOTE_LINE_RE.match(line)
            if m:
                quotes.append(
                    Quote(
                        index=int(m.group(1)),
                        text=m.group(2).strip(),
                        source_file=m.group(3).strip(),
                    )
                )
        return quotes

    def _detect_refusal(
        self, answer: str, sources: str, quotes: str
    ) -> bool:
        """Определить, является ли ответ отказом ("не знаю").

        Критерии:
        - answer содержит фразы-маркеры отказа
        - ИЛИ блок sources/quotes помечен как пустой
        """
        answer_lower = answer.lower()
        if any(p in answer_lower for p in _REFUSAL_PATTERNS):
            return True
        if _EMPTY_SOURCES_PATTERN.search(sources):
            return True
        return False

    # ── Верификация цитат ────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Нормализация: нижний регистр, убрать лишние пробелы."""
        return " ".join(text.lower().split())

    def _verify_quote(self, quote_text: str, chunks: list[str]) -> bool:
        """Проверить, что цитата действительно есть в одном из чанков.

        Алгоритм:
        1. Точный подстрочный поиск (нормализованный текст).
        2. Скользящее окно длины len(quote): ищем окно с совпадением > 0.85.

        Args:
            quote_text: Текст цитаты из ответа LLM.
            chunks:     Тексты чанков, отправленных в контекст.

        Returns:
            True если цитата верифицирована.
        """
        if not quote_text:
            return False

        norm_quote = self._normalize(quote_text)
        if not norm_quote:
            return False

        for chunk in chunks:
            norm_chunk = self._normalize(chunk)
            # Точный подстрочный поиск
            if norm_quote in norm_chunk:
                return True
            # Скользящее окно
            if self._char_similarity_window(norm_quote, norm_chunk) > 0.85:
                return True

        return False

    @staticmethod
    def _char_similarity_window(query: str, text: str) -> float:
        """Посимвольное сходство: скользящее окно длины len(query) по text.

        Для каждой позиции окна считаем долю совпадающих символов.
        Возвращаем максимум по всем позициям.

        Без difflib — чистая реализация.
        """
        q_len = len(query)
        t_len = len(text)

        if q_len == 0:
            return 1.0
        if t_len < q_len:
            # Текст короче запроса — считаем покрытие наоборот
            matches = sum(1 for a, b in zip(query, text) if a == b)
            return matches / q_len

        best = 0.0
        for start in range(t_len - q_len + 1):
            window = text[start: start + q_len]
            matches = sum(1 for a, b in zip(query, window) if a == b)
            ratio = matches / q_len
            if ratio > best:
                best = ratio
                if best >= 0.85:  # ранний выход
                    return best

        return best
