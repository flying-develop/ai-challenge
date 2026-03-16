"""Стратегии разбиения документов на чанки (паттерн Strategy).

Реализует 4 стратегии нарезки:
- fixed_500     — фиксированный размер 500 токенов
- fixed_1000    — фиксированный размер 1000 токенов
- overlap_600_100 — с перекрытием 100 токенов
- structural    — по заголовкам Markdown

Оценка токенов — эвристика без tiktoken (коэф. 1.4 для русского BPE).
chunk_id — детерминированный SHA256(source + index + text[:200])[:16].
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..loader import Document


# ---------------------------------------------------------------------------
# Оценка токенов
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Оценить количество токенов без tiktoken.

    Эвристика: words * 1.4 + specials * 0.3.
    Коэффициент 1.4 — среднее для русского BPE (субсловные токены).

    Args:
        text: Входной текст.

    Returns:
        Приближённое количество токенов.
    """
    words = len(text.split())
    specials = len(re.findall(r'[^\w\s]', text))
    return int(words * 1.4 + specials * 0.3)


# ---------------------------------------------------------------------------
# Dataclass Chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Единица индексации — фрагмент документа.

    Attributes:
        chunk_id:    Детерминированный ID (SHA256 от source+index+text[:200])[:16].
        text:        Текст чанка.
        source:      Относительный путь к файлу.
        file:        Basename файла.
        section:     Заголовок ближайшего раздела.
        doc_title:   Заголовок документа.
        chunk_index: Порядковый номер в документе.
        token_count: Оценка количества токенов.
        strategy:    Имя стратегии.
        metadata:    Дополнительные поля.
    """

    chunk_id: str
    text: str
    source: str
    file: str
    section: str
    doc_title: str
    chunk_index: int
    token_count: int
    strategy: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Вспомогательная функция: chunk_id
# ---------------------------------------------------------------------------

def _make_chunk_id(source: str, index: int, text: str, strategy: str = "") -> str:
    """Детерминированный chunk_id — SHA256(strategy + source + index + text[:200])[:16]."""
    raw = f"{strategy}:{source}:{index}:{text[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _find_section(content: str, pos: int, sections: list) -> str:
    """Найти заголовок раздела, которому принадлежит позиция pos."""
    current = ""
    for level, title, start, end in sections:
        if start <= pos:
            current = title
        else:
            break
    return current


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class ChunkingStrategy(ABC):
    """Абстрактная стратегия нарезки документов на чанки.

    Реализует паттерн Strategy: конкретные стратегии переопределяют
    chunk_document(), а chunk_all() — шаблонный метод.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя стратегии (используется в БД)."""
        ...

    @abstractmethod
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Разбить один документ на чанки.

        Args:
            doc: Загруженный документ.

        Returns:
            Список чанков.
        """
        ...

    def chunk_all(self, documents: list[Document]) -> list[Chunk]:
        """Разбить список документов на чанки (шаблонный метод).

        Args:
            documents: Список документов.

        Returns:
            Все чанки всех документов.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _make_chunk(
        self,
        text: str,
        doc: Document,
        index: int,
        section: str = "",
    ) -> Chunk:
        """Фабричный метод для создания Chunk с корректным chunk_id."""
        return Chunk(
            chunk_id=_make_chunk_id(doc.source, index, text, self.name),
            text=text,
            source=doc.source,
            file=doc.source.split("/")[-1] if "/" in doc.source else doc.source,
            section=section,
            doc_title=doc.title,
            chunk_index=index,
            token_count=estimate_tokens(text),
            strategy=self.name,
        )


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class FixedSizeChunker(ChunkingStrategy):
    """Нарезка по параграфам с фиксированным лимитом токенов.

    Алгоритм:
    1. Разбивает content на параграфы (двойной перевод строки).
    2. Накапливает параграфы пока не превышен max_tokens.
    3. Если один параграф > лимита — дополнительно нарезает по строкам.
    """

    def __init__(self, max_tokens: int = 500, strategy_name: str | None = None) -> None:
        self._max_tokens = max_tokens
        self._name = strategy_name or f"fixed_{max_tokens}"

    @property
    def name(self) -> str:
        return self._name

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Разбить документ на чанки фиксированного размера."""
        paragraphs = re.split(r'\n{2,}', doc.content)
        raw_chunks = self._split_paragraphs(paragraphs)

        chunks: list[Chunk] = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            # Найти раздел для первого символа текста
            pos = doc.content.find(text[:50]) if text else 0
            section = _find_section(doc.content, max(pos, 0), doc.sections)
            chunks.append(self._make_chunk(text, doc, i, section))

        return chunks

    def _split_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Накапливать параграфы в чанки с учётом лимита."""
        result: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = estimate_tokens(para)

            # Один параграф превышает лимит — нарезать по строкам
            if para_tokens > self._max_tokens:
                if current_parts:
                    result.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                result.extend(self._split_by_lines(para))
                continue

            if current_tokens + para_tokens > self._max_tokens and current_parts:
                result.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            result.append("\n\n".join(current_parts))

        return result

    def _split_by_lines(self, text: str) -> list[str]:
        """Нарезать длинный параграф по строкам."""
        lines = text.split("\n")
        result: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = estimate_tokens(line)
            if current_tokens + line_tokens > self._max_tokens and current_lines:
                result.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            result.append("\n".join(current_lines))

        return result


# ---------------------------------------------------------------------------
# FixedOverlapChunker
# ---------------------------------------------------------------------------

class FixedOverlapChunker(FixedSizeChunker):
    """Фиксированный размер чанков с перекрытием (overlap).

    Наследуется от FixedSizeChunker.
    К каждому чанку (кроме первого) препендит хвост предыдущего (~overlap_tokens).

    Args:
        max_tokens:    Максимальный размер базового чанка.
        overlap_tokens: Количество токенов перекрытия (хвост предыдущего чанка).
    """

    def __init__(
        self,
        max_tokens: int = 600,
        overlap_tokens: int = 100,
        strategy_name: str | None = None,
    ) -> None:
        super().__init__(max_tokens=max_tokens, strategy_name=strategy_name or f"overlap_{max_tokens}_{overlap_tokens}")
        self._overlap_tokens = overlap_tokens

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Разбить документ с перекрытием между чанками."""
        base_chunks = super().chunk_document(doc)
        if len(base_chunks) <= 1:
            return base_chunks

        result: list[Chunk] = [base_chunks[0]]

        for i, chunk in enumerate(base_chunks[1:], 1):
            prev_text = base_chunks[i - 1].text
            overlap_text = self._tail(prev_text)

            if overlap_text:
                new_text = overlap_text + "\n\n" + chunk.text
            else:
                new_text = chunk.text

            pos = doc.content.find(chunk.text[:50]) if chunk.text else 0
            section = _find_section(doc.content, max(pos, 0), doc.sections)

            result.append(Chunk(
                chunk_id=_make_chunk_id(doc.source, i, new_text, self.name),
                text=new_text,
                source=doc.source,
                file=chunk.file,
                section=section,
                doc_title=doc.title,
                chunk_index=i,
                token_count=estimate_tokens(new_text),
                strategy=self.name,
            ))

        return result

    def _tail(self, text: str) -> str:
        """Взять хвост текста ~overlap_tokens токенов."""
        words = text.split()
        # Приближение: overlap_tokens / 1.4 ≈ слов
        target_words = max(1, int(self._overlap_tokens / 1.4))
        if len(words) <= target_words:
            return text
        return " ".join(words[-target_words:])


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------

class StructuralChunker(ChunkingStrategy):
    """Нарезка по заголовкам ## / ### из Document.sections.

    Алгоритм:
    1. Каждый раздел (между заголовками) — один чанк.
    2. Маленькие разделы (< min_tokens) мержатся с предыдущим.
    3. Большие разделы (> max_tokens) дорезаются через FixedSizeChunker.
    """

    def __init__(self, max_tokens: int = 1000, min_tokens: int = 50) -> None:
        self._max_tokens = max_tokens
        self._min_tokens = min_tokens
        self._splitter = FixedSizeChunker(max_tokens=max_tokens, strategy_name="structural_sub")

    @property
    def name(self) -> str:
        return "structural"

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Разбить документ по структуре заголовков."""
        if not doc.sections:
            # Нет заголовков — фолбэк на FixedSizeChunker
            sub = FixedSizeChunker(max_tokens=self._max_tokens, strategy_name="structural")
            chunks = sub.chunk_document(doc)
            for c in chunks:
                c.strategy = self.name
            return chunks

        # Собрать тексты разделов
        section_texts: list[tuple[str, str]] = []  # (section_title, text)
        for level, title, start, end in doc.sections:
            text = doc.content[start:end].strip()
            if text:
                section_texts.append((title, text))

        # Текст до первого заголовка (preface)
        first_start = doc.sections[0][2]
        preface = doc.content[:first_start].strip()
        if preface:
            section_texts.insert(0, ("", preface))

        # Мерж маленьких разделов + нарезка больших
        merged = self._merge_small(section_texts)
        chunks: list[Chunk] = []
        global_index = 0

        for section_title, text in merged:
            tokens = estimate_tokens(text)
            if tokens > self._max_tokens:
                # Дорезать через FixedSizeChunker
                import copy
                sub_doc = copy.copy(doc)
                sub_doc.content = text
                sub_doc.sections = []
                sub_chunks = self._splitter.chunk_document(sub_doc)
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        chunk_id=_make_chunk_id(doc.source, global_index, sc.text, self.name),
                        text=sc.text,
                        source=doc.source,
                        file=sc.file,
                        section=section_title,
                        doc_title=doc.title,
                        chunk_index=global_index,
                        token_count=sc.token_count,
                        strategy=self.name,
                    ))
                    global_index += 1
            else:
                chunks.append(self._make_chunk(text, doc, global_index, section_title))
                global_index += 1

        return chunks

    def _merge_small(self, sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Объединить маленькие разделы с предыдущим."""
        if not sections:
            return []

        result: list[tuple[str, str]] = [sections[0]]

        for title, text in sections[1:]:
            tokens = estimate_tokens(text)
            if tokens < self._min_tokens and result:
                prev_title, prev_text = result[-1]
                merged_text = prev_text + "\n\n" + text
                result[-1] = (prev_title, merged_text)
            else:
                result.append((title, text))

        return result


# ---------------------------------------------------------------------------
# Реестр стратегий
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, ChunkingStrategy] = {
    "fixed_500": FixedSizeChunker(max_tokens=500, strategy_name="fixed_500"),
    "fixed_1000": FixedSizeChunker(max_tokens=1000, strategy_name="fixed_1000"),
    "overlap_600_100": FixedOverlapChunker(max_tokens=600, overlap_tokens=100, strategy_name="overlap_600_100"),
    "structural": StructuralChunker(max_tokens=1000, min_tokens=50),
}


def get_strategy(name: str) -> ChunkingStrategy:
    """Получить стратегию по имени.

    Args:
        name: Одно из fixed_500 | fixed_1000 | overlap_600_100 | structural.

    Returns:
        Экземпляр ChunkingStrategy.

    Raises:
        ValueError: Если стратегия не найдена.
    """
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES)
        raise ValueError(f"Неизвестная стратегия '{name}'. Доступны: {available}")
    return STRATEGIES[name]
