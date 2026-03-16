"""Загрузчик Markdown-документов с парсингом Hugo frontmatter и структуры заголовков.

Модуль реализует DocumentLoader — первый шаг пайплайна RAG-индексации.
Рекурсивно обходит директорию, читает .md файлы, извлекает frontmatter (YAML)
и строит карту разделов по заголовкам ## / ###.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError as exc:
    raise ImportError("Установите pyyaml: pip install pyyaml") from exc


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Загруженный и распарсенный Markdown-документ.

    Attributes:
        content:     Чистый текст без frontmatter.
        source:      Относительный путь от корня docs.
        file_path:   Абсолютный путь к файлу.
        title:       Заголовок из frontmatter.title или первого #-заголовка.
        weight:      Hugo-вес (frontmatter.weight) для сортировки.
        frontmatter: Все поля YAML-frontmatter.
        sections:    Список (level, title, start_pos, end_pos) для нарезки.
    """

    content: str
    source: str
    file_path: str
    title: str
    weight: int
    frontmatter: dict
    sections: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Регулярки
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Извлечь YAML frontmatter и вернуть (meta_dict, clean_content).

    Если frontmatter отсутствует — возвращает ({}, raw).
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}

    content = raw[match.end():]
    return meta, content


def _parse_sections(content: str) -> list[tuple[int, str, int, int]]:
    """Построить карту разделов по ##/### заголовкам.

    Returns:
        Список кортежей (level, title, start_pos, end_pos).
        end_pos следующего заголовка = start_pos текущего (или len(content)).
    """
    matches = list(_HEADING_RE.finditer(content))
    sections: list[tuple[int, str, int, int]] = []

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((level, title, start, end))

    return sections


def _extract_title(meta: dict, content: str) -> str:
    """Получить заголовок из frontmatter или первого #-заголовка."""
    if meta.get("title"):
        return str(meta["title"])

    m = _HEADING_RE.search(content)
    if m:
        return m.group(2).strip()

    return ""


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Рекурсивный загрузчик .md файлов из директории.

    Парсит Hugo Markdown: YAML frontmatter + структура заголовков.
    Пропускает пустые файлы и _index.md с менее чем 50 символами полезного текста.

    Usage::

        loader = DocumentLoader("./podkop-wiki/content")
        docs = loader.load()
        print(f"Загружено {len(docs)} документов")
    """

    def __init__(self, docs_path: str | Path) -> None:
        """
        Args:
            docs_path: Путь к корневой директории документов.
        """
        self.docs_path = Path(docs_path).resolve()

    def load(self) -> list[Document]:
        """Рекурсивно загрузить все .md файлы.

        Returns:
            Список Document, отсортированный по weight, затем по source.
        """
        docs: list[Document] = []

        for md_file in sorted(self.docs_path.rglob("*.md")):
            doc = self._load_file(md_file)
            if doc is not None:
                docs.append(doc)

        # Сортировка: сначала по weight (0 если не задан), затем по source
        docs.sort(key=lambda d: (d.weight, d.source))
        return docs

    def _load_file(self, file_path: Path) -> Optional[Document]:
        """Загрузить и распарсить один .md файл.

        Returns:
            Document или None если файл нужно пропустить.
        """
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        if not raw.strip():
            return None

        meta, content = _parse_frontmatter(raw)

        # Фильтрация _index.md с малым количеством полезного текста
        is_index = file_path.name == "_index.md"
        useful_text = content.strip()
        if is_index and len(useful_text) < 50:
            return None

        sections = _parse_sections(content)
        title = _extract_title(meta, content)
        weight = int(meta.get("weight", 0)) if meta.get("weight") is not None else 0

        # Относительный путь от корня docs
        try:
            source = str(file_path.relative_to(self.docs_path))
        except ValueError:
            source = str(file_path)

        return Document(
            content=content,
            source=source,
            file_path=str(file_path),
            title=title,
            weight=weight,
            frontmatter=meta,
            sections=sections,
        )
