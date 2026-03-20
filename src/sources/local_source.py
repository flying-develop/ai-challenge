"""Источник документации: локальные Markdown-файлы.

Рекурсивно обходит директорию, читает все .md файлы,
парсит Hugo frontmatter, возвращает RawDocument-ы.

Использует существующий DocumentLoader из rag_indexer.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .source import DocumentSource, RawDocument


class LocalMarkdownSource(DocumentSource):
    """Источник документации из локальной директории с .md файлами.

    Рекурсивно собирает все .md файлы, парсит frontmatter,
    возвращает RawDocument-ы с контентом и метаданными.

    Args:
        docs_path: Путь к директории с документацией.
    """

    def __init__(self, docs_path: str | Path) -> None:
        self.docs_path = Path(docs_path)

    def fetch(self) -> list[RawDocument]:
        """Загрузить все .md файлы из директории.

        Returns:
            Список RawDocument-ов, отсортированных по пути.
        """
        # Добавляем rag_indexer в sys.path если нужно
        _ensure_rag_indexer_in_path()

        from rag_indexer.src.loader import DocumentLoader

        loader = DocumentLoader(self.docs_path)
        documents = loader.load()

        raw_docs = []
        for doc in documents:
            raw_docs.append(RawDocument(
                content=doc.content,
                source_path=doc.source,
                title=doc.title,
            ))

        return raw_docs

    @property
    def source_description(self) -> str:
        return f"local:{self.docs_path}"


def _ensure_rag_indexer_in_path() -> None:
    """Добавить корень проекта в sys.path для импорта rag_indexer."""
    # Определяем корень проекта (2 уровня вверх от этого файла)
    project_root = Path(__file__).parent.parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
