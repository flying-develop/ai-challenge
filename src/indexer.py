"""IndexManager: фасад для индексации документации из разных источников.

Объединяет DocumentSource (Local/URL/GitHub) с IndexingPipeline.
При переиндексации старый индекс ПОЛНОСТЬЮ заменяется новым.

Использование:
    manager = IndexManager(db_path="./output/index.db", embedding_provider=...)
    manager.reindex("path", "./podkop-wiki/content")
    manager.reindex("url", "https://laravel.com/docs/13.x")
    manager.reindex("github", "https://github.com/user/repo")
    print(manager.status())
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_indexer.src.embedding.provider import EmbeddingProvider
from rag_indexer.src.storage.index_store import IndexStore
from rag_indexer.src.chunking.strategies import STRATEGIES, ChunkingStrategy
from rag_indexer.src.loader import DocumentLoader, Document

from src.sources.source import RawDocument, DocumentSource
from src.sources.local_source import LocalMarkdownSource
from src.sources.url_source import URLDocSource
from src.sources.github_source import GitHubRepoSource


_BATCH_SIZE = 25


class IndexManager:
    """Менеджер индексирования с поддержкой разных источников.

    Один вызов reindex() полностью заменяет предыдущий индекс.
    Сохраняет метаданные источника (тип, аргумент, дата) в index_meta.

    Args:
        db_path:            Путь к SQLite-базе индекса.
        embedding_provider: Провайдер эмбеддингов (Qwen или Local).
        strategies:         Список стратегий нарезки (по умолчанию все).
    """

    def __init__(
        self,
        db_path: str | Path,
        embedding_provider: EmbeddingProvider,
        strategies: Optional[list[ChunkingStrategy]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self.strategies = strategies or list(STRATEGIES.values())

    def reindex(self, source_type: str, source_arg: str) -> None:
        """Переиндексировать документацию из нового источника.

        Старый индекс полностью удаляется перед записью нового.

        Args:
            source_type: Тип источника: "path", "url", "github".
            source_arg:  Аргумент источника: путь, URL или GitHub URL.

        Raises:
            ValueError: Если source_type не поддерживается.
        """
        source = self._create_source(source_type, source_arg)
        print(f"\n[IndexManager] Источник: {source.source_description}")

        # Загружаем сырые документы
        raw_docs = source.fetch()
        if not raw_docs:
            print("[IndexManager] Нет документов для индексации")
            return

        print(f"[IndexManager] Получено {len(raw_docs)} документов")

        # Конвертируем RawDocument → Document для чанкера
        documents = _raw_to_documents(raw_docs)

        # Открываем хранилище и очищаем старый индекс
        with IndexStore(self.db_path) as store:
            print(f"[IndexManager] Очищаю старый индекс...")
            deleted = store.clear_all()
            print(f"  Удалено {deleted} чанков")

            # Индексируем каждую стратегию
            for strategy in self.strategies:
                self._index_strategy(documents, strategy, store)

            # Сохраняем метаданные источника
            store.set_meta("source_type", source_type)
            store.set_meta("source_arg", source_arg)
            store.set_meta("source_description", source.source_description)
            store.set_meta("indexed_at", time.strftime("%Y-%m-%d %H:%M:%S"))
            store.set_meta("doc_count", str(len(raw_docs)))

        print(f"\n[IndexManager] Индексация завершена: {source.source_description}")

    def status(self) -> str:
        """Получить статус текущего индекса.

        Returns:
            Человекочитаемая строка со статусом индекса.
        """
        if not self.db_path.exists():
            return "Индекс не создан. Запустите /index path|url|github ..."

        with IndexStore(self.db_path) as store:
            stats = store.get_stats()
            source_type = store.get_meta("source_type") or "?"
            source_arg = store.get_meta("source_arg") or "?"
            source_desc = store.get_meta("source_description") or "?"
            indexed_at = store.get_meta("indexed_at") or "?"
            doc_count = store.get_meta("doc_count") or "?"

        lines = [
            "─" * 50,
            "  Статус индекса:",
            f"  Источник:   {source_desc}",
            f"  Обновлён:   {indexed_at}",
            f"  Документов: {doc_count}",
            f"  Чанков:     {stats['chunks']}",
            f"  Стратегий:  {len(store.get_all_strategies() if False else [])}",
            "─" * 50,
        ]
        # Отдельно получаем список стратегий
        with IndexStore(self.db_path) as store:
            strats = store.get_all_strategies()
        lines[-1] = f"  Стратегий:  {len(strats)} ({', '.join(strats)})"
        lines.append("─" * 50)
        return "\n".join(lines)

    def _index_strategy(
        self,
        documents: list[Document],
        strategy: ChunkingStrategy,
        store: IndexStore,
    ) -> None:
        """Индексировать документы одной стратегией.

        Args:
            documents: Список Document-ов.
            strategy:  Стратегия нарезки чанков.
            store:     Открытый IndexStore.
        """
        print(f"\n  [Strategy: {strategy.name}]")

        # Нарезка
        chunks = strategy.chunk_all(documents)
        print(f"    Чанков: {len(chunks)}")

        if not chunks:
            return

        # Эмбеддинги батчами
        all_embeddings = []
        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start: batch_start + _BATCH_SIZE]
            texts = [c.text for c in batch]
            embeddings = self.embedding_provider.embed_texts(texts)
            all_embeddings.extend(embeddings)

        # Сохранение
        store.store_batch(chunks, all_embeddings,
                          model_name=self.embedding_provider.model_name)
        print(f"    Сохранено: {len(chunks)} чанков")


def _raw_to_documents(raw_docs: list[RawDocument]) -> list[Document]:
    """Конвертировать RawDocument-ы в Document-ы для чанкера.

    Args:
        raw_docs: Список RawDocument от источника.

    Returns:
        Список Document для передачи в ChunkingStrategy.
    """
    from rag_indexer.src.loader import Document

    documents = []
    for raw in raw_docs:
        doc = Document(
            content=raw.content,
            source=raw.source_path,
            file_path=raw.source_path,
            title=raw.title,
            weight=0,
            frontmatter={},
            sections=[],
        )
        documents.append(doc)
    return documents


def _create_source_from_env() -> Optional[tuple[str, str]]:
    """Получить конфигурацию источника из переменных окружения (для авто-запуска)."""
    source_type = os.environ.get("INDEX_SOURCE_TYPE", "")
    source_arg = os.environ.get("INDEX_SOURCE_ARG", "")
    if source_type and source_arg:
        return source_type, source_arg
    return None


def create_index_manager(db_path: str | Path) -> IndexManager:
    """Создать IndexManager с настройками из .env.

    Args:
        db_path: Путь к SQLite-базе.

    Returns:
        Сконфигурированный IndexManager.
    """
    embedding_provider = EmbeddingProvider.create("qwen")
    return IndexManager(
        db_path=db_path,
        embedding_provider=embedding_provider,
    )


# Фабрика источников (используется в IndexManager и AdminCLI)
def _create_source(source_type: str, source_arg: str) -> DocumentSource:
    """Создать DocumentSource по типу и аргументу."""
    if source_type == "path":
        return LocalMarkdownSource(source_arg)
    elif source_type == "url":
        return URLDocSource(source_arg)
    elif source_type == "github":
        return GitHubRepoSource(source_arg)
    else:
        raise ValueError(f"Неизвестный тип источника: {source_type!r}. "
                         f"Допустимые: path, url, github")


# Патчим IndexManager._create_source как метод
IndexManager._create_source = lambda self, t, a: _create_source(t, a)
