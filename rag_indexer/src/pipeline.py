"""Оркестратор пайплайна индексации документов.

Связывает все компоненты:
  DocumentLoader → ChunkingStrategy → EmbeddingProvider → IndexStore

Поддерживает:
  - Индексацию одной или всех стратегий
  - Батч-обработку эмбеддингов
  - Прогресс в stdout
  - Сравнение стратегий в виде ASCII-таблицы
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .chunking.strategies import ChunkingStrategy, Chunk, STRATEGIES
from .embedding.provider import EmbeddingProvider
from .loader import DocumentLoader, Document
from .storage.index_store import IndexStore

_BATCH_SIZE = 25  # Батч для эмбеддингов (ограничение DashScope)


# ---------------------------------------------------------------------------
# IndexingPipeline
# ---------------------------------------------------------------------------

class IndexingPipeline:
    """Оркестратор пайплайна индексации RAG.

    Usage::

        pipeline = IndexingPipeline(
            docs_path="./podkop-wiki/content",
            db_path="./output/index.db",
            embedding_provider=EmbeddingProvider.create("qwen"),
            strategies=list(STRATEGIES.values()),
        )
        pipeline.run()                    # все стратегии
        pipeline.run("fixed_500")         # одна стратегия
        pipeline.compare_strategies()     # ASCII-таблица

    Args:
        docs_path:           Путь к директории с .md файлами.
        db_path:             Путь к SQLite-базе.
        embedding_provider:  Провайдер эмбеддингов.
        strategies:          Список стратегий (по умолчанию все 4).
    """

    def __init__(
        self,
        docs_path: str | Path,
        db_path: str | Path,
        embedding_provider: EmbeddingProvider,
        strategies: Optional[list[ChunkingStrategy]] = None,
    ) -> None:
        self.docs_path = Path(docs_path)
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self.strategies = strategies or list(STRATEGIES.values())
        self._documents: Optional[list[Document]] = None

    # ---------------------------------------------------------------------------
    # Загрузка документов (кэш)
    # ---------------------------------------------------------------------------

    def _load_documents(self) -> list[Document]:
        """Загрузить документы (кэшируется внутри пайплайна)."""
        if self._documents is None:
            loader = DocumentLoader(self.docs_path)
            self._documents = loader.load()
            print(f"[Loader] Загружено {len(self._documents)} документов из {self.docs_path}")
        return self._documents

    # ---------------------------------------------------------------------------
    # Основной метод
    # ---------------------------------------------------------------------------

    def run(self, strategy_name: Optional[str] = None) -> None:
        """Запустить пайплайн индексации.

        Args:
            strategy_name: Имя конкретной стратегии или None для всех.
        """
        docs = self._load_documents()

        if strategy_name:
            strategies_to_run = [s for s in self.strategies if s.name == strategy_name]
            if not strategies_to_run:
                available = ", ".join(s.name for s in self.strategies)
                print(f"[ERROR] Стратегия '{strategy_name}' не найдена. Доступны: {available}", file=sys.stderr)
                return
        else:
            strategies_to_run = self.strategies

        with IndexStore(self.db_path) as store:
            for strategy in strategies_to_run:
                self._run_strategy(docs, strategy, store)

        print("\n[Pipeline] Индексация завершена.")

    def _run_strategy(
        self,
        docs: list[Document],
        strategy: ChunkingStrategy,
        store: IndexStore,
    ) -> None:
        """Выполнить пайплайн для одной стратегии.

        Steps:
          1. Нарезать чанки
          2. Сгенерировать эмбеддинги батчами по 25
          3. Сохранить в SQLite
          4. Вывести статистику
        """
        print(f"\n[Strategy: {strategy.name}]")

        # Шаг 1: Нарезка
        print(f"  Нарезка чанков...", end=" ", flush=True)
        chunks = strategy.chunk_all(docs)
        print(f"{len(chunks)} чанков")

        if not chunks:
            print(f"  [WARN] Нет чанков для стратегии {strategy.name}")
            return

        # Шаг 2: Эмбеддинги батчами
        print(f"  Генерация эмбеддингов ({self.embedding_provider.model_name})...")
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start: batch_start + _BATCH_SIZE]
            texts = [c.text for c in batch]
            batch_idx = batch_start // _BATCH_SIZE + 1
            total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE
            print(f"    Батч {batch_idx}/{total_batches} ({len(texts)} текстов)...", end=" ", flush=True)

            embeddings = self.embedding_provider.embed_texts(texts)
            all_embeddings.extend(embeddings)
            print("OK")

        # Шаг 3: Сохранение
        print(f"  Сохранение в SQLite...", end=" ", flush=True)
        store.store_batch(chunks, all_embeddings, model_name=self.embedding_provider.model_name)
        print("OK")

        # Шаг 4: Статистика
        stats = store.get_stats(strategy.name)
        print(
            f"  Итог: {stats['chunks']} чанков | "
            f"avg={stats['avg_tokens']} | "
            f"min={stats['min_tokens']} | "
            f"max={stats['max_tokens']} | "
            f"total={stats['total_tokens']} токенов"
        )

    # ---------------------------------------------------------------------------
    # Сравнение стратегий
    # ---------------------------------------------------------------------------

    def compare_strategies(self) -> dict:
        """Сравнить все стратегии и вывести ASCII-таблицу.

        Returns:
            dict: {strategy_name: stats_dict}
        """
        with IndexStore(self.db_path) as store:
            available = store.get_all_strategies()

            if not available:
                print("[INFO] Индекс пуст. Сначала запустите индексацию.")
                return {}

            all_stats: dict[str, dict] = {}
            for strategy_name in available:
                all_stats[strategy_name] = store.get_stats(strategy_name)

        # ASCII-таблица
        _print_comparison_table(all_stats)
        return all_stats


# ---------------------------------------------------------------------------
# Вспомогательная функция: вывод таблицы
# ---------------------------------------------------------------------------

def _print_comparison_table(stats: dict[str, dict]) -> None:
    """Вывести ASCII-таблицу сравнения стратегий в stdout."""
    col_w = [18, 8, 11, 7, 7, 11]
    headers = ["Strategy", "Chunks", "Avg Tokens", "Min", "Max", "Total Tok"]

    def row_line(cells: list[str]) -> str:
        parts = [f" {c:<{col_w[i]}} " for i, c in enumerate(cells)]
        return "║" + "║".join(parts) + "║"

    def sep_line(left: str, mid: str, right: str, fill: str = "═") -> str:
        parts = [fill * (col_w[i] + 2) for i in range(len(col_w))]
        return left + mid.join(parts) + right

    print()
    print(sep_line("╔", "╦", "╗"))
    print(row_line(headers))
    print(sep_line("╠", "╬", "╣"))

    for strategy_name, s in stats.items():
        cells = [
            strategy_name,
            str(s["chunks"]),
            str(s["avg_tokens"]),
            str(s["min_tokens"]),
            str(s["max_tokens"]),
            str(s["total_tokens"]),
        ]
        print(row_line(cells))

    print(sep_line("╚", "╩", "╝"))
    print()
