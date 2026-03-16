"""SQLite-хранилище для чанков, эмбеддингов и метаданных индекса.

Схема:
  chunks     — текст + метаданные чанков
  embeddings — numpy float32 векторы (BLOB)
  index_meta — произвольные ключи/значения (статистика, версии)

Векторный поиск: cosine similarity через numpy в памяти (OK до ~100K чанков).
INSERT OR REPLACE — повторный запуск не дублирует данные (идемпотентность).
PRAGMA WAL для параллельных читателей.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .._math import vector_to_blob, blob_to_vector, cosine_similarities, argsort_desc
from ..chunking.strategies import Chunk


# ---------------------------------------------------------------------------
# Dataclass результата поиска
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Результат поиска по косинусному сходству.

    Attributes:
        chunk:       Найденный чанк.
        score:       Косинусное сходство (0..1, чем выше — тем релевантнее).
        rank:        Позиция в результатах (начиная с 1).
    """

    chunk: Chunk
    score: float
    rank: int


# ---------------------------------------------------------------------------
# IndexStore
# ---------------------------------------------------------------------------

class IndexStore:
    """SQLite-хранилище для RAG-индекса.

    Поддерживает батч-запись, векторный поиск и статистику.
    Использует WAL-режим для производительности.

    Usage::

        store = IndexStore("./output/index.db")
        store.store_batch(chunks, embeddings)
        results = store.search(query_vector, strategy="fixed_500", top_k=5)
        stats = store.get_stats()
        store.close()
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id   TEXT PRIMARY KEY,
        text       TEXT NOT NULL,
        source     TEXT NOT NULL,
        file       TEXT NOT NULL,
        section    TEXT NOT NULL,
        doc_title  TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        token_count INTEGER NOT NULL,
        strategy   TEXT NOT NULL,
        metadata   TEXT DEFAULT '{}',
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id  TEXT PRIMARY KEY,
        vector    BLOB NOT NULL,
        model     TEXT NOT NULL,
        dimension INTEGER NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS index_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);
    CREATE INDEX IF NOT EXISTS idx_chunks_source   ON chunks(source);
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Args:
            db_path: Путь к SQLite-файлу (создаётся если не существует).
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._create_schema()

    # ---------------------------------------------------------------------------
    # Инициализация
    # ---------------------------------------------------------------------------

    def _apply_pragmas(self) -> None:
        """Применить PRAGMA для производительности."""
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_schema(self) -> None:
        """Создать таблицы и индексы."""
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    # ---------------------------------------------------------------------------
    # Запись
    # ---------------------------------------------------------------------------

    def store_batch(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        model_name: str = "unknown",
    ) -> None:
        """Батч-сохранение чанков и эмбеддингов (одна транзакция).

        Использует INSERT OR REPLACE — повторный запуск идемпотентен.

        Args:
            chunks:     Список чанков.
            embeddings: Список векторов (должен совпадать по длине).
            model_name: Название модели эмбеддингов.

        Raises:
            ValueError: Если длины списков не совпадают.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Несоответствие длин: chunks={len(chunks)}, embeddings={len(embeddings)}"
            )

        now = time.time()
        chunk_rows = []
        embedding_rows = []

        for chunk, emb in zip(chunks, embeddings):
            dimension = len(emb)

            chunk_rows.append((
                chunk.chunk_id,
                chunk.text,
                chunk.source,
                chunk.file,
                chunk.section,
                chunk.doc_title,
                chunk.chunk_index,
                chunk.token_count,
                chunk.strategy,
                json.dumps(chunk.metadata, ensure_ascii=False),
                now,
            ))

            embedding_rows.append((
                chunk.chunk_id,
                vector_to_blob(emb),
                model_name,
                dimension,
            ))

        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, text, source, file, section, doc_title,
                 chunk_index, token_count, strategy, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (chunk_id, vector, model, dimension)
                VALUES (?, ?, ?, ?)
                """,
                embedding_rows,
            )

    # ---------------------------------------------------------------------------
    # Поиск
    # ---------------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        strategy: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Векторный поиск по косинусному сходству (numpy в памяти).

        Args:
            query_vector: Вектор запроса (должен совпадать по размерности).
            strategy:     Фильтр по стратегии (None = все стратегии).
            top_k:        Количество результатов.

        Returns:
            Список SearchResult, отсортированных по убыванию score.
        """
        # Загрузить чанки + векторы
        if strategy:
            rows = self._conn.execute(
                """
                SELECT c.chunk_id, c.text, c.source, c.file, c.section,
                       c.doc_title, c.chunk_index, c.token_count, c.strategy,
                       c.metadata, e.vector, e.dimension
                FROM chunks c
                JOIN embeddings e ON c.chunk_id = e.chunk_id
                WHERE c.strategy = ?
                """,
                (strategy,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT c.chunk_id, c.text, c.source, c.file, c.section,
                       c.doc_title, c.chunk_index, c.token_count, c.strategy,
                       c.metadata, e.vector, e.dimension
                FROM chunks c
                JOIN embeddings e ON c.chunk_id = e.chunk_id
                """,
            ).fetchall()

        if not rows:
            return []

        # Матрица векторов для батч-вычисления cosine similarity
        matrix = [blob_to_vector(row["vector"]) for row in rows]
        scores = cosine_similarities(query_vector, matrix)

        # Top-K
        top_indices = argsort_desc(scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = rows[idx]
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                text=row["text"],
                source=row["source"],
                file=row["file"],
                section=row["section"],
                doc_title=row["doc_title"],
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                strategy=row["strategy"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            results.append(SearchResult(chunk=chunk, score=float(scores[idx]), rank=rank))

        return results

    # ---------------------------------------------------------------------------
    # Статистика
    # ---------------------------------------------------------------------------

    def get_stats(self, strategy: Optional[str] = None) -> dict:
        """Агрегированная статистика по индексу.

        Args:
            strategy: Фильтр по стратегии (None = все).

        Returns:
            dict с полями: chunks, avg_tokens, min_tokens, max_tokens,
                           total_tokens, documents, sections, strategy.
        """
        where = "WHERE strategy = ?" if strategy else ""
        params = (strategy,) if strategy else ()

        row = self._conn.execute(
            f"""
            SELECT
                COUNT(*)           AS chunks,
                AVG(token_count)   AS avg_tokens,
                MIN(token_count)   AS min_tokens,
                MAX(token_count)   AS max_tokens,
                SUM(token_count)   AS total_tokens,
                COUNT(DISTINCT source) AS documents,
                COUNT(DISTINCT section) AS sections
            FROM chunks
            {where}
            """,
            params,
        ).fetchone()

        return {
            "strategy": strategy or "all",
            "chunks": row["chunks"] or 0,
            "avg_tokens": round(row["avg_tokens"] or 0),
            "min_tokens": row["min_tokens"] or 0,
            "max_tokens": row["max_tokens"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "documents": row["documents"] or 0,
            "sections": row["sections"] or 0,
        }

    def get_all_strategies(self) -> list[str]:
        """Вернуть список всех стратегий в индексе."""
        rows = self._conn.execute(
            "SELECT DISTINCT strategy FROM chunks ORDER BY strategy"
        ).fetchall()
        return [r["strategy"] for r in rows]

    # ---------------------------------------------------------------------------
    # Управление данными
    # ---------------------------------------------------------------------------

    def clear_strategy(self, strategy: str) -> int:
        """Удалить все чанки (и эмбеддинги) для стратегии.

        Args:
            strategy: Имя стратегии.

        Returns:
            Количество удалённых чанков.
        """
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM chunks WHERE strategy = ?", (strategy,)
            )
        return cursor.rowcount

    def set_meta(self, key: str, value: str) -> None:
        """Сохранить произвольное значение в index_meta."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_meta(self, key: str) -> Optional[str]:
        """Получить значение из index_meta."""
        row = self._conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ---------------------------------------------------------------------------
    # Контекстный менеджер
    # ---------------------------------------------------------------------------

    def close(self) -> None:
        """Закрыть соединение с БД."""
        self._conn.close()

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
