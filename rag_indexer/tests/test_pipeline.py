"""Минимальные тесты пайплайна RAG-индексации.

Покрывают:
  - DocumentLoader: парсинг frontmatter и секций
  - ChunkingStrategy: все 4 стратегии, chunk_id детерминированность
  - EmbeddingProvider: LocalRandomEmbedder (без API)
  - IndexStore: store_batch + search + get_stats + clear_strategy
  - IndexingPipeline: полный прогон с синтетическими документами
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Путь к пакету src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.loader import DocumentLoader, Document, _parse_frontmatter, _parse_sections
from src.chunking.strategies import (
    Chunk,
    FixedSizeChunker,
    FixedOverlapChunker,
    StructuralChunker,
    STRATEGIES,
    estimate_tokens,
)
from src.embedding.provider import EmbeddingProvider, LocalRandomEmbedder
from src.storage.index_store import IndexStore
from src.pipeline import IndexingPipeline


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
---
title: Тест
weight: 5
toc: true
---

## Раздел первый

Это первый раздел документа.
Содержит несколько предложений.

Второй параграф первого раздела.

## Раздел второй

Это второй раздел.
Он тоже содержит текст.

### Подраздел 2.1

Текст подраздела.
"""

SAMPLE_MD_NO_FM = """\
## Только заголовок

Документ без frontmatter.
"""


@pytest.fixture
def sample_doc() -> Document:
    """Синтетический документ с frontmatter."""
    meta, content = _parse_frontmatter(SAMPLE_MD)
    sections = _parse_sections(content)
    return Document(
        content=content,
        source="test/sample.md",
        file_path="/tmp/sample.md",
        title=meta.get("title", ""),
        weight=meta.get("weight", 0),
        frontmatter=meta,
        sections=sections,
    )


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Временный SQLite-файл."""
    return tmp_path / "test_index.db"


@pytest.fixture
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Временная директория с .md файлами."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "page1.md").write_text(SAMPLE_MD, encoding="utf-8")
    (docs / "page2.md").write_text(SAMPLE_MD_NO_FM, encoding="utf-8")

    # _index.md без полезного содержимого — должен быть пропущен
    (docs / "_index.md").write_text("---\ntitle: Index\n---\n\n", encoding="utf-8")

    return docs


# ---------------------------------------------------------------------------
# Тесты: estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_simple(self):
        # 10 слов * 1.4 ≈ 14
        result = estimate_tokens("one two three four five six seven eight nine ten")
        assert result > 10

    def test_with_specials(self):
        # Спец-символы добавляют немного
        r1 = estimate_tokens("hello world")
        r2 = estimate_tokens("hello, world!")
        assert r2 >= r1


# ---------------------------------------------------------------------------
# Тесты: _parse_frontmatter
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_with_frontmatter(self):
        meta, content = _parse_frontmatter(SAMPLE_MD)
        assert meta["title"] == "Тест"
        assert meta["weight"] == 5
        assert "## Раздел первый" in content

    def test_without_frontmatter(self):
        meta, content = _parse_frontmatter(SAMPLE_MD_NO_FM)
        assert meta == {}
        assert "## Только заголовок" in content

    def test_invalid_yaml(self):
        text = "---\n: invalid: yaml: :\n---\n\ncontent"
        meta, content = _parse_frontmatter(text)
        assert isinstance(meta, dict)  # Не должен падать


# ---------------------------------------------------------------------------
# Тесты: _parse_sections
# ---------------------------------------------------------------------------

class TestParseSections:
    def test_sections_found(self, sample_doc):
        sections = sample_doc.sections
        assert len(sections) >= 2
        # Первый раздел уровня 2
        level, title, start, end = sections[0]
        assert level == 2
        assert "первый" in title.lower()

    def test_section_positions(self, sample_doc):
        for level, title, start, end in sample_doc.sections:
            assert 0 <= start < end <= len(sample_doc.content)


# ---------------------------------------------------------------------------
# Тесты: DocumentLoader
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    def test_load_files(self, tmp_docs_dir):
        loader = DocumentLoader(tmp_docs_dir)
        docs = loader.load()
        # page1.md и page2.md должны загрузиться, _index.md — нет
        assert len(docs) == 2

    def test_frontmatter_parsed(self, tmp_docs_dir):
        loader = DocumentLoader(tmp_docs_dir)
        docs = loader.load()
        with_fm = [d for d in docs if d.title == "Тест"]
        assert len(with_fm) == 1
        assert with_fm[0].weight == 5

    def test_source_relative(self, tmp_docs_dir):
        loader = DocumentLoader(tmp_docs_dir)
        docs = loader.load()
        for d in docs:
            assert not os.path.isabs(d.source), f"source должен быть относительным: {d.source}"

    def test_empty_dir(self, tmp_path):
        loader = DocumentLoader(tmp_path)
        docs = loader.load()
        assert docs == []


# ---------------------------------------------------------------------------
# Тесты: FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_basic(self, sample_doc):
        chunker = FixedSizeChunker(max_tokens=500, strategy_name="fixed_500")
        chunks = chunker.chunk_document(sample_doc)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.token_count <= 600  # с небольшим запасом

    def test_chunk_fields(self, sample_doc):
        chunker = FixedSizeChunker(max_tokens=500, strategy_name="fixed_500")
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert c.chunk_id
            assert c.text
            assert c.source == sample_doc.source
            assert c.strategy == "fixed_500"

    def test_chunk_id_deterministic(self, sample_doc):
        chunker = FixedSizeChunker(max_tokens=500, strategy_name="fixed_500")
        chunks1 = chunker.chunk_document(sample_doc)
        chunks2 = chunker.chunk_document(sample_doc)
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2, "chunk_id должен быть детерминированным"

    def test_no_empty_chunks(self, sample_doc):
        chunker = FixedSizeChunker(max_tokens=500, strategy_name="fixed_500")
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert c.text.strip(), "Чанк не должен быть пустым"


# ---------------------------------------------------------------------------
# Тесты: FixedOverlapChunker
# ---------------------------------------------------------------------------

class TestFixedOverlapChunker:
    def test_overlap_applied(self, sample_doc):
        # Создадим документ с заведомо несколькими чанками
        chunker = FixedOverlapChunker(max_tokens=100, overlap_tokens=30, strategy_name="overlap_100_30")
        chunks = chunker.chunk_document(sample_doc)
        assert len(chunks) >= 1

    def test_strategy_name(self, sample_doc):
        chunker = FixedOverlapChunker(max_tokens=600, overlap_tokens=100, strategy_name="overlap_600_100")
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert c.strategy == "overlap_600_100"


# ---------------------------------------------------------------------------
# Тесты: StructuralChunker
# ---------------------------------------------------------------------------

class TestStructuralChunker:
    def test_basic(self, sample_doc):
        chunker = StructuralChunker(max_tokens=1000, min_tokens=50)
        chunks = chunker.chunk_document(sample_doc)
        assert len(chunks) >= 1

    def test_strategy_name(self, sample_doc):
        chunker = StructuralChunker()
        chunks = chunker.chunk_document(sample_doc)
        for c in chunks:
            assert c.strategy == "structural"

    def test_sections_used(self, sample_doc):
        chunker = StructuralChunker(max_tokens=1000, min_tokens=10)
        chunks = chunker.chunk_document(sample_doc)
        # Хотя бы один чанк должен иметь секцию
        sections = [c.section for c in chunks if c.section]
        assert len(sections) > 0


# ---------------------------------------------------------------------------
# Тесты: все 4 стратегии из реестра
# ---------------------------------------------------------------------------

class TestAllStrategies:
    def test_all_strategies_produce_chunks(self, sample_doc):
        for name, strategy in STRATEGIES.items():
            chunks = strategy.chunk_document(sample_doc)
            assert len(chunks) >= 1, f"Стратегия {name} не дала чанков"

    def test_chunk_all(self, sample_doc):
        for name, strategy in STRATEGIES.items():
            docs = [sample_doc]
            chunks = strategy.chunk_all(docs)
            assert all(isinstance(c, Chunk) for c in chunks)


# ---------------------------------------------------------------------------
# Тесты: LocalRandomEmbedder
# ---------------------------------------------------------------------------

class TestLocalRandomEmbedder:
    def test_dimension(self):
        emb = LocalRandomEmbedder(dimension=128)
        vecs = emb.embed_texts(["hello"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 128

    def test_deterministic(self):
        emb = LocalRandomEmbedder(dimension=64)
        v1 = emb.embed_texts(["одинаковый текст"])[0]
        v2 = emb.embed_texts(["одинаковый текст"])[0]
        assert v1 == v2

    def test_different_texts(self):
        emb = LocalRandomEmbedder(dimension=64)
        v1 = emb.embed_texts(["текст один"])[0]
        v2 = emb.embed_texts(["текст два"])[0]
        assert v1 != v2

    def test_normalized(self):
        import math
        emb = LocalRandomEmbedder(dimension=128)
        vecs = emb.embed_texts(["проверка нормализации"])
        vec = vecs[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5, f"Вектор не нормализован: {norm}"

    def test_empty_list(self):
        emb = LocalRandomEmbedder()
        result = emb.embed_texts([])
        assert result == []


# ---------------------------------------------------------------------------
# Тесты: EmbeddingProvider.create()
# ---------------------------------------------------------------------------

class TestEmbeddingProviderFactory:
    def test_local_provider(self):
        p = EmbeddingProvider.create("local")
        assert isinstance(p, LocalRandomEmbedder)

    def test_qwen_fallback_without_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        p = EmbeddingProvider.create("qwen")
        assert isinstance(p, LocalRandomEmbedder)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Неизвестный провайдер"):
            EmbeddingProvider.create("unknown_xyz")


# ---------------------------------------------------------------------------
# Тесты: IndexStore
# ---------------------------------------------------------------------------

class TestIndexStore:
    def _make_chunks(self, count: int = 3, strategy: str = "fixed_500") -> list[Chunk]:
        """Создать синтетические чанки."""
        chunks = []
        for i in range(count):
            text = f"Тестовый текст чанка номер {i}. Содержит полезную информацию."
            chunks.append(Chunk(
                chunk_id=f"test_{strategy}_{i:04d}",
                text=text,
                source=f"doc{i}.md",
                file=f"doc{i}.md",
                section=f"Раздел {i}",
                doc_title=f"Документ {i}",
                chunk_index=i,
                token_count=estimate_tokens(text),
                strategy=strategy,
            ))
        return chunks

    def test_store_and_stats(self, tmp_db):
        chunks = self._make_chunks(5, "fixed_500")
        emb = LocalRandomEmbedder(dimension=64)
        embeddings = emb.embed_texts([c.text for c in chunks])

        with IndexStore(tmp_db) as store:
            store.store_batch(chunks, embeddings, model_name="local-random")
            stats = store.get_stats("fixed_500")

        assert stats["chunks"] == 5
        assert stats["avg_tokens"] > 0

    def test_idempotent_insert(self, tmp_db):
        """Повторный запуск не дублирует данные."""
        chunks = self._make_chunks(3, "fixed_500")
        emb = LocalRandomEmbedder(dimension=64)
        embeddings = emb.embed_texts([c.text for c in chunks])

        with IndexStore(tmp_db) as store:
            store.store_batch(chunks, embeddings, "local-random")
            store.store_batch(chunks, embeddings, "local-random")  # повтор
            stats = store.get_stats("fixed_500")

        assert stats["chunks"] == 3, "INSERT OR REPLACE должен предотвращать дубли"

    def test_search_returns_results(self, tmp_db):
        chunks = self._make_chunks(5, "fixed_500")
        emb = LocalRandomEmbedder(dimension=64)
        embeddings = emb.embed_texts([c.text for c in chunks])

        with IndexStore(tmp_db) as store:
            store.store_batch(chunks, embeddings, "local-random")
            query_vec = emb.embed_texts(["тестовый запрос"])[0]
            results = store.search(query_vec, top_k=3)

        assert len(results) == 3
        assert results[0].rank == 1
        # Результаты отсортированы по убыванию score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_strategy_filter(self, tmp_db):
        chunks_a = self._make_chunks(3, "fixed_500")
        chunks_b = self._make_chunks(3, "fixed_1000")
        emb = LocalRandomEmbedder(dimension=64)

        with IndexStore(tmp_db) as store:
            store.store_batch(chunks_a, emb.embed_texts([c.text for c in chunks_a]), "local")
            store.store_batch(chunks_b, emb.embed_texts([c.text for c in chunks_b]), "local")

            q = emb.embed_texts(["запрос"])[0]
            results_a = store.search(q, strategy="fixed_500", top_k=10)
            results_b = store.search(q, strategy="fixed_1000", top_k=10)

        assert all(r.chunk.strategy == "fixed_500" for r in results_a)
        assert all(r.chunk.strategy == "fixed_1000" for r in results_b)

    def test_clear_strategy(self, tmp_db):
        chunks = self._make_chunks(4, "fixed_500")
        emb = LocalRandomEmbedder(dimension=64)

        with IndexStore(tmp_db) as store:
            store.store_batch(chunks, emb.embed_texts([c.text for c in chunks]), "local")
            deleted = store.clear_strategy("fixed_500")
            stats = store.get_stats("fixed_500")

        assert deleted == 4
        assert stats["chunks"] == 0

    def test_get_all_strategies(self, tmp_db):
        emb = LocalRandomEmbedder(dimension=64)
        with IndexStore(tmp_db) as store:
            for strat in ["fixed_500", "fixed_1000"]:
                chunks = self._make_chunks(2, strat)
                store.store_batch(chunks, emb.embed_texts([c.text for c in chunks]), "local")
            strategies = store.get_all_strategies()

        assert "fixed_500" in strategies
        assert "fixed_1000" in strategies


# ---------------------------------------------------------------------------
# Тесты: IndexingPipeline
# ---------------------------------------------------------------------------

class TestIndexingPipeline:
    def test_full_run(self, tmp_db, tmp_docs_dir):
        emb = LocalRandomEmbedder(dimension=64)
        pipeline = IndexingPipeline(
            docs_path=tmp_docs_dir,
            db_path=tmp_db,
            embedding_provider=emb,
            strategies=[STRATEGIES["fixed_500"]],
        )
        pipeline.run("fixed_500")

        with IndexStore(tmp_db) as store:
            stats = store.get_stats("fixed_500")

        assert stats["chunks"] > 0

    def test_all_strategies(self, tmp_db, tmp_docs_dir):
        emb = LocalRandomEmbedder(dimension=64)
        pipeline = IndexingPipeline(
            docs_path=tmp_docs_dir,
            db_path=tmp_db,
            embedding_provider=emb,
        )
        pipeline.run()

        with IndexStore(tmp_db) as store:
            strategies = store.get_all_strategies()

        assert len(strategies) == 4

    def test_compare_strategies(self, tmp_db, tmp_docs_dir, capsys):
        emb = LocalRandomEmbedder(dimension=64)
        pipeline = IndexingPipeline(
            docs_path=tmp_docs_dir,
            db_path=tmp_db,
            embedding_provider=emb,
        )
        pipeline.run()
        stats = pipeline.compare_strategies()

        assert len(stats) == 4
        captured = capsys.readouterr()
        assert "Strategy" in captured.out
        assert "fixed_500" in captured.out

    def test_idempotent_run(self, tmp_db, tmp_docs_dir):
        """Повторный запуск не дублирует данные."""
        emb = LocalRandomEmbedder(dimension=64)
        pipeline = IndexingPipeline(
            docs_path=tmp_docs_dir,
            db_path=tmp_db,
            embedding_provider=emb,
            strategies=[STRATEGIES["fixed_500"]],
        )

        pipeline.run("fixed_500")
        with IndexStore(tmp_db) as store:
            count1 = store.get_stats("fixed_500")["chunks"]

        pipeline._documents = None  # сбросить кэш
        pipeline.run("fixed_500")
        with IndexStore(tmp_db) as store:
            count2 = store.get_stats("fixed_500")["chunks"]

        assert count1 == count2, "Повторный запуск не должен дублировать чанки"
