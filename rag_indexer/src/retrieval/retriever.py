"""RAG retrievers: Vector, BM25, Hybrid (RRF)."""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..storage.index_store import IndexStore, SearchResult as StoreSearchResult
from ..embedding.provider import EmbeddingProvider


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source: str
    section: str
    doc_title: str
    strategy: str      # chunking strategy (fixed_500, structural, ...)
    retriever: str     # "vector" | "bm25" | "hybrid"


class RetrievalStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


class VectorRetriever(RetrievalStrategy):
    def __init__(self, store: IndexStore, embedder: EmbeddingProvider, strategy_filter: Optional[str] = None):
        self._store = store
        self._embedder = embedder
        self._strategy_filter = strategy_filter

    @property
    def name(self) -> str:
        return "vector"

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        vector = self._embedder.embed_texts([query])[0]
        store_results: list[StoreSearchResult] = self._store.search(
            vector, self._strategy_filter, top_k
        )
        return [
            RetrievalResult(
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text,
                score=r.score,
                source=r.chunk.source,
                section=r.chunk.section,
                doc_title=r.chunk.doc_title,
                strategy=r.chunk.strategy,
                retriever="vector",
            )
            for r in store_results
        ]


_STOP_WORDS = {
    "и", "в", "на", "с", "по", "для", "что", "как", "это", "или",
    "не", "из", "к", "а", "о", "от", "но", "да", "же", "бы",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "and", "or", "for", "with", "at", "by",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r'[а-яёa-z0-9]+', text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


class BM25Retriever(RetrievalStrategy):
    def __init__(
        self,
        store: IndexStore,
        strategy_filter: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._store = store
        self._strategy_filter = strategy_filter
        self._k1 = k1
        self._b = b
        self._index_built = False
        self._chunks: list = []
        self._tokenized: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._df: dict[str, int] = {}
        self._N: int = 0

    @property
    def name(self) -> str:
        return "bm25"

    def _build_index(self) -> None:
        if self._index_built:
            return
        self._chunks = self._store.get_all_chunks(self._strategy_filter)
        self._N = len(self._chunks)
        self._tokenized = [_tokenize(c.text) for c in self._chunks]
        self._doc_lengths = [len(t) for t in self._tokenized]
        self._avg_dl = sum(self._doc_lengths) / max(self._N, 1)
        self._df = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._index_built = True

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        return math.log((self._N - n + 0.5) / (n + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self._build_index()
        query_terms = _tokenize(query)
        if not query_terms or not self._chunks:
            return []

        scores: list[tuple[float, int]] = []
        k1, b, avg_dl = self._k1, self._b, self._avg_dl
        for idx, tokens in enumerate(self._tokenized):
            dl = self._doc_lengths[idx]
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for term in query_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self._idf(term)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            c = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=score,
                    source=c.source,
                    section=c.section or "",
                    doc_title=c.doc_title or "",
                    strategy=c.strategy,
                    retriever="bm25",
                )
            )
        return results


class HybridRetriever(RetrievalStrategy):
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._vw = vector_weight
        self._bw = bm25_weight
        self._rrf_k = rrf_k

    @property
    def name(self) -> str:
        return "hybrid"

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        fetch_k = top_k * 3
        vector_results = self._vector.search(query, top_k=fetch_k)
        bm25_results = self._bm25.search(query, top_k=fetch_k)

        k = self._rrf_k
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        for rank, r in enumerate(vector_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + self._vw / (k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        for rank, r in enumerate(bm25_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + self._bw / (k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        results = []
        for cid in sorted_ids[:top_k]:
            r = result_map[cid]
            results.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=rrf_scores[cid],
                    source=r.source,
                    section=r.section,
                    doc_title=r.doc_title,
                    strategy=r.strategy,
                    retriever="hybrid",
                )
            )
        return results
