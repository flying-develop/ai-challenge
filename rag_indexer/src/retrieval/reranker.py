"""Реранкеры для RAG-пайплайна.

Паттерн Strategy:
    Reranker (ABC) → ThresholdFilter, CohereReranker, OllamaReranker

Использование:
    from src.retrieval.reranker import CohereReranker, ThresholdFilter

    reranker = CohereReranker()
    results = reranker.rerank(query, candidates, top_k=5)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import warnings
from abc import ABC, abstractmethod
from dataclasses import replace

from .retriever import RetrievalResult


class Reranker(ABC):
    """Абстрактный реранкер.

    Принимает список кандидатов от retriever (обычно initial_k > top_k)
    и возвращает переранжированный список лучших top_k.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя реранкера для логирования и отображения в CLI."""
        ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Переранжировать результаты поиска.

        Args:
            query:   Исходный вопрос пользователя.
            results: Список кандидатов от retriever (initial_k штук).
            top_k:   Сколько лучших результатов вернуть.

        Returns:
            Список до top_k результатов, отсортированных по убыванию релевантности.
        """
        ...


class ThresholdFilter(Reranker):
    """Фильтрация по порогу релевантности.

    Оставляет только результаты со score >= threshold,
    пересортировывает по score и обрезает до top_k.
    Не требует внешних API — работает всегда.
    """

    name = "threshold"

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Минимальный score для включения в результат.
                       Значение из .env: RERANK_THRESHOLD (по умолчанию 0.3).
        """
        self.threshold = threshold

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        filtered = [r for r in results if r.score >= self.threshold]
        filtered.sort(key=lambda r: r.score, reverse=True)
        return filtered[:top_k]


class CohereReranker(Reranker):
    """Реранкинг через Cohere Rerank API v2 (rerank-v3.5).

    Кросс-энкодер оценивает пару (query, document) целиком —
    это качественнее, чем cosine similarity отдельных эмбеддингов.
    Модель rerank-v3.5: multilingual, контекст 4096 токенов.

    При отсутствии COHERE_API_KEY — fallback на ThresholdFilter с предупреждением.
    При ошибке API (2 попытки с паузой 1с) — fallback на ThresholdFilter.
    """

    name = "cohere"

    def __init__(
        self,
        model: str = "rerank-v3.5",
        top_n: int = 5,
        proxy: str = "",
        max_doc_chars: int = 100,
    ):
        """
        Args:
            model:         Cohere rerank model (по умолчанию rerank-v3.5).
            top_n:         Сколько результатов запросить у API (max = top_k).
            proxy:         URL прокси-сервера, например http://user:pass@host:port.
                           Если не передан — берётся из COHERE_PROXY env.
            max_doc_chars: Максимум символов на документ перед отправкой в API.
                           По умолчанию 100 — достаточно для оценки релевантности,
                           payload ~7KB вместо 35KB.
        """
        self.model = model
        self.top_n = top_n
        self.api_key = os.environ.get("COHERE_API_KEY", "")
        self.proxy = proxy or os.environ.get("COHERE_PROXY", "")
        self.max_doc_chars = max_doc_chars

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not self.api_key:
            warnings.warn(
                "COHERE_API_KEY не задан. Используется ThresholdFilter.",
                stacklevel=2,
            )
            return ThresholdFilter().rerank(query, results, top_k)

        if not results:
            return results

        documents = [r.text[: self.max_doc_chars] for r in results]
        payload = json.dumps(
            {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
            }
        ).encode()

        import sys as _sys
        print(
            f"[Cohere] proxy={'YES' if self.proxy else 'NO'} "
            f"docs={len(documents)} payload={len(payload)}b",
            file=_sys.stderr,
        )

        req = urllib.request.Request(
            "https://api.cohere.com/v2/rerank",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        if self.proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy})
            )
        else:
            opener = urllib.request.build_opener()

        last_exc: Exception | None = None
        data: dict = {}
        for attempt in range(2):
            try:
                with opener.open(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(1)

        if last_exc is not None:
            warnings.warn(
                f"Cohere API недоступен: {last_exc}. Используется ThresholdFilter.",
                stacklevel=2,
            )
            return ThresholdFilter().rerank(query, results, top_k)

        reranked = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            reranked.append(replace(results[idx], score=score))

        return reranked[:top_k]


class OllamaReranker(Reranker):
    """Реранкинг через локальную модель в Ollama.

    Для каждого чанка спрашивает модель оценить релевантность (0–10).
    Работает полностью оффлайн, без облачных API-ключей.

    При недоступности Ollama — fallback на ThresholdFilter с предупреждением.

    Установка Ollama:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull qwen2.5:0.5b
        ollama serve &
    """

    name = "ollama"

    _PROMPT_TEMPLATE = (
        "Оцени релевантность текста вопросу от 0 до 10.\n"
        "Вопрос: {query}\n"
        "Текст: {text}\n"
        "Верни ТОЛЬКО одно число от 0 до 10."
    )

    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434",
    ):
        """
        Args:
            model:    Имя модели в Ollama. Значение из .env: OLLAMA_RERANK_MODEL.
            base_url: URL сервера Ollama. Значение из .env: OLLAMA_BASE_URL.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _is_available(self) -> bool:
        """Проверить доступность Ollama через GET /api/tags."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3):
                return True
        except Exception:
            return False

    def _score_chunk(self, query: str, text: str) -> float:
        """Оценить релевантность одного чанка, вернуть score в диапазоне 0.0–1.0."""
        prompt = self._PROMPT_TEMPLATE.format(query=query, text=text[:100])
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            raw = data.get("response", "0").strip()
            nums = re.findall(r"\d+(?:\.\d+)?", raw)
            return float(nums[0]) / 10.0 if nums else 0.0
        except Exception:
            return 0.0

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not self._is_available():
            warnings.warn(
                "Ollama недоступна. Используется ThresholdFilter.",
                stacklevel=2,
            )
            return ThresholdFilter().rerank(query, results, top_k)

        scored = []
        for r in results:
            score = self._score_chunk(query, r.text)
            scored.append(replace(r, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
