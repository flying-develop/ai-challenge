"""Провайдеры эмбеддингов (паттерн Strategy).

Реализует два провайдера:
- QwenEmbedder      — DashScope API (text-embedding-v3), urllib, батч до 25
- LocalRandomEmbedder — детерминированные псевдо-эмбеддинги для тестирования

Graceful fallback: если DASHSCOPE_API_KEY не задан — автоматически
используется LocalRandomEmbedder с предупреждением в stderr.

HTTP-запросы только через urllib (инвариант stdlib, без httpx/requests).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .._math import make_random_vector


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Абстрактный провайдер эмбеддингов.

    Реализует паттерн Strategy: конкретные классы переопределяют embed_texts().
    Фабричный метод create() позволяет создать провайдер по имени.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Название модели."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Размерность векторов."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для списка текстов.

        Args:
            texts: Список строк для эмбеддинга.

        Returns:
            Список векторов (каждый — list[float] длиной dimension).
        """
        ...

    @staticmethod
    def create(provider: str = "qwen", **kwargs) -> "EmbeddingProvider":
        """Фабричный метод: создать провайдер по имени.

        Args:
            provider: "qwen" или "local".
            **kwargs: Параметры провайдера (api_key, dimension, etc).

        Returns:
            Экземпляр EmbeddingProvider.

        Notes:
            Если provider="qwen" и DASHSCOPE_API_KEY не задан —
            автоматически возвращается LocalRandomEmbedder.
        """
        if provider == "qwen":
            api_key = kwargs.get("api_key") or os.environ.get("DASHSCOPE_API_KEY", "")
            if not api_key:
                print(
                    "[WARNING] DASHSCOPE_API_KEY не задан — "
                    "используется LocalRandomEmbedder (тестовый режим).",
                    file=sys.stderr,
                )
                return LocalRandomEmbedder(**{k: v for k, v in kwargs.items() if k != "api_key"})
            return QwenEmbedder(api_key=api_key, **{k: v for k, v in kwargs.items() if k != "api_key"})

        if provider == "local":
            return LocalRandomEmbedder(**kwargs)

        raise ValueError(f"Неизвестный провайдер '{provider}'. Доступны: qwen, local")


# ---------------------------------------------------------------------------
# QwenEmbedder
# ---------------------------------------------------------------------------

class QwenEmbedder(EmbeddingProvider):
    """Провайдер эмбеддингов через DashScope API (Qwen text-embedding-v3).

    Endpoint: POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/
              text-embedding/text-embedding
    Батч: до 25 текстов за запрос.
    Retry: 3 попытки с экспоненциальной задержкой (1s, 2s, 4s).

    Args:
        api_key:   DASHSCOPE_API_KEY.
        dimension: Размерность векторов (по умолчанию 1024).
        timeout:   Таймаут HTTP-запроса в секундах.
    """

    _ENDPOINT = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "text-embedding/text-embedding"
    )
    _BATCH_SIZE = 25
    _MAX_RETRIES = 3

    def __init__(
        self,
        api_key: str,
        dimension: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._dimension = dimension
        self._timeout = timeout
        # SSL-контекст с явным указанием TLS 1.2+ для совместимости
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    @property
    def model_name(self) -> str:
        return "text-embedding-v3"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги через DashScope API (батчами по 25).

        Args:
            texts: Список текстов.

        Returns:
            Список векторов в том же порядке.

        Raises:
            RuntimeError: При исчерпании попыток.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[batch_start: batch_start + self._BATCH_SIZE]
            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Отправить батч с retry-логикой (экспоненциальная задержка).

        SSL EOF (_ssl.c EOF occurred) и сетевые ошибки — ретраятся.
        HTTP-ошибки (4xx/5xx) — ретраятся только если 5xx.
        """
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(self._MAX_RETRIES):
            try:
                return self._embed_batch(batch)
            except RuntimeError as exc:
                # HTTP 4xx — не ретраим, сразу пробрасываем
                msg = str(exc)
                if "DashScope HTTP 4" in msg:
                    raise
                last_error = exc
            except (urllib.error.URLError, OSError, ssl.SSLError) as exc:
                last_error = exc

            if attempt < self._MAX_RETRIES - 1:
                print(
                    f"    [retry {attempt + 1}/{self._MAX_RETRIES - 1}] "
                    f"{type(last_error).__name__}: {last_error} — ожидание {delay:.0f}с...",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(
            f"DashScope API недоступен после {self._MAX_RETRIES} попыток: {last_error}"
        )

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Один HTTP POST запрос к DashScope.

        Returns:
            Список векторов в порядке text_index.

        Raises:
            urllib.error.URLError: При сетевой ошибке.
            RuntimeError: При HTTP ошибке или некорректном ответе.
        """
        payload = {
            "model": "text-embedding-v3",
            "input": {"texts": batch},
            "parameters": {"dimension": self._dimension},
        }
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self._ENDPOINT,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=self._ssl_ctx) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope HTTP {exc.code}: {body}") from exc

        result = json.loads(body)

        # Парсинг ответа
        try:
            embeddings_raw = result["output"]["embeddings"]
            # Сортировка по text_index для гарантии порядка
            embeddings_raw.sort(key=lambda e: e["text_index"])
            return [e["embedding"] for e in embeddings_raw]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Неожиданный формат ответа DashScope: {result}") from exc


# ---------------------------------------------------------------------------
# LocalRandomEmbedder
# ---------------------------------------------------------------------------

class LocalRandomEmbedder(EmbeddingProvider):
    """Детерминированные псевдо-эмбеддинги для тестирования без API.

    Алгоритм: seed = MD5(text)[:8] → numpy random → L2-нормализация.
    Одинаковый текст → одинаковый вектор (воспроизводимость).
    Только для верификации пайплайна, не для production.

    Args:
        dimension: Размерность векторов (по умолчанию 1024).
    """

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "local-random"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Сгенерировать детерминированные псевдо-эмбеддинги.

        Args:
            texts: Список текстов.

        Returns:
            Список L2-нормализованных векторов.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Получить детерминированный эмбеддинг для одного текста."""
        # seed из MD5(text)[:8] — int
        md5_hex = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        seed = int(md5_hex, 16) % (2**31)
        return make_random_vector(seed, self._dimension)
