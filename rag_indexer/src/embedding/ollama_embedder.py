"""OllamaEmbedder — локальная генерация эмбеддингов через Ollama.

Использует эндпоинт POST /api/embed (batch API).
Работает полностью оффлайн, без облачных API-ключей.

Рекомендуемые модели:
    nomic-embed-text — 274MB, dim=768 (рекомендуется)
    all-minilm       — 67MB,  dim=384 (быстрая, для слабых машин)
    qwen2.5:3b       — можно, но медленнее (LLM, не embedding-модель)

Установка:
    ollama pull nomic-embed-text
    ollama serve  # если не запущен

HTTP-запросы только через urllib (инвариант stdlib, без httpx/requests).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from .provider import EmbeddingProvider


class OllamaEmbedder(EmbeddingProvider):
    """Генерация эмбеддингов через локальную Ollama.

    Использует эндпоинт /api/embed (batch — список текстов за один запрос).
    Размерность определяется автоматически при первом вызове.

    Args:
        model:    Имя embedding-модели в Ollama (например, nomic-embed-text).
        base_url: URL сервера Ollama (по умолчанию http://localhost:11434).
        timeout:  Таймаут HTTP-запроса в секундах (по умолчанию 60).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        batch_size: int = 10,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size  # Ollama медленнее DashScope — меньший батч надёжнее
        self._dimension: int | None = None  # определяется при первом вызове

    @property
    def model_name(self) -> str:
        """Название модели в формате 'ollama/<model>'."""
        return f"ollama/{self.model}"

    @property
    def dimension(self) -> int:
        """Размерность векторов.

        Определяется при первом вызове embed_texts() через пробный запрос.
        """
        if self._dimension is None:
            test = self.embed_texts(["test"])
            self._dimension = len(test[0])
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для списка текстов через Ollama /api/embed.

        Разбивает список на подбатчи по batch_size (по умолчанию 10),
        чтобы не превышать таймаут при большом числе текстов.

        Args:
            texts: Список строк для эмбеддинга.

        Returns:
            Список векторов в том же порядке, что и входные тексты.

        Raises:
            RuntimeError: Если Ollama недоступна или вернула ошибку.

        Example:
            POST http://localhost:11434/api/embed
            {
                "model": "nomic-embed-text",
                "input": ["text1", "text2", ...]
            }
            Ответ: {"embeddings": [[0.1, 0.2, ...], [0.3, ...], ...]}
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Один HTTP POST запрос к Ollama /api/embed для подбатча.

        Args:
            texts: Подбатч текстов (не более batch_size).

        Returns:
            Список векторов.

        Raises:
            RuntimeError: При HTTP-ошибке или недоступности сервера.
        """
        body = json.dumps({
            "model": self.model,
            "input": texts,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_err = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama /api/embed HTTP {exc.code}: {body_err}\n"
                f"Убедитесь, что модель загружена: ollama pull {self.model}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama недоступна по адресу {self.base_url}: {exc.reason}\n"
                "Запустите: ollama serve"
            ) from exc

        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(
                f"Неожиданный формат ответа Ollama /api/embed: {data}"
            )

        # Сохраняем размерность при первом успешном вызове
        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])

        return embeddings

    def is_available(self) -> bool:
        """Проверить доступность Ollama и наличие модели.

        Returns:
            True если сервер доступен и модель загружена.
        """
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=3
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            # Проверяем оба варианта: "model" и "model:latest"
            return any(
                m == self.model or m == f"{self.model}:latest"
                for m in models
            )
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"OllamaEmbedder(model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"batch_size={self.batch_size}, "
            f"dim={self._dimension})"
        )
