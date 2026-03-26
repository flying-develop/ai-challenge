"""Конфигурация стека провайдеров и проверка готовности.

Предоставляет:
    create_providers(mode) — фабрика для создания всех провайдеров под выбранный режим
    StackHealthCheck       — проверка готовности всех компонентов локального стека

Переключение режима через .env:
    LLM_MODE=local   — всё через Ollama (LLM + embeddings + reranker)
    LLM_MODE=cloud   — облачные API (Qwen/OpenAI + DashScope + Cohere)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Фабрика провайдеров
# ---------------------------------------------------------------------------

def create_providers(mode: str | None = None) -> dict:
    """Создать все провайдеры под выбранный режим.

    Args:
        mode: "local" или "cloud". Если None — берётся из LLM_MODE env.

    Returns:
        Словарь с ключами: llm_fn, embedder, reranker.

    Примеры::

        providers = create_providers("local")
        providers["embedder"].embed_texts(["hello"])
        providers["llm_fn"]("system", "user query")
    """
    if mode is None:
        mode = os.environ.get("LLM_MODE", "cloud").lower()

    if mode == "local":
        return _create_local_providers()
    else:
        return _create_cloud_providers()


def _create_local_providers() -> dict:
    """Создать локальные провайдеры через Ollama."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    rerank_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:0.5b")

    # LLM через Ollama
    from llm_agent.infrastructure.ollama_client import OllamaHttpClient
    llm_client = OllamaHttpClient(model=llm_model, base_url=base_url, timeout=300.0)

    def llm_fn(system: str, user: str) -> str:
        from llm_agent.domain.models import ChatMessage
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=user))
        response = llm_client.generate(messages)
        return response.text

    # Embedder через Ollama
    from rag_indexer.src.embedding.ollama_embedder import OllamaEmbedder
    embedder = OllamaEmbedder(model=embed_model, base_url=base_url)

    # Reranker через Ollama
    from rag_indexer.src.retrieval.reranker import OllamaReranker
    reranker = OllamaReranker(model=rerank_model, base_url=base_url)

    return {
        "mode": "local",
        "llm_fn": llm_fn,
        "llm_client": llm_client,
        "embedder": embedder,
        "reranker": reranker,
        "llm_model": llm_model,
        "embed_model": embed_model,
        "rerank_model": rerank_model,
    }


def _create_cloud_providers() -> dict:
    """Создать облачные провайдеры (Qwen/DashScope/Cohere)."""
    # LLM через make_llm_fn (автовыбор провайдера)
    from src.llm_helper import make_llm_fn
    llm_fn = make_llm_fn(timeout=60.0)

    # Embedder через DashScope Qwen
    from rag_indexer.src.embedding.provider import EmbeddingProvider
    embedder = EmbeddingProvider.create("qwen")

    # Reranker через Cohere (с fallback на threshold)
    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if cohere_key:
        from rag_indexer.src.retrieval.reranker import CohereReranker
        reranker = CohereReranker()
    else:
        from rag_indexer.src.retrieval.reranker import ThresholdFilter
        reranker = ThresholdFilter(threshold=0.0)

    llm_model = os.environ.get("QWEN_MODEL", "qwen-plus")
    embed_model = embedder.model_name
    rerank_model = "rerank-v3.5" if cohere_key else "threshold"

    return {
        "mode": "cloud",
        "llm_fn": llm_fn,
        "embedder": embedder,
        "reranker": reranker,
        "llm_model": llm_model,
        "embed_model": embed_model,
        "rerank_model": rerank_model,
    }


# ---------------------------------------------------------------------------
# StackHealthCheck
# ---------------------------------------------------------------------------

class StackHealthCheck:
    """Проверяет готовность всех компонентов локального стека.

    Использование::

        checker = StackHealthCheck()
        results = checker.check_local()
        checker.print_table(results)
    """

    def __init__(
        self,
        base_url: str | None = None,
        llm_model: str | None = None,
        embed_model: str | None = None,
        db_path: str | None = None,
    ):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm_model = llm_model or os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")
        self.embed_model = embed_model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.db_path = Path(db_path or os.environ.get("RAG_DB_PATH", "./output/index.db"))

    def check_local(self) -> dict:
        """Выполнить полную проверку локального стека.

        Проверяет:
            1. Ollama сервер запущен
            2. LLM-модель загружена
            3. Embedding-модель загружена
            4. Тестовая генерация работает
            5. Тестовый embed работает
            6. Индекс существует и не пустой
            7. Telegram BOT_TOKEN задан

        Returns:
            Словарь {check_name: {"ok": bool, "detail": str}}.
        """
        results = {}
        results["ollama_server"] = self._check_ollama_server()
        results["llm_model"] = self._check_model(self.llm_model)
        results["embed_model"] = self._check_model(self.embed_model)
        results["llm_test"] = self._test_generate()
        results["embed_test"] = self._test_embed()
        results["index"] = self._check_index()
        results["telegram"] = self._check_telegram_token()
        return results

    def _check_ollama_server(self) -> dict:
        """Проверить доступность Ollama сервера."""
        try:
            t0 = time.time()
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                resp.read()
            elapsed = time.time() - t0
            return {"ok": True, "detail": f"{self.base_url} ({elapsed:.1f}s)"}
        except Exception as exc:
            return {"ok": False, "detail": f"Недоступен: {exc}"}

    def _check_model(self, model_name: str) -> dict:
        """Проверить, что модель загружена в Ollama."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            found = any(
                m == model_name or m == f"{model_name}:latest"
                for m in models
            )
            if found:
                return {"ok": True, "detail": "загружена"}
            else:
                available = ", ".join(models) if models else "(нет моделей)"
                return {
                    "ok": False,
                    "detail": f"не найдена. Загрузите: ollama pull {model_name}",
                }
        except Exception as exc:
            return {"ok": False, "detail": f"ошибка: {exc}"}

    def _test_generate(self) -> dict:
        """Тестовый запрос к LLM-модели."""
        payload = json.dumps({
            "model": self.llm_model,
            "prompt": "Скажи только: OK",
            "stream": False,
            "options": {"num_predict": 5},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp.read()
            elapsed = time.time() - t0
            return {"ok": True, "detail": f"{elapsed:.1f}s"}
        except Exception as exc:
            return {"ok": False, "detail": f"ошибка: {exc}"}

    def _test_embed(self) -> dict:
        """Тестовый запрос к embedding-модели."""
        payload = json.dumps({
            "model": self.embed_model,
            "input": ["test embedding"],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            embeddings = data.get("embeddings", [[]])
            dim = len(embeddings[0]) if embeddings else 0
            return {"ok": True, "detail": f"{elapsed:.1f}s, dim={dim}"}
        except Exception as exc:
            return {"ok": False, "detail": f"ошибка: {exc}"}

    def _check_index(self) -> dict:
        """Проверить наличие и заполненность индекса."""
        if not self.db_path.exists():
            return {"ok": False, "detail": f"не найден: {self.db_path}"}
        try:
            # Добавляем путь к проекту для импорта
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from rag_indexer.src.storage.index_store import IndexStore
            with IndexStore(self.db_path) as store:
                stats = store.get_stats()
            count = stats.get("chunks", 0)
            if count == 0:
                return {"ok": False, "detail": "пустой (0 чанков)"}
            return {"ok": True, "detail": f"{count} чанков"}
        except Exception as exc:
            return {"ok": False, "detail": f"ошибка: {exc}"}

    def _check_telegram_token(self) -> dict:
        """Проверить наличие Telegram токена."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if token and token != "your_bot_token_here":
            masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
            return {"ok": True, "detail": f"задан ({masked})"}
        return {"ok": False, "detail": "TELEGRAM_BOT_TOKEN не задан"}

    def print_table(self, results: dict) -> None:
        """Вывести результаты проверки в виде таблицы.

        Args:
            results: Словарь от check_local().
        """
        labels = {
            "ollama_server": f"Ollama сервер",
            "llm_model": f"LLM: {self.llm_model}",
            "embed_model": f"Embed: {self.embed_model}",
            "llm_test": "Тест генерации",
            "embed_test": "Тест эмбеддинга",
            "index": "Индекс",
            "telegram": "Telegram",
        }

        print("┌" + "─" * 44 + "┐")
        print("│  Stack Health Check (LOCAL mode)           │")
        print("├" + "─" * 24 + "┬" + "─" * 19 + "┤")

        all_ok = True
        for key, label in labels.items():
            r = results.get(key, {"ok": False, "detail": "?"})
            icon = "✅" if r["ok"] else "❌"
            if not r["ok"]:
                all_ok = False
            detail = r["detail"][:17]
            print(f"│ {label:<22} │ {icon} {detail:<16} │")

        print("└" + "─" * 24 + "┴" + "─" * 19 + "┘")

        if all_ok:
            print("\n✅ Стек готов к работе.")
        else:
            print("\n❌ Некоторые компоненты недоступны. Исправьте ошибки выше.")

    def get_failed(self, results: dict, skip_optional: bool = False) -> list[str]:
        """Получить список неуспешных проверок.

        Args:
            results:         Словарь от check_local().
            skip_optional:   Если True — исключает опциональные проверки (telegram).

        Returns:
            Список ключей с ok=False.
        """
        _optional = {"telegram"}
        return [
            k for k, v in results.items()
            if not v.get("ok", False) and (not skip_optional or k not in _optional)
        ]
