#!/usr/bin/env python3
"""День 28: сравнение local vs cloud RAG стека.

Скрипт БЕЗ интерактивного ввода. Прогоняет одинаковый набор вопросов
через два стека и сравнивает качество, скорость и стабильность.

    Шаг 1 — Проверка обоих стеков
    Шаг 2 — Прогон 10 вопросов (или 3 в режиме --quick)
    Шаг 3 — Прогон 3 антивопросов
    Шаг 4 — Тест стабильности (2 вопроса × 3 прогона)
    Шаг 5 — Итоговые таблицы
    Шаг 6 — Автоматические выводы + рекомендация

Стеки:
    LOCAL:  Ollama qwen2.5 + nomic-embed-text + index_local.db
    CLOUD:  Qwen (DashScope) + text-emb-v3 + Cohere (или threshold) + index_cloud.db

Подготовка (одноразово):
    cd rag_indexer
    python main.py index --docs ../podkop-wiki/content --db output/index_local.db --embedder ollama
    python main.py index --docs ../podkop-wiki/content --db output/index_cloud.db --embedder qwen

Запуск:
    python demo_local_vs_cloud.py              # полный прогон (~5-10 мин)
    python demo_local_vs_cloud.py --quick      # только 3 вопроса (~2-3 мин)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_PROJECT_ROOT = Path(__file__).parent
# Только project root на sys.path — rag_indexer используется как namespace package
# (from rag_indexer.src.xxx import ...) чтобы избежать конфликта двух пакетов src
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _sep(title: str = "", width: int = 62) -> None:
    """Вывести секционный разделитель."""
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("=" * pad + f" {title} " + "=" * max(0, width - pad - len(title) - 2))
    else:
        print("=" * width)


def _check_ollama(base_url: str) -> tuple[bool, str]:
    """Проверить доступность Ollama и наличие нужных моделей."""
    import json
    import urllib.request as _req

    try:
        with _req.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")
        emb_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        llm_ok = any(m == llm_model or m == f"{llm_model}:latest" for m in models)
        emb_ok = any(m == emb_model or m == f"{emb_model}:latest" for m in models)
        if llm_ok and emb_ok:
            return True, f"LLM={llm_model}, embed={emb_model}"
        missing = []
        if not llm_ok:
            missing.append(f"LLM {llm_model}")
        if not emb_ok:
            missing.append(f"embed {emb_model}")
        return False, f"модели не загружены: {', '.join(missing)}"
    except Exception as exc:
        return False, f"сервер недоступен: {exc}"


def _check_cloud() -> tuple[bool, str]:
    """Проверить наличие облачных API-ключей."""
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    cohere_key = os.environ.get("COHERE_API_KEY")
    if not qwen_key:
        return False, "QWEN_API_KEY / DASHSCOPE_API_KEY не задан"
    status = "Qwen OK"
    if cohere_key:
        status += ", Cohere OK"
    else:
        status += ", Cohere нет (fallback threshold)"
    return True, status


def _check_index(db_path: Path) -> tuple[bool, str]:
    """Проверить существование и заполненность индекса."""
    if not db_path.exists():
        return False, f"не найден: {db_path}"
    try:
        from rag_indexer.src.storage.index_store import IndexStore
        with IndexStore(db_path) as store:
            stats = store.get_stats()
        count = stats.get("chunks", 0)
        if count == 0:
            return False, "пустой (0 чанков)"
        return True, f"{count} чанков"
    except Exception as exc:
        return False, f"ошибка чтения: {exc}"


# ---------------------------------------------------------------------------
# Шаг 1: Проверка стеков
# ---------------------------------------------------------------------------


def step1_health_check(local_db: Path, cloud_db: Path) -> tuple[bool, bool]:
    """Проверить готовность обоих стеков.

    Returns:
        (local_ok, cloud_ok) — доступность каждого стека.
    """
    _sep("Шаг 1: Проверка стеков")

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Local stack
    ollama_ok, ollama_msg = _check_ollama(base_url)
    local_idx_ok, local_idx_msg = _check_index(local_db)

    print("  LOCAL stack:")
    print(f"    {'✅' if ollama_ok else '❌'} Ollama: {ollama_msg}")
    print(f"    {'✅' if local_idx_ok else '❌'} index_local.db: {local_idx_msg}")
    if not local_idx_ok:
        print(
            f"\n  Для создания локального индекса выполните:\n"
            f"    cd rag_indexer\n"
            f"    python main.py index --docs ../podkop-wiki/content \\\n"
            f"      --db output/index_local.db --embedder ollama"
        )
    local_ok = ollama_ok and local_idx_ok

    print()

    # Cloud stack
    cloud_key_ok, cloud_key_msg = _check_cloud()
    cloud_idx_ok, cloud_idx_msg = _check_index(cloud_db)

    print("  CLOUD stack:")
    print(f"    {'✅' if cloud_key_ok else '❌'} API ключи: {cloud_key_msg}")
    print(f"    {'✅' if cloud_idx_ok else '❌'} index_cloud.db: {cloud_idx_msg}")
    if not cloud_idx_ok:
        print(
            f"\n  Для создания облачного индекса выполните:\n"
            f"    cd rag_indexer\n"
            f"    python main.py index --docs ../podkop-wiki/content \\\n"
            f"      --db output/index_cloud.db --embedder qwen"
        )
    cloud_ok = cloud_key_ok and cloud_idx_ok

    print()
    if not local_ok and not cloud_ok:
        print("  ❌ Ни один стек не готов. Исправьте ошибки выше.")
    elif not cloud_ok:
        print("  ⚠️  Cloud стек недоступен. Сравнение только по local.")
    else:
        print("  ✅ Оба стека готовы к сравнению.")

    return local_ok, cloud_ok


# ---------------------------------------------------------------------------
# Построение пайплайнов
# ---------------------------------------------------------------------------


def _build_local_pipeline(local_db: Path):
    """Создать RAGPipeline для локального стека (Ollama).

    Компоненты: OllamaEmbedder + HybridRetriever + OllamaReranker + Ollama LLM.
    """
    from rag_indexer.src.embedding.ollama_embedder import OllamaEmbedder
    from rag_indexer.src.retrieval.reranker import OllamaReranker
    from rag_indexer.src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever
    from rag_indexer.src.retrieval.pipeline import RAGPipeline
    from rag_indexer.src.storage.index_store import IndexStore

    base_url     = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model  = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    rerank_model = os.environ.get(
        "OLLAMA_RERANK_MODEL",
        os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b"),
    )
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")

    embedder  = OllamaEmbedder(model=embed_model, base_url=base_url)
    reranker  = OllamaReranker(model=rerank_model, base_url=base_url)
    store     = IndexStore(local_db)
    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(store=store, embedder=embedder),
        bm25_retriever=BM25Retriever(store=store),
    )

    # LLM через OllamaHttpClient (из llm_agent — всегда доступен в project)
    from llm_agent.infrastructure.ollama_client import OllamaHttpClient
    _llm_client = OllamaHttpClient(model=llm_model, base_url=base_url, timeout=300.0)

    def llm_fn(system: str, user: str) -> str:
        from llm_agent.domain.models import ChatMessage
        msgs = []
        if system:
            msgs.append(ChatMessage(role="system", content=system))
        msgs.append(ChatMessage(role="user", content=user))
        return _llm_client.generate(msgs).text

    return RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        reranker=reranker,
        use_structured=True,
    )


def _build_cloud_pipeline(cloud_db: Path):
    """Создать RAGPipeline для облачного стека (Qwen + Cohere/threshold).

    Компоненты: QwenEmbedder + HybridRetriever + CohereReranker + Qwen LLM.
    QWEN_API_KEY используется как алиас для DASHSCOPE_API_KEY если последний не задан.
    """
    from rag_indexer.src.embedding.provider import EmbeddingProvider
    from rag_indexer.src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever
    from rag_indexer.src.retrieval.pipeline import RAGPipeline
    from rag_indexer.src.storage.index_store import IndexStore

    # QWEN_API_KEY → DASHSCOPE_API_KEY алиас для эмбеддера
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    embedder  = EmbeddingProvider.create("qwen", api_key=qwen_key)
    store     = IndexStore(cloud_db)
    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(store=store, embedder=embedder),
        bm25_retriever=BM25Retriever(store=store),
    )

    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if cohere_key:
        from rag_indexer.src.retrieval.reranker import CohereReranker
        reranker = CohereReranker()
    else:
        from rag_indexer.src.retrieval.reranker import ThresholdFilter
        reranker = ThresholdFilter(threshold=0.0)

    # LLM через src.llm_helper (project root's src — не rag_indexer/src)
    from src.llm_helper import make_llm_fn
    llm_fn = make_llm_fn(timeout=60.0)

    return RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        reranker=reranker,
        use_structured=True,
    )


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------


def main(quick: bool = False) -> None:
    """Запустить полное сравнение local vs cloud."""
    _sep("День 28: Local vs Cloud RAG Benchmark", 62)
    print(
        f"  Режим: {'quick (3 вопроса)' if quick else 'full (10 вопросов + антивопросы + стабильность)'}"
    )
    print()

    # Пути к индексам
    output_dir = _PROJECT_ROOT / "rag_indexer" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_db = output_dir / "index_local.db"
    cloud_db = output_dir / "index_cloud.db"

    # --- Шаг 1: Health check ---
    local_ok, cloud_ok = step1_health_check(local_db, cloud_db)
    print()

    if not local_ok and not cloud_ok:
        print("[STOP] Невозможно запустить бенчмарк без хотя бы одного стека.")
        sys.exit(1)

    # --- Шаги 2-6: Бенчмарк ---
    _sep("Шаги 2-6: Бенчмарк")

    local_pipeline = None
    cloud_pipeline = None

    if local_ok:
        print("  [>] Инициализация local pipeline...", end=" ", flush=True)
        t0 = time.time()
        try:
            local_pipeline = _build_local_pipeline(local_db)
            print(f"OK ({time.time() - t0:.1f}s)")
        except Exception as exc:
            print(f"❌ {exc}")

    if cloud_ok:
        print("  [>] Инициализация cloud pipeline...", end=" ", flush=True)
        t0 = time.time()
        try:
            cloud_pipeline = _build_cloud_pipeline(cloud_db)
            print(f"OK ({time.time() - t0:.1f}s)")
        except Exception as exc:
            print(f"❌ {exc}")

    if local_pipeline is None and cloud_pipeline is None:
        print("\n[STOP] Не удалось создать ни один пайплайн.")
        sys.exit(1)

    # Запуск бенчмарка
    from rag_indexer.src.retrieval.benchmark import RAGBenchmark

    benchmark = RAGBenchmark(
        local_pipeline=local_pipeline,
        cloud_pipeline=cloud_pipeline,
    )

    results, anti_results, stability_results = benchmark.run(quick=quick)

    # --- Шаг 5: Таблицы ---
    _sep("Шаг 5: Итоговые таблицы")
    benchmark.print_all_tables(results, anti_results, stability_results)

    # --- Шаг 6: Выводы ---
    _sep("Шаг 6: Выводы")
    benchmark.print_conclusions(results, anti_results, stability_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="День 28: сравнение local vs cloud RAG стека"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый режим: только 3 вопроса (без антивопросов и стабильности)",
    )
    args = parser.parse_args()
    main(quick=args.quick)
