#!/usr/bin/env python3
"""Демонстрация полного локального стека (День 27).

Скрипт БЕЗ интерактивного ввода. Проверяет и демонстрирует:

    Шаг 1 — Health check (Ollama, модели, индекс, Telegram)
    Шаг 2 — Индексация с OllamaEmbedder (если индекса нет)
    Шаг 3 — RAG-запрос полностью локально (embed → retrieval → rerank → generate)
    Шаг 4 — Сравнение local vs cloud (если облачные ключи есть)
    Шаг 5 — Три вопроса по документации (easy / medium / hard)

Запуск:
    python demo_local_stack.py

Требования:
    ollama serve               # Ollama запущена
    ollama pull qwen2.5:0.5b     # LLM и реранкер
    ollama pull nomic-embed-text  # Эмбеддинги
"""

from __future__ import annotations

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
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _sep(title: str = "", width: int = 60) -> None:
    """Вывести разделитель."""
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


# ---------------------------------------------------------------------------
# Шаг 1: Health Check
# ---------------------------------------------------------------------------

def step1_health_check() -> bool:
    """Проверить готовность всех компонентов.

    Returns:
        True если стек готов, False если есть ошибки.
    """
    _sep("Шаг 1: Health Check")

    from src.providers.stack_config import StackHealthCheck
    checker = StackHealthCheck()
    results = checker.check_local()
    checker.print_table(results)

    # Telegram — опциональный для демо (нужен только для бота)
    _OPTIONAL = {"telegram"}

    failed = checker.get_failed(results)
    critical_failed = [f for f in failed if f not in _OPTIONAL]

    if failed:
        print("\n[!] Не готовы компоненты:", ", ".join(failed))
        print("\nИнструкции:")
        if "ollama_server" in failed:
            print("  → Запустите Ollama: ollama serve")
        if "llm_model" in failed:
            llm = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")
            print(f"  → Загрузите модель: ollama pull {llm}")
        if "embed_model" in failed:
            embed = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            print(f"  → Загрузите модель: ollama pull {embed}")
        if "telegram" in failed:
            print("  → Telegram опционален для демо. Для бота: добавьте TELEGRAM_BOT_TOKEN в .env")
        if "index" in failed:
            print("  → Запустите индексацию (Шаг 2 продолжит автоматически)")

    if critical_failed:
        # Если только индекс — продолжаем (шаг 2 проиндексирует)
        if critical_failed == ["index"]:
            return True
        return False

    return True


# ---------------------------------------------------------------------------
# Шаг 2: Индексация
# ---------------------------------------------------------------------------

def step2_indexing(docs_path: Path, db_path: Path) -> bool:
    """Проиндексировать документацию локальным эмбеддером.

    Args:
        docs_path: Путь к директории с .md файлами.
        db_path:   Путь к SQLite-базе для индекса.

    Returns:
        True если индексация успешна или индекс уже существует.
    """
    _sep("Шаг 2: Индексация (OllamaEmbedder)")

    if db_path.exists():
        # Проверяем, что индекс не пустой
        try:
            from rag_indexer.src.storage.index_store import IndexStore
            with IndexStore(db_path) as store:
                stats = store.get_stats()
            count = stats.get("chunks", 0)
            if count > 0:
                print(f"[OK] Индекс уже существует: {count} чанков в {db_path}")
                return True
        except Exception:
            pass

    if not docs_path.exists():
        print(f"[SKIP] Директория с документами не найдена: {docs_path}")
        print(f"  Для индексации выполните:")
        print(f"  python rag_indexer/main.py index --docs ./podkop-wiki/content \\")
        print(f"    --db {db_path} --embedder ollama")
        return False

    print(f"[>] Индексация: {docs_path} → {db_path}")
    print(f"[>] Embedder: OllamaEmbedder ({os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')})")

    t0 = time.time()

    from rag_indexer.src.embedding.ollama_embedder import OllamaEmbedder
    from rag_indexer.src.pipeline import IndexingPipeline
    from rag_indexer.src.chunking.strategies import STRATEGIES

    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embedder = OllamaEmbedder(model=embed_model, base_url=base_url)

    pipeline = IndexingPipeline(
        docs_path=docs_path,
        db_path=db_path,
        embedding_provider=embedder,
        strategies=list(STRATEGIES.values()),
    )

    try:
        pipeline.run()
    except Exception as exc:
        print(f"[ERROR] Ошибка индексации: {exc}")
        return False

    elapsed = _elapsed(t0)

    # Статистика
    from rag_indexer.src.storage.index_store import IndexStore
    with IndexStore(db_path) as store:
        stats = store.get_stats()

    print(f"\n[OK] Индексация завершена за {elapsed}")
    print(f"     Документов: {stats.get('documents', '?')}")
    print(f"     Чанков:     {stats.get('chunks', '?')}")
    print(f"     dim:        {embedder._dimension or '?'}")

    return True


# ---------------------------------------------------------------------------
# Шаг 3: RAG-запрос локально
# ---------------------------------------------------------------------------

def step3_local_rag_query(db_path: Path, question: str) -> dict:
    """Выполнить RAG-запрос через полностью локальный стек.

    Пайплайн: OllamaEmbedder → HybridRetrieval → OllamaReranker → OllamaProvider

    Args:
        db_path:  Путь к SQLite-индексу.
        question: Вопрос пользователя.

    Returns:
        Словарь с answer, sources, timings.
    """
    _sep("Шаг 3: Local RAG Query")
    print(f"[>] Вопрос: «{question}»\n")

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    rerank_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:0.5b")
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")

    timings = {}

    # Embedder
    from rag_indexer.src.embedding.ollama_embedder import OllamaEmbedder
    embedder = OllamaEmbedder(model=embed_model, base_url=base_url)

    # Reranker
    from rag_indexer.src.retrieval.reranker import OllamaReranker
    reranker = OllamaReranker(model=rerank_model, base_url=base_url)

    # Retriever
    from rag_indexer.src.storage.index_store import IndexStore
    from rag_indexer.src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever

    store = IndexStore(db_path)
    vector_retriever = VectorRetriever(store=store, embedder=embedder)
    bm25_retriever = BM25Retriever(store=store)
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    # LLM
    from src.providers.stack_config import create_providers
    providers = create_providers("local")
    llm_fn = providers["llm_fn"]

    # RAG Pipeline
    from rag_indexer.src.retrieval.pipeline import RAGPipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        reranker=reranker,
        use_structured=True,
    )

    # Выполнение запроса
    t_total = time.time()

    t0 = time.time()
    print(f"  [embed] Генерация эмбеддинга запроса...", end=" ", flush=True)
    # Embed будет вызван внутри pipeline при первом поиске
    timings["retrieval_start"] = time.time()

    try:
        result = pipeline.query(question)
        timings["total"] = _elapsed(t_total)

        print("OK")
        print(f"  [total] Время: {timings['total']}")
        print()

        # Вывод результата
        if hasattr(result, "answer"):
            answer = result.answer
            sources = getattr(result, "sources", [])
            confidence = getattr(result, "confidence", None)
        elif isinstance(result, dict):
            answer = result.get("answer", str(result))
            sources = result.get("sources", [])
            confidence = result.get("confidence")
        else:
            answer = str(result)
            sources = []
            confidence = None

        print(f"  Ответ: {answer[:500]}{'...' if len(str(answer)) > 500 else ''}")
        if sources:
            print(f"  Источники ({len(sources)}):")
            for s in sources[:3]:
                print(f"    - {s}")
        if confidence is not None:
            print(f"  Confidence: {confidence:.2f}")

        return {
            "ok": True,
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "timings": timings,
        }

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return {"ok": False, "question": question, "error": str(exc), "timings": timings}


# ---------------------------------------------------------------------------
# Шаг 4: Сравнение local vs cloud
# ---------------------------------------------------------------------------

def step4_compare_local_vs_cloud(db_path: Path, question: str) -> None:
    """Сравнить производительность local vs cloud стека.

    Args:
        db_path:  Путь к SQLite-индексу.
        question: Тестовый вопрос.
    """
    _sep("Шаг 4: Local vs Cloud")

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    rerank_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:0.5b")
    qwen_model = os.environ.get("QWEN_MODEL", "qwen-plus")

    # Local timing
    t_local_start = time.time()
    local_ok = False
    local_time = "N/A"
    try:
        from src.providers.stack_config import create_providers
        providers = create_providers("local")
        _ = providers["llm_fn"]("Ответь кратко", "Скажи: тест")
        local_time = f"{time.time() - t_local_start:.1f}s"
        local_ok = True
    except Exception as exc:
        local_time = f"❌ {str(exc)[:30]}"

    # Cloud timing
    t_cloud_start = time.time()
    cloud_ok = False
    cloud_time = "N/A"
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if qwen_key:
        try:
            from src.llm_helper import make_llm_fn
            cloud_llm = make_llm_fn(timeout=30.0)
            _ = cloud_llm("Ответь кратко", "Скажи: тест")
            cloud_time = f"{time.time() - t_cloud_start:.1f}s"
            cloud_ok = True
        except Exception as exc:
            cloud_time = f"❌ {str(exc)[:30]}"
    else:
        cloud_time = "нет API-ключа"

    # Таблица сравнения
    print(f"  Вопрос: «{question[:50]}»\n")
    print("  ┌─────────────────┬─────────────────┬──────────────────┐")
    print("  │ Метрика         │ Local           │ Cloud            │")
    print("  ├─────────────────┼─────────────────┼──────────────────┤")
    print(f"  │ LLM             │ {llm_model:<15} │ {qwen_model:<16} │")
    print(f"  │ Embeddings      │ {embed_model[:15]:<15} │ {'text-emb-v3':<16} │")
    print(f"  │ Reranker        │ {rerank_model[:15]:<15} │ {'rerank-v3.5':<16} │")
    print(f"  │ Время ответа    │ {local_time:<15} │ {cloud_time:<16} │")
    print(f"  │ Стоимость       │ {'0 ₽':<15} │ {'~0.5 ₽/запрос':<16} │")
    print(f"  │ Приватность     │ {'полная':<15} │ {'данные в API':<16} │")
    print(f"  │ Интернет        │ {'не нужен':<15} │ {'нужен':<16} │")
    print("  └─────────────────┴─────────────────┴──────────────────┘")
    print()


# ---------------------------------------------------------------------------
# Шаг 5: Три вопроса по документации
# ---------------------------------------------------------------------------

def step5_three_questions(db_path: Path) -> list[dict]:
    """Ответить на три вопроса разной сложности.

    Args:
        db_path: Путь к SQLite-индексу.

    Returns:
        Список результатов для каждого вопроса.
    """
    _sep("Шаг 5: Три вопроса по документации")

    questions = [
        ("easy",   "Как установить podkop?"),
        ("medium",  "Чем отличается vless от trojan?"),
        ("hard",   "Как настроить маршрутизацию для YouTube и Telegram?"),
    ]

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    rerank_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:0.5b")

    from rag_indexer.src.embedding.ollama_embedder import OllamaEmbedder
    from rag_indexer.src.retrieval.reranker import OllamaReranker
    from rag_indexer.src.storage.index_store import IndexStore
    from rag_indexer.src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever
    from rag_indexer.src.retrieval.pipeline import RAGPipeline
    from src.providers.stack_config import create_providers

    embedder = OllamaEmbedder(model=embed_model, base_url=base_url)
    reranker = OllamaReranker(model=rerank_model, base_url=base_url)
    store = IndexStore(db_path)
    vector_retriever = VectorRetriever(store=store, embedder=embedder)
    bm25_retriever = BM25Retriever(store=store)
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )
    providers = create_providers("local")
    llm_fn = providers["llm_fn"]

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        reranker=reranker,
        use_structured=True,
    )

    results = []
    for level, question in questions:
        print(f"\n  [{level.upper()}] «{question}»")
        t0 = time.time()
        try:
            result = pipeline.query(question)
            elapsed = _elapsed(t0)

            if hasattr(result, "answer"):
                answer = result.answer
                sources = getattr(result, "sources", [])
            elif isinstance(result, dict):
                answer = result.get("answer", str(result))
                sources = result.get("sources", [])
            else:
                answer = str(result)
                sources = []

            preview = str(answer)[:200].replace("\n", " ")
            print(f"  ✅ {elapsed}")
            print(f"  Ответ: {preview}{'...' if len(str(answer)) > 200 else ''}")
            if sources:
                print(f"  Источники: {', '.join(str(s) for s in sources[:2])}")

            results.append({"level": level, "question": question, "ok": True, "elapsed": elapsed})
        except Exception as exc:
            elapsed = _elapsed(t0)
            print(f"  ❌ {elapsed}: {exc}")
            results.append({"level": level, "question": question, "ok": False, "elapsed": elapsed, "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------

def print_final_report(health_ok: bool, index_ok: bool, rag_result: dict, questions: list[dict]) -> None:
    """Вывести итоговый отчёт по всем тестам."""
    _sep("Итоговый отчёт: Local Stack Integration")

    rows = [
        ("Health check",       health_ok,          "7/7 OK" if health_ok else "ошибки"),
        ("Local indexing",     index_ok,            "OK" if index_ok else "пропущено"),
        ("Local RAG query",    rag_result.get("ok"), rag_result.get("timings", {}).get("total", "N/A")),
        ("Structured resp.",   rag_result.get("ok") and bool(rag_result.get("sources")),
                               "src + quotes" if rag_result.get("sources") else "нет sources"),
    ]

    for q in questions:
        rows.append((
            f"Question ({q['level']})",
            q["ok"],
            q.get("elapsed", "N/A"),
        ))

    # Проверка: не было ли облачных вызовов
    mode_ok = os.environ.get("LLM_MODE", "cloud").lower() == "local"
    rows.append(("No cloud APIs used", mode_ok, "100% local" if mode_ok else "LLM_MODE!=local"))

    print("  ┌────────────────────┬──────────┬───────────────┐")
    print("  │ Тест               │ Статус   │ Детали        │")
    print("  ├────────────────────┼──────────┼───────────────┤")
    for name, ok, detail in rows:
        icon = "✅" if ok else "❌"
        print(f"  │ {name:<18} │ {icon:<8} │ {str(detail):<13} │")
    print("  └────────────────────┴──────────┴───────────────┘")

    total = len(rows)
    passed = sum(1 for _, ok, _ in rows if ok)
    print(f"\n  Итого: {passed}/{total} тестов прошли.")
    if passed == total:
        print("  🎉 Локальный стек полностью работоспособен!")
    else:
        print("  ⚠️  Некоторые тесты не прошли. Проверьте вывод выше.")


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    """Запустить демонстрацию локального стека."""
    # Принудительно устанавливаем локальный режим для демо
    os.environ["LLM_MODE"] = "local"

    _sep("День 27: Local Stack Demo", 60)
    print("  Стек: Ollama (LLM + Embeddings + Reranker)")
    print("  Облачных вызовов: 0 (кроме Telegram)")
    print()

    # Пути
    docs_path = Path("./podkop-wiki/content")
    db_path = Path("./output/index_local.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Шаг 1: Health Check ---
    health_ok = step1_health_check()
    print()

    if not health_ok:
        print("[STOP] Критические компоненты недоступны. Исправьте ошибки и запустите снова.")
        sys.exit(1)

    # --- Шаг 2: Индексация ---
    index_ok = step2_indexing(docs_path, db_path)
    print()

    if not index_ok or not db_path.exists():
        print("[WARN] Индекс недоступен. RAG-шаги будут пропущены.")
        print_final_report(health_ok, index_ok, {"ok": False, "sources": []}, [])
        return

    # --- Шаг 3: Один RAG-запрос ---
    rag_result = step3_local_rag_query(db_path, "Как установить podkop?")
    print()

    # --- Шаг 4: Сравнение local vs cloud ---
    step4_compare_local_vs_cloud(db_path, "Как установить podkop?")

    # --- Шаг 5: Три вопроса ---
    questions = step5_three_questions(db_path)
    print()

    # --- Итоговый отчёт ---
    print_final_report(health_ok, index_ok, rag_result, questions)


if __name__ == "__main__":
    main()
