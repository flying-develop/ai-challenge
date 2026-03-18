#!/usr/bin/env python3
"""Демо: реранкинг и query rewriting в RAG-пайплайне.

Сравнивает 4-5 конфигураций на 5 контрольных вопросах:
    baseline  — hybrid, без реранкинга
    threshold — hybrid + ThresholdFilter(0.3)
    cohere    — hybrid + Cohere Rerank v3.5  (нужен COHERE_API_KEY)
    ollama    — hybrid + Ollama qwen2.5:3b   (нужен запущенный ollama)
    full      — hybrid + QueryRewrite + Cohere Rerank (нужны оба)

Для ускорения Ollama задайте лёгкую модель:
    OLLAMA_RERANK_MODEL=qwen2.5:0.5b  (в 3-5 раз быстрее qwen2.5:3b)

Запуск:
    python demo_reranking.py

Требования:
    - Индекс: rag_indexer/output/index.db  (python main.py index ...)
    - .env: DASHSCOPE_API_KEY (эмбеддинги), QWEN_API_KEY (LLM)
    - .env (опционально): COHERE_API_KEY, OLLAMA_BASE_URL
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import warnings

# Добавить rag_indexer в путь
_ROOT = os.path.dirname(os.path.abspath(__file__))
_RAG_DIR = os.path.join(_ROOT, "rag_indexer")
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

from src.storage.index_store import IndexStore
from src.embedding.provider import EmbeddingProvider
from src.retrieval.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from src.retrieval.reranker import ThresholdFilter, CohereReranker, OllamaReranker
from src.retrieval.query_rewrite import QueryRewriter
from src.retrieval.pipeline import RAGPipeline, PipelineEvaluator
from src.retrieval.evaluator import EVAL_QUESTIONS

DEFAULT_DB = os.path.join(_RAG_DIR, "output", "index.db")
DEMO_QUESTION = "Как установить podkop на роутер?"


def _make_cohere_opener():
    """Создать urllib opener с прокси если задан COHERE_PROXY."""
    proxy = os.environ.get("COHERE_PROXY", "")
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()


# ─── LLM ─────────────────────────────────────────────────────────────────────

def _make_llm_opener():
    """Создать urllib opener с прокси если задан LLM_PROXY."""
    proxy = os.environ.get("LLM_PROXY", "")
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()


def make_llm_fn():
    """Создать llm_fn через Qwen API (OpenAI-совместимый)."""
    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    model = os.environ.get("QWEN_MODEL", "qwen-plus")

    if not api_key:
        print("[WARN] QWEN_API_KEY не задан — ответы LLM будут пропущены", file=sys.stderr)
        return None

    opener = _make_llm_opener()

    def llm_fn(system: str, user: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = json.dumps(
            {"model": model, "messages": messages, "max_tokens": 400}
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with opener.open(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    return llm_fn


# ─── Проверка доступности ─────────────────────────────────────────────────────

def check_cohere() -> bool:
    """Вернуть True, если COHERE_API_KEY задан и ключ рабочий."""
    key = os.environ.get("COHERE_API_KEY", "")
    if not key:
        return False
    try:
        payload = json.dumps(
            {
                "model": "rerank-v3.5",
                "query": "test",
                "documents": ["doc1", "doc2"],
                "top_n": 2,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.cohere.com/v2/rerank",
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        opener = _make_cohere_opener()
        with opener.open(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception:
        return False


def check_ollama() -> bool:
    """Вернуть True, если Ollama запущена и доступна."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        return len(data.get("models", [])) > 0
    except Exception:
        return False


# ─── Шаг 1: Инициализация ─────────────────────────────────────────────────────

def step1_init(db_path: str):
    """Загрузить индекс, создать retrievers и проверить доступность API."""
    print("\n" + "═" * 70)
    print("  ШАГ 1: Инициализация")
    print("═" * 70)

    if not os.path.exists(db_path):
        print(f"[ERROR] Индекс не найден: {db_path}")
        print("  Создайте индекс: python main.py index --docs ./podkop-wiki/content --db ./output/index.db")
        sys.exit(1)

    store = IndexStore(db_path)
    embedder = EmbeddingProvider.create("qwen")
    strategy_filter = os.environ.get("RAG_STRATEGY_FILTER") or None

    vector_ret = VectorRetriever(store, embedder, strategy_filter)
    bm25_ret = BM25Retriever(store, strategy_filter)
    hybrid_ret = HybridRetriever(vector_ret, bm25_ret)

    # Статистика индекса
    try:
        stats = store.get_stats()
        print(f"  Индекс: {db_path}")
        print(f"  Чанков: {stats.get('chunks', '?')}")
    except Exception:
        print(f"  Индекс: {db_path}")

    # Проверка Cohere
    cohere_ok = check_cohere()
    ollama_ok = check_ollama()
    llm_fn = make_llm_fn()

    print(f"\n  Доступность компонентов:")
    print(f"    LLM (Qwen)  : {'✓ OK' if llm_fn else '✗ нет ключа (LLM-ответы пропущены)'}")
    print(f"    Cohere API  : {'✓ OK' if cohere_ok else '✗ нет ключа (конфигурации cohere/full пропущены)'}")
    ollama_model_hint = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:3b")
    print(f"    Ollama      : {'✓ OK  модель: ' + ollama_model_hint if ollama_ok else '✗ не запущена (конфигурация ollama пропущена)'}")
    if ollama_ok and ollama_model_hint == "qwen2.5:3b":
        print(f"    [TIP] Для ускорения: OLLAMA_RERANK_MODEL=qwen2.5:0.5b")

    return store, hybrid_ret, llm_fn, cohere_ok, ollama_ok


# ─── Шаг 2: Один вопрос, все конфигурации ────────────────────────────────────

def step2_single_question(
    hybrid_ret, llm_fn, cohere_ok: bool, ollama_ok: bool
) -> None:
    """Показать как работает каждый режим на одном демо-вопросе."""
    if not llm_fn:
        print("\n  [WARN] LLM не доступен, шаг 2 пропущен.")
        return

    print("\n" + "═" * 70)
    print(f'  ШАГ 2: Один вопрос через все доступные конфигурации')
    print("═" * 70)
    print(f'\n  Вопрос: "{DEMO_QUESTION}"\n')

    ollama_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:3b")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    threshold = float(os.environ.get("RERANK_THRESHOLD", "0.3"))

    configs = [
        ("baseline",   RAGPipeline(hybrid_ret, llm_fn)),
        ("threshold",  RAGPipeline(hybrid_ret, llm_fn, reranker=ThresholdFilter(threshold))),
    ]
    if cohere_ok:
        configs.append(("cohere", RAGPipeline(hybrid_ret, llm_fn, reranker=CohereReranker())))
    if ollama_ok:
        configs.append(("ollama", RAGPipeline(
            hybrid_ret, llm_fn,
            reranker=OllamaReranker(model=ollama_model, base_url=ollama_base),
        )))
    if cohere_ok:
        rewriter = QueryRewriter(llm_fn)
        configs.append(("full", RAGPipeline(
            hybrid_ret, llm_fn,
            reranker=CohereReranker(),
            query_rewriter=rewriter,
        )))

    for cfg_name, pipeline in configs:
        print(f"  ── [{cfg_name.upper()}] {'─' * (50 - len(cfg_name))}")
        initial_k = 5 if cfg_name == "baseline" else 10
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rag_answer = pipeline.answer(DEMO_QUESTION, top_k=5, initial_k=initial_k)
        for w in caught:
            print(f"  [WARN] {w.message}")

        if rag_answer.rewrite_variants:
            print(f"  Варианты запроса:")
            for i, v in enumerate(rag_answer.rewrite_variants):
                marker = "(исходный)" if i == 0 else f"(вариант {i})"
                print(f"    {marker}: {v}")

        print(f"  Кандидаты → финал: {rag_answer.initial_results_count} → {rag_answer.final_results_count}")
        print(f"  Источники:")
        seen = set()
        for r in rag_answer.sources:
            if r.source not in seen:
                print(f"    [{r.score:.3f}] {r.source}")
                seen.add(r.source)
        print(f"  Время: поиск={rag_answer.retrieval_time_ms:.0f}ms  "
              f"реранк={rag_answer.rerank_time_ms:.0f}ms  "
              f"llm={rag_answer.llm_time_ms:.0f}ms")
        print(f"\n  Ответ:\n    {rag_answer.answer[:400].replace(chr(10), chr(10) + '    ')}\n")


# ─── Шаг 3: Полный прогон 10 вопросов ────────────────────────────────────────

def step3_full_eval(
    hybrid_ret, llm_fn, cohere_ok: bool, ollama_ok: bool
) -> list | None:
    """Прогнать 10 контрольных вопросов по всем доступным конфигурациям."""
    if not llm_fn:
        print("\n  [WARN] LLM не доступен, шаг 3 пропущен.")
        return None

    print("\n" + "═" * 70)
    print("  ШАГ 3: Полный прогон — 5 вопросов × все конфигурации")
    print("═" * 70)

    ollama_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:3b")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    threshold = float(os.environ.get("RERANK_THRESHOLD", "0.3"))

    pipelines = [
        RAGPipeline(hybrid_ret, llm_fn),
        RAGPipeline(hybrid_ret, llm_fn, reranker=ThresholdFilter(threshold)),
    ]
    if cohere_ok:
        pipelines.append(RAGPipeline(hybrid_ret, llm_fn, reranker=CohereReranker()))
    if ollama_ok:
        pipelines.append(RAGPipeline(
            hybrid_ret, llm_fn,
            reranker=OllamaReranker(model=ollama_model, base_url=ollama_base),
        ))
    if cohere_ok:
        rewriter = QueryRewriter(llm_fn)
        pipelines.append(RAGPipeline(
            hybrid_ret, llm_fn,
            reranker=CohereReranker(),
            query_rewriter=rewriter,
        ))

    print(f"\n  Конфигурации: {', '.join(p.name for p in pipelines)}")
    print(f"  Вопросов: {len(EVAL_QUESTIONS)}")
    print(f"  Итого вызовов LLM: {len(pipelines) * len(EVAL_QUESTIONS)}\n")

    evaluator = PipelineEvaluator(pipelines, top_k=5, initial_k=10)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        results = evaluator.run()

    evaluator.print_comparison_table(results)
    evaluator.print_timing_summary(results)

    return results


# ─── Шаг 4: Cohere vs Ollama ──────────────────────────────────────────────────

def step4_cohere_vs_ollama(results, cohere_ok: bool, ollama_ok: bool) -> None:
    """Сравнить Cohere и Ollama по качеству, скорости и стоимости."""
    if not results or not (cohere_ok and ollama_ok):
        if cohere_ok and not ollama_ok:
            print("\n  [INFO] Ollama недоступна — сравнение Cohere vs Ollama пропущено.")
        elif not cohere_ok and ollama_ok:
            print("\n  [INFO] Cohere недоступен — сравнение Cohere vs Ollama пропущено.")
        return

    print("\n" + "═" * 70)
    print("  ШАГ 4: Сравнение Cohere vs Ollama")
    print("═" * 70)

    cohere_res = [r for r in results if r.pipeline_name == "hybrid+cohere"]
    ollama_res = [r for r in results if r.pipeline_name == "hybrid+ollama"]

    if not cohere_res or not ollama_res:
        print("  Нет данных для сравнения.")
        return

    def avg(lst, key):
        return sum(getattr(r, key) for r in lst) / len(lst)

    cohere_kw = avg(cohere_res, "keyword_rate")
    ollama_kw = avg(ollama_res, "keyword_rate")
    cohere_rer = avg(cohere_res, "rerank_time_ms")
    ollama_rer = avg(ollama_res, "rerank_time_ms")

    print(f"\n  {'Метрика':<30} {'Cohere':>12} {'Ollama':>12}")
    print("  " + "─" * 55)
    print(f"  {'keyword_rate (avg)':<30} {cohere_kw:>12.3f} {ollama_kw:>12.3f}")
    print(f"  {'rerank_time_ms (avg)':<30} {cohere_rer:>11.0f}ms {ollama_rer:>11.0f}ms")
    print(f"  {'Стоимость':<30} {'платный API':>12} {'бесплатно':>12}")
    print(f"  {'Конфиденциальность':<30} {'облако':>12} {'локально':>12}")
    print(f"  {'Требования':<30} {'API-ключ':>12} {'GPU/CPU':>12}")

    winner_quality = "Cohere" if cohere_kw > ollama_kw else ("Ollama" if ollama_kw > cohere_kw else "Равно")
    winner_speed = "Cohere" if cohere_rer < ollama_rer else "Ollama"
    print(f"\n  Вывод: качество → {winner_quality}, скорость → {winner_speed}")


# ─── Шаг 5: Эффект query rewrite ─────────────────────────────────────────────

def step5_rewrite_effect(results, cohere_ok: bool) -> None:
    """Показать влияние query rewrite на quality."""
    if not results or not cohere_ok:
        return

    cohere_res = [r for r in results if r.pipeline_name == "hybrid+cohere"]
    full_res = [r for r in results if r.pipeline_name == "hybrid+rewrite+cohere"]

    if not cohere_res or not full_res:
        return

    print("\n" + "═" * 70)
    print("  ШАГ 5: Эффект Query Rewrite (hybrid+cohere vs hybrid+rewrite+cohere)")
    print("═" * 70)

    cohere_by_q = {r.question_id: r for r in cohere_res}
    full_by_q = {r.question_id: r for r in full_res}

    improvements = []
    degradations = []
    same = []

    print(f"\n  {'#':<3} {'Вопрос':<42} {'cohere':>8} {'full':>8} {'delta':>7}")
    print("  " + "─" * 72)
    for qid in sorted(cohere_by_q):
        cr = cohere_by_q[qid]
        fr = full_by_q.get(qid)
        if not fr:
            continue
        delta = fr.keyword_rate - cr.keyword_rate
        q_short = cr.question[:40] + ".." if len(cr.question) > 40 else cr.question
        sign = "+" if delta > 0 else ""
        print(f"  {qid:<3} {q_short:<42} {cr.keyword_rate:>8.2f} {fr.keyword_rate:>8.2f} {sign}{delta:>6.2f}")
        if delta > 0.05:
            improvements.append(qid)
        elif delta < -0.05:
            degradations.append(qid)
        else:
            same.append(qid)

    avg_cohere = sum(r.keyword_rate for r in cohere_res) / len(cohere_res)
    avg_full = sum(r.keyword_rate for r in full_res) / len(full_res)
    print("  " + "─" * 72)
    print(f"  {'AVG':<46} {avg_cohere:>8.2f} {avg_full:>8.2f} {avg_full - avg_cohere:>+7.2f}")

    print(f"\n  Query rewrite помог (delta > 0.05): вопросы {improvements or 'нет'}")
    print(f"  Стало хуже (delta < -0.05):         вопросы {degradations or 'нет'}")


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    db_path = os.environ.get("RAG_DB_PATH", DEFAULT_DB)

    store, hybrid_ret, llm_fn, cohere_ok, ollama_ok = step1_init(db_path)

    step2_single_question(hybrid_ret, llm_fn, cohere_ok, ollama_ok)

    results = step3_full_eval(hybrid_ret, llm_fn, cohere_ok, ollama_ok)

    step4_cohere_vs_ollama(results, cohere_ok, ollama_ok)

    step5_rewrite_effect(results, cohere_ok)

    print("\n" + "═" * 70)
    print("  Готово.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
