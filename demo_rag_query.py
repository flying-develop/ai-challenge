#!/usr/bin/env python3
"""Demo script for RAG query pipeline."""
import argparse
import os
import sys

# Add rag_indexer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag_indexer"))

from src.storage.index_store import IndexStore
from src.embedding.provider import EmbeddingProvider
from src.retrieval.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from src.retrieval.rag_query import RAGQueryBuilder
from src.retrieval.evaluator import RAGEvaluator, EVAL_QUESTIONS


DEFAULT_DB = os.path.join(os.path.dirname(__file__), "rag_indexer", "output", "index.db")
DEFAULT_QUERY = "Как установить Podkop на OpenWrt?"


def make_llm_fn():
    """Create an LLM callable using DashScope/Qwen API."""
    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("QWEN_MODEL", "qwen-plus")

    if not api_key:
        print("[WARN] QWEN_API_KEY not set — LLM answers will be skipped", file=sys.stderr)
        return None

    def llm_fn(system: str, user: str) -> str:
        import urllib.request
        import json as _json
        payload = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1024,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    return llm_fn


def demo_single_query(query: str, retrievers: dict, llm_fn, top_k: int = 5):
    """Run a single query in all requested modes and print side-by-side."""
    builder = RAGQueryBuilder()
    print(f"\n{'='*72}")
    print(f"ЗАПРОС: {query}")
    print(f"{'='*72}\n")

    for name, retriever in retrievers.items():
        print(f"--- [{name.upper()}] ---")
        results = retriever.search(query, top_k=top_k)
        if not results:
            print("  Нет результатов.\n")
            continue

        print(f"Найдено {len(results)} чанков:")
        seen = set()
        for r in results:
            if r.source not in seen:
                print(f"  [{r.score:.3f}] {r.source} / {r.section or '-'}")
                seen.add(r.source)

        if llm_fn:
            ctx = builder.build(query, results)
            try:
                answer = llm_fn(ctx.system_prompt, ctx.user_prompt)
                print(f"\nОтвет:\n{answer}")
            except Exception as e:
                print(f"\n[Ошибка LLM: {e}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="RAG demo for Podkop wiki")
    parser.add_argument("--query", "-q", default=DEFAULT_QUERY, help="Запрос")
    parser.add_argument("--mode", choices=["no_rag", "vector", "bm25", "hybrid", "all"], default="all")
    parser.add_argument("--eval", action="store_true", help="Запустить evaluation (10 вопросов)")
    parser.add_argument("--db", default=DEFAULT_DB, help="Путь к index.db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", default=None, help="Фильтр chunking strategy")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[ERROR] Индекс не найден: {args.db}", file=sys.stderr)
        print("Сначала запустите: python rag_indexer/src/pipeline/run_indexing.py", file=sys.stderr)
        sys.exit(1)

    print(f"Загрузка индекса: {args.db}")
    store = IndexStore(args.db)
    embedder = EmbeddingProvider.create("qwen")

    vector_ret = VectorRetriever(store, embedder, args.strategy)
    bm25_ret = BM25Retriever(store, args.strategy)
    hybrid_ret = HybridRetriever(vector_ret, bm25_ret)

    all_retrievers = {
        "vector": vector_ret,
        "bm25": bm25_ret,
        "hybrid": hybrid_ret,
    }

    if args.mode == "all":
        retrievers = all_retrievers
    elif args.mode == "no_rag":
        retrievers = {}
    else:
        retrievers = {args.mode: all_retrievers[args.mode]}

    llm_fn = make_llm_fn()

    if args.eval:
        print("\nЗапуск evaluation (10 вопросов × retrievers)...")
        evaluator = RAGEvaluator(
            list(all_retrievers.values()),
            llm_fn=llm_fn,
            top_k=args.top_k,
        )
        results = evaluator.run()
        evaluator.print_report(results)
    else:
        if retrievers:
            demo_single_query(args.query, retrievers, llm_fn, top_k=args.top_k)
        else:
            print("Режим no_rag: запрос идёт напрямую в LLM без контекста.")
            if llm_fn:
                answer = llm_fn("Ты — ассистент по документации Podkop.", args.query)
                print(f"\nОтвет:\n{answer}")


if __name__ == "__main__":
    main()
