"""Демо-скрипт: структурированный RAG с цитатами и источниками (День 24).

Шаги:
    1. Один вопрос из документации — полный форматированный ответ с цитатами.
    2. Антивопрос — ожидаем is_refusal=True.
    3. Прогон 10 основных + 3 антивопроса — сводная таблица.
    4. Сравнение: старый промпт (без цитат) vs новый — keyword_hit_rate.
    5. Итоговые метрики.

Запуск:
    python demo_citations.py

Требования:
    - rag_indexer/output/index.db (предварительно проиндексировать)
    - DASHSCOPE_API_KEY или QWEN_API_KEY в .env
    - QWEN_API_KEY или OPENAI_API_KEY для LLM
"""
from __future__ import annotations

import os
import sys
import time

# ── Загрузка .env ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Путь к RAG-модулям ───────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_rag_dir = os.path.join(_script_dir, "rag_indexer")
if _rag_dir not in sys.path:
    sys.path.insert(0, _rag_dir)

# ── Импорты RAG ───────────────────────────────────────────────────────────────
try:
    from src.storage.index_store import IndexStore
    from src.embedding.provider import EmbeddingProvider
    from src.retrieval.retriever import VectorRetriever, BM25Retriever, HybridRetriever
    from src.retrieval.pipeline import RAGPipeline
    from src.retrieval.evaluator import EVAL_QUESTIONS, ANTI_QUESTIONS
    from src.retrieval.confidence import ConfidenceScorer, ConfidenceLevel
    from src.retrieval.formatter import (
        format_structured_response,
        format_refusal,
        format_confidence_level,
    )
    from src.retrieval.rag_query import RAGQueryBuilder
except ImportError as exc:
    print(f"[ОШИБКА] Не удалось импортировать RAG-модули: {exc}")
    print("Убедитесь, что индекс создан: cd rag_indexer && python main.py index")
    sys.exit(1)


# ── LLM-функция (через mcp_server или прямой API) ───────────────────────────
def _make_llm_fn():
    """Создаёт LLM-функцию из доступных провайдеров."""
    try:
        from mcp_server.llm_client import create_llm_fn
        return create_llm_fn(timeout=60.0)
    except Exception:
        pass

    # Fallback: прямой вызов через openai-совместимый API
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if qwen_key:
        import urllib.request
        import json

        base_url = os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        model = os.environ.get("QWEN_MODEL", "qwen-plus")

        def _qwen_llm(system: str, user: str) -> str:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 1500,
            }).encode()
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {qwen_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]

        return _qwen_llm

    raise RuntimeError(
        "Нет доступного LLM. Задайте QWEN_API_KEY или OPENAI_API_KEY в .env"
    )


# ── Инициализация ретривера ───────────────────────────────────────────────────
def _init_retriever(db_path: str):
    """Инициализирует HybridRetriever из индекса."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Индекс не найден: {db_path}\n"
            "Создайте индекс: cd rag_indexer && python main.py index"
        )
    store = IndexStore(db_path)
    embedder = EmbeddingProvider.create("qwen")
    vector_ret = VectorRetriever(store, embedder)
    bm25_ret = BM25Retriever(store)
    return HybridRetriever(vector_ret, bm25_ret), store


# ── Вспомогательные функции ───────────────────────────────────────────────────
def _print_separator(title: str = "", char: str = "═", width: int = 70) -> None:
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{'═' * side} {title} {'═' * (width - side - len(title) - 2)}")
    else:
        print(char * width)


def _keyword_hits(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 1: Один вопрос — полный разбор
# ════════════════════════════════════════════════════════════════════════════
def step1_single_question(pipeline: RAGPipeline) -> None:
    _print_separator("ШАГ 1: Один вопрос — полный форматированный ответ")
    question = "Как установить podkop на роутер?"
    print(f"Вопрос: {question}\n")

    rag_answer = pipeline.answer(question, top_k=5, initial_k=20)
    sr = rag_answer.structured

    if sr is None:
        print("[WARN] Structured response не получен")
        print(rag_answer.answer)
        return

    # Вывод
    if sr.is_refusal:
        print(format_refusal(sr))
    else:
        print(format_structured_response(sr))

    # Confidence
    scorer = ConfidenceScorer()
    level = (
        ConfidenceLevel.HIGH if sr.confidence >= scorer.HIGH_THRESHOLD
        else ConfidenceLevel.MEDIUM if sr.confidence >= scorer.DEFAULT_THRESHOLD
        else ConfidenceLevel.LOW
    )
    print(f"\n{format_confidence_level(level, sr.confidence)}")
    print(f"Время: retrieval={rag_answer.retrieval_time_ms:.0f}ms  llm={rag_answer.llm_time_ms:.0f}ms")


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 2: Антивопрос
# ════════════════════════════════════════════════════════════════════════════
def step2_anti_question(pipeline: RAGPipeline) -> None:
    _print_separator("ШАГ 2: Антивопрос — ожидаем отказ")
    question = "Как настроить podkop для майнинга биткоинов?"
    print(f"Вопрос: {question}\n")
    print("Ожидание: is_refusal=True, sources=[], quotes=[]\n")

    rag_answer = pipeline.answer(question, top_k=5, initial_k=20)
    sr = rag_answer.structured

    if sr is None:
        print("[WARN] Structured response не получен")
        print(rag_answer.answer)
        return

    print(format_refusal(sr) if sr.is_refusal else format_structured_response(sr))

    result_icon = "✅" if sr.is_refusal else "❌"
    print(f"\n{result_icon} is_refusal={sr.is_refusal}  confidence={sr.confidence:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 3: Прогон 10 + 3 вопросов
# ════════════════════════════════════════════════════════════════════════════
def step3_full_eval(pipeline: RAGPipeline) -> dict:
    _print_separator("ШАГ 3: Прогон 10 основных + 3 антивопроса")

    scorer = ConfidenceScorer()

    # Основные вопросы
    print("\nОсновные вопросы...\n")
    main_rows = []
    for q in EVAL_QUESTIONS:
        print(f"  [{q['id']}] {q['question'][:50]}...", end=" ", flush=True)
        try:
            rag = pipeline.answer(q["question"], top_k=5, initial_k=20)
        except Exception as exc:
            print(f"ОШИБКА: {exc}")
            continue

        sr = rag.structured
        if sr:
            row = {
                "id": q["id"],
                "question": q["question"],
                "num_src": len(sr.sources),
                "num_q": len(sr.quotes),
                "ver": len(sr.verified_quotes),
                "total_q": len(sr.quotes),
                "ver_pct": int(sr.verified_ratio * 100) if sr.quotes else 0,
                "conf": rag.confidence,
                "is_ref": sr.is_refusal,
            }
        else:
            row = {"id": q["id"], "question": q["question"],
                   "num_src": 0, "num_q": 0, "ver": 0, "total_q": 0,
                   "ver_pct": 0, "conf": rag.confidence, "is_ref": False}
        main_rows.append(row)
        print(f"conf={row['conf']:.2f}  quotes={row['num_q']}  verified={row['ver_pct']}%")

    # Таблица основных вопросов
    w = 33
    print(f"\n╔═════╦{'═' * w}╦═══════╦════════╦════════╦══════════╗")
    print(f"║  #  ║ {'Вопрос':<{w - 1}}║  Src  ║ Quotes ║ Verif  ║  Conf    ║")
    print(f"╠═════╬{'═' * w}╬═══════╬════════╬════════╬══════════╣")
    sum_src = sum_q = sum_ver = sum_total_q = 0
    sum_conf = 0.0
    for row in main_rows:
        q_short = row["question"][: w - 2]
        ver_str = f"{row['ver_pct']}%" if row["total_q"] else "—"
        print(
            f"║ {row['id']:<4}║ {q_short:<{w - 1}}║ {row['num_src']:<5} ║ {row['num_q']:<6} ║ {ver_str:<6} ║ {row['conf']:.2f}     ║"
        )
        sum_src += row["num_src"]
        sum_q += row["num_q"]
        sum_conf += row["conf"]
        if row["total_q"]:
            sum_ver += row["ver"]
            sum_total_q += row["total_q"]
    n = len(main_rows) or 1
    avg_src = sum_src / n
    avg_q = sum_q / n
    avg_conf = sum_conf / n
    avg_ver_pct = int(sum_ver / sum_total_q * 100) if sum_total_q else 0
    print(f"╠═════╬{'═' * w}╬═══════╬════════╬════════╬══════════╣")
    print(
        f"║ AVG ║ {'':<{w - 1}}║ {avg_src:<5.1f} ║ {avg_q:<6.1f} ║ {avg_ver_pct}%    ║ {avg_conf:.2f}     ║"
    )
    print(f"╚═════╩{'═' * w}╩═══════╩════════╩════════╩══════════╝")

    # Антивопросы
    print("\nАнтивопросы...\n")
    anti_rows = []
    for aq in ANTI_QUESTIONS:
        print(f"  [{aq['id']}] {aq['question'][:50]}...", end=" ", flush=True)
        try:
            rag = pipeline.answer(aq["question"], top_k=5, initial_k=20)
        except Exception as exc:
            print(f"ОШИБКА: {exc}")
            continue
        sr = rag.structured
        is_ref = sr.is_refusal if sr else False
        conf = rag.confidence
        anti_rows.append({"id": aq["id"], "question": aq["question"],
                           "is_ref": is_ref, "conf": conf})
        icon = "✅" if is_ref else "❌"
        print(f"{icon}  conf={conf:.2f}")

    print(f"\n╔═════╦{'═' * w}╦══════════╦═══════╗")
    print(f"║  #  ║ {'Вопрос':<{w - 1}}║ Отказал? ║ Conf  ║")
    print(f"╠═════╬{'═' * w}╬══════════╬═══════╣")
    for row in anti_rows:
        q_short = row["question"][: w - 2]
        ref_str = "✅ да" if row["is_ref"] else "❌ нет"
        print(f"║ {row['id']:<4}║ {q_short:<{w - 1}}║ {ref_str:<8} ║ {row['conf']:.2f}  ║")
    print(f"╚═════╩{'═' * w}╩══════════╩═══════╝")

    return {
        "main_rows": main_rows,
        "anti_rows": anti_rows,
        "avg_src": avg_src,
        "avg_q": avg_q,
        "avg_ver_pct": avg_ver_pct,
        "avg_conf": avg_conf,
        "sum_ver": sum_ver,
        "sum_total_q": sum_total_q,
    }


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 4: Сравнение старый vs новый промпт
# ════════════════════════════════════════════════════════════════════════════
def step4_comparison(retriever, llm_fn) -> None:
    _print_separator("ШАГ 4: Сравнение — без цитат vs с цитатами")

    old_pipeline = RAGPipeline(retriever=retriever, llm_fn=llm_fn, use_structured=False)
    new_pipeline = RAGPipeline(retriever=retriever, llm_fn=llm_fn, use_structured=True)

    questions = EVAL_QUESTIONS[:3]  # берём первые 3 для краткости
    print(f"\nСравниваем на {len(questions)} вопросах...\n")

    results_old = []
    results_new = []

    for q in questions:
        print(f"  [{q['id']}] {q['question'][:50]}...", end=" ", flush=True)

        # Старый промпт
        try:
            old_ans = old_pipeline.answer(q["question"], top_k=5, initial_k=20)
            old_kw = _keyword_hits(old_ans.answer, q["keywords"])
            old_ver = 0
        except Exception as exc:
            old_kw = 0
            old_ver = 0

        # Новый промпт
        try:
            new_ans = new_pipeline.answer(q["question"], top_k=5, initial_k=20)
            new_kw = _keyword_hits(new_ans.answer, q["keywords"])
            sr = new_ans.structured
            new_ver = int(sr.verified_ratio * 100) if (sr and sr.quotes) else 0
        except Exception as exc:
            new_kw = 0
            new_ver = 0

        results_old.append(old_kw / len(q["keywords"]) if q["keywords"] else 0)
        results_new.append(new_kw / len(q["keywords"]) if q["keywords"] else 0)
        print(f"kw_old={old_kw}/{len(q['keywords'])}  kw_new={new_kw}/{len(q['keywords'])}  ver={new_ver}%")

    avg_old = sum(results_old) / len(results_old) if results_old else 0.0
    avg_new = sum(results_new) / len(results_new) if results_new else 0.0

    print(f"\n  Средний keyword_hit_rate:")
    print(f"    Без цитат (старый):  {avg_old:.2f}")
    print(f"    С цитатами (новый):  {avg_new:.2f}")
    delta = avg_new - avg_old
    sign = "+" if delta >= 0 else ""
    print(f"    Изменение:           {sign}{delta:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 5: Итоговые метрики
# ════════════════════════════════════════════════════════════════════════════
def step5_summary(eval_data: dict) -> None:
    _print_separator("ШАГ 5: Итоговые метрики")

    main_rows = eval_data.get("main_rows", [])
    anti_rows = eval_data.get("anti_rows", [])
    n_main = len(main_rows) or 1

    with_src = sum(1 for r in main_rows if r["num_src"] > 0)
    with_q = sum(1 for r in main_rows if r["num_q"] > 0)
    refusals_ok = sum(1 for r in anti_rows if r["is_ref"])
    false_refusals = sum(1 for r in main_rows if r["is_ref"])
    avg_ver = eval_data.get("avg_ver_pct", 0)

    print(f"\n  Общие показатели:")
    print(f"    Вопросов с источниками:     {with_src}/{n_main} ({int(with_src / n_main * 100)}%)")
    print(f"    Вопросов с цитатами:        {with_q}/{n_main} ({int(with_q / n_main * 100)}%)")
    print(f"    Средний verified_ratio:     {avg_ver}%")
    print(f"    Антивопросов с отказом:     {refusals_ok}/{len(anti_rows)} ({'100%' if len(anti_rows) and refusals_ok == len(anti_rows) else f'{int(refusals_ok / max(len(anti_rows), 1) * 100)}%'})")
    print(f"    Ложных отказов:             {false_refusals}/{n_main} ({int(false_refusals / n_main * 100)}%)")

    # Критерии готовности
    print("\n  Критерии готовности:")
    checks = [
        ("Вопросов с источниками = 100%", with_src == n_main),
        ("Вопросов с цитатами = 100%", with_q == n_main),
        ("Средний verified_ratio ≥ 80%", avg_ver >= 80),
        ("Антивопросов с отказом = 100%", refusals_ok == len(anti_rows) if anti_rows else True),
        ("Ложных отказов = 0%", false_refusals == 0),
    ]
    for label, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"    {icon} {label}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    db_path = os.path.join(_script_dir, "rag_indexer", "output", "index.db")

    print("=" * 70)
    print("  DEMO: Structured RAG с цитатами и источниками (День 24)")
    print("=" * 70)
    print(f"\n  База: {db_path}")

    # Инициализация
    print("\nИнициализация...", end=" ", flush=True)
    try:
        retriever, store = _init_retriever(db_path)
        print("ретривер ✓")
    except FileNotFoundError as exc:
        print(f"\n[ОШИБКА] {exc}")
        return

    print("LLM...", end=" ", flush=True)
    try:
        llm_fn = _make_llm_fn()
        print("✓\n")
    except RuntimeError as exc:
        print(f"\n[ОШИБКА] {exc}")
        return

    # Structured pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        use_structured=True,
    )

    # ── Шаги ─────────────────────────────────────────────────────────────
    step1_single_question(pipeline)
    step2_anti_question(pipeline)
    eval_data = step3_full_eval(pipeline)
    step4_comparison(retriever, llm_fn)
    step5_summary(eval_data)

    print("\n" + "=" * 70)
    print("  Демо завершено.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
