#!/usr/bin/env python3
"""День 29: оптимизация локальной модели под RAG support-бот.

Три оси оптимизации:
    1. Параметры генерации — 4 конфигурации (baseline, precise, fast, rag_tuned)
    2. Квантизация       — Q4_K_M (baseline) vs Q8_0 vs Q4_0
    3. Промпт-шаблон     — базовый системный промпт vs Modelfile с запечёнными параметрами

Шаги:
    1. Health check (Ollama + индекс)
    2. Сравнение параметров генерации (4 конфига × 3 вопроса)
    3. Сравнение квантизаций (baseline vs q8 vs q4, если загружены)
    4. Сравнение промпт-шаблонов (базовый vs Modelfile.rag)
    5. Итоговые таблицы
    6. Выводы и рекомендация

Подготовка (одноразово):
    # Создать индекс если нет:
    cd rag_indexer && python main.py index --docs ../podkop-wiki/content \\
        --db output/index_local.db --embedder ollama

    # Создать Modelfile-модели (опционально):
    ollama create qwen2.5-rag -f modelfiles/Modelfile.rag
    ollama create qwen2.5-q8 -f modelfiles/Modelfile.q8
    ollama create qwen2.5-q4 -f modelfiles/Modelfile.q4

Запуск:
    python demo_optimization.py            # полный прогон
    python demo_optimization.py --quick    # 3 вопроса из каждого набора
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Вопросы для оптимизационного бенчмарка
# ---------------------------------------------------------------------------

OPT_QUESTIONS = [
    {
        "id": 1,
        "question": "Как установить Podkop на OpenWrt?",
        "keywords": ["установить", "openwrt", "podkop", "установка"],
    },
    {
        "id": 2,
        "question": "Как настроить WireGuard туннель в Podkop?",
        "keywords": ["wireguard", "туннель", "настройка", "wg"],
    },
    {
        "id": 3,
        "question": "Что делать если заблокированные сайты не открываются?",
        "keywords": ["заблокированные", "сайты", "не открываются", "troubleshooting"],
    },
    {
        "id": 4,
        "question": "Как совместить Podkop с AdGuard Home?",
        "keywords": ["adguard", "adguard home", "совместить", "интеграция"],
    },
    {
        "id": 5,
        "question": "В чём разница между sing-box, xray и mihomo?",
        "keywords": ["sing-box", "xray", "mihomo", "разница"],
    },
]

QUICK_COUNT = 3  # вопросов в --quick режиме


# ---------------------------------------------------------------------------
# Dataclass результатов
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    """Результат одного прогона для одного вопроса и одной конфигурации."""

    question_id: int
    question: str
    config_name: str           # имя конфига/модели/шаблона

    keyword_hit: float = 0.0   # доля ожидаемых keywords в ответе (0..1)
    has_answer_block: bool = False
    has_sources_block: bool = False
    has_quotes_block: bool = False
    is_refusal: bool = False   # ответ содержит "не найдена" или похожее

    llm_time_ms: float = 0.0   # только время LLM (без поиска)
    total_time_ms: float = 0.0  # retrieval + llm

    answer_preview: str = ""   # первые 200 символов ответа


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _sep(title: str = "", width: int = 64) -> None:
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("=" * pad + f" {title} " + "=" * max(0, width - pad - len(title) - 2))
    else:
        print("=" * width)


def _keyword_hit(answer: str, keywords: list[str]) -> float:
    """Доля ключевых слов, найденных в ответе."""
    if not keywords:
        return 0.0
    lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords)


def _has_block(answer: str, block: str) -> bool:
    return f"[{block}]" in answer


def _is_refusal(answer: str) -> bool:
    markers = ["не найдена", "нет информации", "не содержит", "не могу"]
    lower = answer.lower()
    return any(m in lower for m in markers)


def _check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """Проверить доступность Ollama и загруженность модели."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        found = any(m == model or m == f"{model}:latest" for m in models)
        if found:
            return True, f"загружена ({model})"
        available = ", ".join(models[:5]) or "(нет моделей)"
        return False, f"не найдена. Загрузите: ollama pull {model}"
    except Exception as exc:
        return False, f"сервер недоступен: {exc}"


def _check_index(db_path: Path) -> tuple[bool, str]:
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
        return False, f"ошибка: {exc}"


# ---------------------------------------------------------------------------
# Шаг 1: Health check
# ---------------------------------------------------------------------------

def step1_health_check(
    local_db: Path,
    base_url: str,
    llm_model: str,
    embed_model: str,
) -> bool:
    _sep("Шаг 1: Проверка стека")

    llm_ok, llm_msg = _check_ollama(base_url, llm_model)
    emb_ok, emb_msg = _check_ollama(base_url, embed_model)
    idx_ok, idx_msg = _check_index(local_db)

    print(f"  {'✅' if llm_ok else '❌'} LLM ({llm_model}): {llm_msg}")
    print(f"  {'✅' if emb_ok else '❌'} Embed ({embed_model}): {emb_msg}")
    print(f"  {'✅' if idx_ok else '❌'} Индекс: {idx_msg}")

    if not local_db.exists():
        print(
            f"\n  Для создания индекса выполните:\n"
            f"    cd rag_indexer\n"
            f"    python main.py index --docs ../podkop-wiki/content \\\n"
            f"      --db output/index_local.db --embedder ollama"
        )

    ok = llm_ok and emb_ok and idx_ok
    print()
    if not ok:
        print("  ❌ Стек не готов. Исправьте ошибки выше.")
    else:
        print("  ✅ Стек готов.")
    return ok


# ---------------------------------------------------------------------------
# Построение пайплайна с заданными параметрами
# ---------------------------------------------------------------------------

def _build_pipeline(
    local_db: Path,
    base_url: str,
    llm_model: str,
    embed_model: str,
    options: dict | None = None,
):
    """Собрать RAGPipeline с заданными параметрами LLM."""
    from rag_indexer.src.embedding.ollama_embedder import OllamaEmbedder
    from rag_indexer.src.retrieval.reranker import ThresholdFilter
    from rag_indexer.src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever
    from rag_indexer.src.retrieval.pipeline import RAGPipeline
    from rag_indexer.src.storage.index_store import IndexStore
    from llm_agent.infrastructure.ollama_client import OllamaHttpClient
    from llm_agent.domain.models import ChatMessage

    embedder  = OllamaEmbedder(model=embed_model, base_url=base_url)
    store     = IndexStore(local_db)
    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(store=store, embedder=embedder),
        bm25_retriever=BM25Retriever(store=store),
    )
    reranker = ThresholdFilter(threshold=0.0)

    llm_client = OllamaHttpClient(
        model=llm_model,
        base_url=base_url,
        timeout=300.0,
        options=options,
    )

    def llm_fn(system: str, user: str) -> str:
        msgs = []
        if system:
            msgs.append(ChatMessage(role="system", content=system))
        msgs.append(ChatMessage(role="user", content=user))
        return llm_client.generate(msgs).text

    return RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        reranker=reranker,
        use_structured=True,
    )


# ---------------------------------------------------------------------------
# Прогон вопросов через один пайплайн
# ---------------------------------------------------------------------------

def _run_questions(
    pipeline,
    questions: list[dict],
    config_name: str,
) -> list[OptResult]:
    results = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            rag_answer = pipeline.answer(q["question"], top_k=5, initial_k=20)
            total_ms = (time.perf_counter() - t0) * 1000
            answer = rag_answer.answer
            llm_ms = rag_answer.llm_time_ms
        except Exception as exc:
            total_ms = (time.perf_counter() - t0) * 1000
            answer = f"[ERROR] {exc}"
            llm_ms = 0.0

        results.append(OptResult(
            question_id=q["id"],
            question=q["question"],
            config_name=config_name,
            keyword_hit=_keyword_hit(answer, q["keywords"]),
            has_answer_block=_has_block(answer, "ANSWER"),
            has_sources_block=_has_block(answer, "SOURCES"),
            has_quotes_block=_has_block(answer, "QUOTES"),
            is_refusal=_is_refusal(answer),
            llm_time_ms=llm_ms,
            total_time_ms=total_ms,
            answer_preview=answer[:200].replace("\n", " "),
        ))
        icon = "✓" if results[-1].keyword_hit >= 0.5 else "✗"
        print(
            f"    [{icon}] Q{q['id']}: кwds={results[-1].keyword_hit:.0%} "
            f"llm={llm_ms:.0f}ms total={total_ms:.0f}ms"
        )
    return results


# ---------------------------------------------------------------------------
# Шаг 2: Сравнение параметров генерации
# ---------------------------------------------------------------------------

def step2_params(
    local_db: Path,
    questions: list[dict],
    base_url: str,
    llm_model: str,
    embed_model: str,
) -> dict[str, list[OptResult]]:
    _sep("Шаг 2: Параметры генерации")
    from src.providers.ollama_config import CONFIGS

    all_results: dict[str, list[OptResult]] = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n  [{cfg_name}] {cfg.description}")
        print(f"  {cfg}")
        pipeline = _build_pipeline(
            local_db, base_url, llm_model, embed_model,
            options=cfg.to_options(),
        )
        results = _run_questions(pipeline, questions, config_name=cfg_name)
        all_results[cfg_name] = results

    return all_results


# ---------------------------------------------------------------------------
# Шаг 3: Сравнение квантизаций
# ---------------------------------------------------------------------------

def step3_quantization(
    local_db: Path,
    questions: list[dict],
    base_url: str,
    embed_model: str,
) -> dict[str, list[OptResult]]:
    _sep("Шаг 3: Квантизация")

    # Модели квантизации: имя_в_ollama → метка
    # qwen2.5:1.5b — дефолтная Q4_K_M
    # qwen2.5:1.5b-q8_0 / q4_0 — скачиваются через ollama pull
    # qwen2.5-q8 / q4 — RAG-обёртки через ollama create -f modelfiles/Modelfile.q*
    base_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
    quant_models = [
        (base_model,                    "Q4_K_M (baseline)"),
        ("qwen2.5:1.5b-q8_0",   "Q8_0 (pull)"),
        ("qwen2.5-q8",                  "Q8_0 (Modelfile)"),
        ("qwen2.5:1.5b-q4_0",   "Q4_0 (pull)"),
        ("qwen2.5-q4",                  "Q4_0 (Modelfile)"),
    ]

    # Параметры RAG-оптимизированные для честного сравнения
    from src.providers.ollama_config import CONFIGS
    options = CONFIGS["rag_tuned"].to_options()

    all_results: dict[str, list[OptResult]] = {}
    for model_name, label in quant_models:
        ok, msg = _check_ollama(base_url, model_name)
        if not ok:
            print(f"\n  [{label}] ПРОПУЩЕНО — {msg}")
            continue
        print(f"\n  [{label}] model={model_name}")
        pipeline = _build_pipeline(
            local_db, base_url, model_name, embed_model,
            options=options,
        )
        results = _run_questions(pipeline, questions, config_name=label)
        all_results[label] = results

    return all_results


# ---------------------------------------------------------------------------
# Шаг 4: Сравнение промпт-шаблонов
# ---------------------------------------------------------------------------

def step4_prompt_template(
    local_db: Path,
    questions: list[dict],
    base_url: str,
    embed_model: str,
) -> dict[str, list[OptResult]]:
    _sep("Шаг 4: Промпт-шаблон")

    base_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
    rag_model  = "qwen2.5-rag"  # создан из Modelfile.rag

    from src.providers.ollama_config import CONFIGS
    options = CONFIGS["rag_tuned"].to_options()
    # Для Modelfile-модели параметры уже запечёны — передаём без options
    options_baked: dict | None = None

    templates = [
        (base_model, "base_model",  options),
        (rag_model,  "modelfile",   options_baked),
    ]

    all_results: dict[str, list[OptResult]] = {}
    for model_name, label, opts in templates:
        ok, msg = _check_ollama(base_url, model_name)
        if not ok:
            print(f"\n  [{label}] ПРОПУЩЕНО — {msg}")
            continue
        print(f"\n  [{label}] model={model_name}")
        pipeline = _build_pipeline(
            local_db, base_url, model_name, embed_model,
            options=opts,
        )
        results = _run_questions(pipeline, questions, config_name=label)
        all_results[label] = results

    return all_results


# ---------------------------------------------------------------------------
# Шаг 5: Итоговые таблицы
# ---------------------------------------------------------------------------

def _print_results_table(
    title: str,
    all_results: dict[str, list[OptResult]],
    questions: list[dict],
) -> None:
    if not all_results:
        print(f"  [{title}] нет данных (все конфигурации пропущены)")
        return

    configs = list(all_results.keys())
    n_cfg = len(configs)
    q_col = 32
    c_col = 12

    total_width = 2 + q_col + 2 + n_cfg * (c_col + 3)
    inner = total_width - 2

    print()
    print("╔" + "═" * inner + "╗")
    print(f"║  {title:<{inner - 2}}║")
    print("╠" + "═" * inner + "╣")

    header = f"║ {'Вопрос':<{q_col}} "
    for cfg in configs:
        short = cfg[:c_col - 1]
        header += f"│ {short:^{c_col - 1}} "
    header += "║"
    print(header)
    print("╠" + "═" * inner + "╣")

    for q in questions:
        q_short = q["question"][:q_col - 1]
        row = f"║ {q_short:<{q_col}} "
        for cfg in configs:
            hits = {r.question_id: r for r in all_results[cfg]}
            r = hits.get(q["id"])
            if r:
                cell = f"{r.keyword_hit:.0%}"
            else:
                cell = "—"
            row += f"│ {cell:^{c_col - 1}} "
        row += "║"
        print(row)

    print("╠" + "═" * inner + "╣")

    # Средние keyword_hit и avg_llm_ms
    avg_row = f"║ {'AVG keyword_hit':<{q_col}} "
    for cfg in configs:
        rs = all_results[cfg]
        avg = sum(r.keyword_hit for r in rs) / len(rs) if rs else 0.0
        avg_row += f"│ {f'{avg:.0%}':^{c_col - 1}} "
    avg_row += "║"
    print(avg_row)

    spd_row = f"║ {'AVG llm_ms':<{q_col}} "
    for cfg in configs:
        rs = all_results[cfg]
        avg = sum(r.llm_time_ms for r in rs) / len(rs) if rs else 0.0
        spd_row += f"│ {f'{avg:.0f}ms':^{c_col - 1}} "
    spd_row += "║"
    print(spd_row)

    str_row = f"║ {'AVG struct (A+S+Q) %':<{q_col}} "
    for cfg in configs:
        rs = all_results[cfg]
        if rs:
            a = sum(r.has_answer_block for r in rs) / len(rs)
            s = sum(r.has_sources_block for r in rs) / len(rs)
            q_b = sum(r.has_quotes_block for r in rs) / len(rs)
            avg_struct = (a + s + q_b) / 3
            str_row += f"│ {f'{avg_struct:.0%}':^{c_col - 1}} "
        else:
            str_row += f"│ {'—':^{c_col - 1}} "
    str_row += "║"
    print(str_row)

    print("╚" + "═" * inner + "╝")


def step5_tables(
    params_results: dict[str, list[OptResult]],
    quant_results: dict[str, list[OptResult]],
    prompt_results: dict[str, list[OptResult]],
    questions: list[dict],
) -> None:
    _sep("Шаг 5: Итоговые таблицы")

    _print_results_table("ПАРАМЕТРЫ ГЕНЕРАЦИИ (keyword_hit)", params_results, questions)
    _print_results_table("КВАНТИЗАЦИЯ (keyword_hit)", quant_results, questions)
    _print_results_table("ПРОМПТ-ШАБЛОН (keyword_hit)", prompt_results, questions)


# ---------------------------------------------------------------------------
# Шаг 6: Выводы
# ---------------------------------------------------------------------------

def _best_config(results: dict[str, list[OptResult]]) -> tuple[str, float]:
    """Найти конфигурацию с лучшим средним keyword_hit."""
    best_name = ""
    best_score = -1.0
    for name, rs in results.items():
        if not rs:
            continue
        avg = sum(r.keyword_hit for r in rs) / len(rs)
        if avg > best_score:
            best_score = avg
            best_name = name
    return best_name, best_score


def _fastest_config(results: dict[str, list[OptResult]]) -> tuple[str, float]:
    """Найти конфигурацию с наименьшим средним llm_time_ms."""
    best_name = ""
    best_ms = float("inf")
    for name, rs in results.items():
        if not rs:
            continue
        avg = sum(r.llm_time_ms for r in rs) / len(rs)
        if avg < best_ms:
            best_ms = avg
            best_name = name
    return best_name, best_ms


def step6_conclusions(
    params_results: dict[str, list[OptResult]],
    quant_results: dict[str, list[OptResult]],
    prompt_results: dict[str, list[OptResult]],
) -> None:
    _sep("Шаг 6: Выводы")

    print("\n  Параметры генерации:")
    if params_results:
        best_q, best_q_score = _best_config(params_results)
        fast_q, fast_q_ms = _fastest_config(params_results)
        for name, rs in params_results.items():
            if not rs:
                continue
            avg_kw = sum(r.keyword_hit for r in rs) / len(rs)
            avg_ms = sum(r.llm_time_ms for r in rs) / len(rs)
            marker = " ← лучшее качество" if name == best_q else ""
            marker2 = " ← быстрее всего" if name == fast_q and name != best_q else ""
            print(f"    {name:<12}: kwd={avg_kw:.0%}  llm={avg_ms:.0f}ms{marker}{marker2}")
    else:
        print("    нет данных")

    print("\n  Квантизация:")
    if quant_results:
        best_q2, best_q2_score = _best_config(quant_results)
        fast_q2, fast_q2_ms = _fastest_config(quant_results)
        for name, rs in quant_results.items():
            if not rs:
                continue
            avg_kw = sum(r.keyword_hit for r in rs) / len(rs)
            avg_ms = sum(r.llm_time_ms for r in rs) / len(rs)
            marker = " ← лучшее качество" if name == best_q2 else ""
            marker2 = " ← быстрее всего" if name == fast_q2 and name != best_q2 else ""
            print(f"    {name:<20}: kwd={avg_kw:.0%}  llm={avg_ms:.0f}ms{marker}{marker2}")
    else:
        print("    нет данных (модели квантизации не загружены)")
        print("    Создайте: ollama create qwen2.5-q8 -f modelfiles/Modelfile.q8")
        print("              ollama create qwen2.5-q4 -f modelfiles/Modelfile.q4")

    print("\n  Промпт-шаблон:")
    if prompt_results:
        best_p, best_p_score = _best_config(prompt_results)
        for name, rs in prompt_results.items():
            if not rs:
                continue
            avg_kw = sum(r.keyword_hit for r in rs) / len(rs)
            avg_ms = sum(r.llm_time_ms for r in rs) / len(rs)
            a = sum(r.has_answer_block for r in rs) / len(rs)
            s = sum(r.has_sources_block for r in rs) / len(rs)
            q = sum(r.has_quotes_block for r in rs) / len(rs)
            struct_avg = (a + s + q) / 3
            marker = " ← лучшее качество" if name == best_p else ""
            print(
                f"    {name:<14}: kwd={avg_kw:.0%}  llm={avg_ms:.0f}ms  "
                f"struct={struct_avg:.0%}{marker}"
            )
    else:
        print("    нет данных (qwen2.5-rag не создан)")
        print("    Создайте: ollama create qwen2.5-rag -f modelfiles/Modelfile.rag")

    # Итоговая рекомендация
    print()
    _sep("Рекомендация", 64)
    best_overall = None
    best_score_overall = -1.0

    for results in [params_results, quant_results, prompt_results]:
        b, sc = _best_config(results)
        if sc > best_score_overall:
            best_score_overall = sc
            best_overall = b

    if best_overall:
        print(f"\n  Лучший результат по keyword_hit: [{best_overall}] = {best_score_overall:.0%}")

    print("""
  Для production support-бота на qwen2.5:1.5b рекомендуется:
    1. Параметры: rag_tuned (temp=0.2, seed=42, ctx=3072, predict=400)
       — детерминированные ответы, достаточный контекст для [ANSWER]/[SOURCES]/[QUOTES]
    2. Квантизация: Q8_0 если RAM позволяет, Q4_K_M (baseline) как компромисс
    3. Промпт-шаблон: Modelfile.rag с запечёнными параметрами и system prompt
       — меньше вероятность забыть передать правильный system prompt в коде

  Следующий шаг: протестировать на реальных пользовательских запросах
  и применить конфиг с лучшим recall к production-деплойменту.
""")


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main(quick: bool = False) -> None:
    _sep("День 29: Оптимизация локальной модели", 64)
    print(f"  Режим: {'quick (3 вопроса)' if quick else 'full (5 вопросов)'}")
    print()

    base_url    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model   = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    output_dir = _PROJECT_ROOT / "rag_indexer" / "output"
    local_db   = output_dir / "index_local.db"

    # --- Шаг 1 ---
    ok = step1_health_check(local_db, base_url, llm_model, embed_model)
    print()
    if not ok:
        print("[STOP] Стек не готов. Исправьте ошибки выше.")
        sys.exit(1)

    questions = OPT_QUESTIONS[:QUICK_COUNT] if quick else OPT_QUESTIONS

    # --- Шаг 2 ---
    print()
    params_results = step2_params(local_db, questions, base_url, llm_model, embed_model)

    # --- Шаг 3 ---
    print()
    quant_results = step3_quantization(local_db, questions, base_url, embed_model)

    # --- Шаг 4 ---
    print()
    prompt_results = step4_prompt_template(local_db, questions, base_url, embed_model)

    # --- Шаг 5 ---
    print()
    step5_tables(params_results, quant_results, prompt_results, questions)

    # --- Шаг 6 ---
    print()
    step6_conclusions(params_results, quant_results, prompt_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="День 29: оптимизация локальной модели под RAG support-бот"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый режим: только 3 вопроса",
    )
    args = parser.parse_args()
    main(quick=args.quick)
