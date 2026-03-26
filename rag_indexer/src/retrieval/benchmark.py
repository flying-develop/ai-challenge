"""RAG Benchmark: сравнение local vs cloud стеков.

Запускает один и тот же набор вопросов через два стека (local и cloud)
и сравнивает ответы по метрикам качества, скорости и стабильности.

Результат: пять сравнительных таблиц + автоматические выводы.

Использование::

    from src.retrieval.benchmark import RAGBenchmark

    benchmark = RAGBenchmark(
        local_pipeline=local_rag,
        cloud_pipeline=cloud_rag,
    )
    results, anti_results, stab_results = benchmark.run(quick=False)
    benchmark.print_all_tables(results, anti_results, stab_results)
    benchmark.print_conclusions(results, anti_results, stab_results)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .pipeline import RAGPipeline
from .evaluator import EVAL_QUESTIONS, ANTI_QUESTIONS


# ---------------------------------------------------------------------------
# Вопросы для теста стабильности (повторяются 3 раза)
# ---------------------------------------------------------------------------

STABILITY_QUESTIONS = [
    {
        "id": 14,
        "question": "Как установить podkop?",
        "keywords": ["установить", "openwrt", "podkop", "установка"],
        "runs": 3,
    },
    {
        "id": 15,
        "question": "Какие протоколы поддерживает podkop?",
        "keywords": ["wireguard", "vless", "trojan", "протокол", "туннель"],
        "runs": 3,
    },
]

# Количество вопросов в быстром режиме
QUICK_MODE_COUNT = 3


# ---------------------------------------------------------------------------
# Dataclasses результатов
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Результат сравнения двух стеков на одном вопросе."""

    question_id: int
    question: str

    # --- Качество ---
    local_keyword_hit: float = 0.0      # доля ожидаемых keywords в ответе (0..1)
    cloud_keyword_hit: float = 0.0
    local_has_sources: bool = False     # есть ли блок [SOURCES] в ответе
    cloud_has_sources: bool = False
    local_has_quotes: bool = False      # есть ли блок [QUOTES] в ответе
    cloud_has_quotes: bool = False
    local_verified_ratio: float = 0.0  # доля верифицированных цитат
    cloud_verified_ratio: float = 0.0
    local_is_refusal: bool = False      # модель отказалась отвечать
    cloud_is_refusal: bool = False

    # --- Скорость ---
    local_retrieval_ms: float = 0.0    # время поиска (embed query + vector search)
    cloud_retrieval_ms: float = 0.0
    local_rerank_ms: float = 0.0       # время реранкинга
    cloud_rerank_ms: float = 0.0
    local_llm_ms: float = 0.0          # время генерации LLM
    cloud_llm_ms: float = 0.0
    local_total_ms: float = 0.0        # сумма всех этапов
    cloud_total_ms: float = 0.0

    # --- Ошибки ---
    local_error: Optional[str] = None
    cloud_error: Optional[str] = None

    # --- Ответы ---
    local_answer: str = ""
    cloud_answer: str = ""
    local_sources: list[str] = field(default_factory=list)
    cloud_sources: list[str] = field(default_factory=list)


@dataclass
class AntiResult:
    """Результат проверки антивопроса (ожидается отказ модели)."""

    question_id: int
    question: str
    local_is_refusal: bool = False     # True = правильно отказал
    cloud_is_refusal: bool = False
    local_error: Optional[str] = None
    cloud_error: Optional[str] = None
    local_total_ms: float = 0.0
    cloud_total_ms: float = 0.0


@dataclass
class StabilityResult:
    """Результат теста стабильности (N прогонов одного вопроса)."""

    question_id: int
    question: str
    local_stability: float = 0.0    # среднее попарное пересечение keywords (0..1)
    cloud_stability: float = 0.0
    local_error: Optional[str] = None
    cloud_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _extract_words(text: str) -> set[str]:
    """Извлечь множество значимых слов (≥ 3 символов) из текста."""
    return {w for w in re.findall(r'\w+', text.lower()) if len(w) >= 3}


def _keyword_hit(answer: str, keywords: list[str]) -> float:
    """Доля ожидаемых keywords, найденных в тексте ответа (0.0–1.0)."""
    if not keywords or not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def _pairwise_stability(answers: list[str]) -> float:
    """Среднее попарное пересечение keyword-множеств (Jaccard index, 0..1).

    Для каждой пары ответов вычисляет |A ∩ B| / |A ∪ B|,
    затем возвращает среднее по всем парам.
    """
    if len(answers) < 2:
        return 1.0
    word_sets = [_extract_words(a) for a in answers]
    scores = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            a, b = word_sets[i], word_sets[j]
            union = a | b
            scores.append(len(a & b) / len(union) if union else 1.0)
    return sum(scores) / len(scores) if scores else 1.0


def _stars(value: float, n: int = 5) -> str:
    """Преобразовать значение [0..1] в строку из n звёзд (★/☆)."""
    filled = round(max(0.0, min(1.0, value)) * n)
    return "★" * filled + "☆" * (n - filled)


# ---------------------------------------------------------------------------
# Основной класс бенчмарка
# ---------------------------------------------------------------------------


class RAGBenchmark:
    """Сравнивает два RAG-стека на одинаковом наборе вопросов.

    Запускает вопросы последовательно (без async), собирает метрики,
    выводит пять сравнительных таблиц и автоматические выводы.

    Args:
        local_pipeline:  RAGPipeline для локального стека (Ollama).
                         None — стек недоступен, метрики будут нулевые.
        cloud_pipeline:  RAGPipeline для облачного стека (Qwen + Cohere).
                         None — сравнение только по local стеку.
    """

    def __init__(
        self,
        local_pipeline: Optional[RAGPipeline] = None,
        cloud_pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        self.local_pipeline = local_pipeline
        self.cloud_pipeline = cloud_pipeline
        self._cloud_available = cloud_pipeline is not None

    # ---------------------------------------------------------- публичный API

    def run(
        self, quick: bool = False
    ) -> tuple[list[ComparisonResult], list[AntiResult], list[StabilityResult]]:
        """Запустить бенчмарк.

        Args:
            quick: True — прогнать только первые 3 вопроса (без антивопросов
                   и теста стабильности), для быстрой проверки.

        Returns:
            Кортеж (основные результаты, антивопросы, стабильность).
        """
        questions = EVAL_QUESTIONS[:QUICK_MODE_COUNT] if quick else EVAL_QUESTIONS
        results: list[ComparisonResult] = []
        anti_results: list[AntiResult] = []
        stability_results: list[StabilityResult] = []

        total = len(questions)
        mode_label = "quick" if quick else "full"
        print(f"\n  Прогон {total} вопросов × 2 стека ({mode_label} mode)...\n")

        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{total}] «{q['question'][:55]}»")
            result = self._run_comparison(q)
            results.append(result)
            self._print_progress_line("LOCAL", result.local_total_ms, result.local_keyword_hit, result.local_error)
            if self._cloud_available:
                self._print_progress_line("CLOUD", result.cloud_total_ms, result.cloud_keyword_hit, result.cloud_error)

        if not quick:
            print(f"\n  Проверка {len(ANTI_QUESTIONS)} антивопросов...\n")
            for q in ANTI_QUESTIONS:
                print(f"  [{q['id']}] «{q['question'][:55]}»")
                anti = self._run_anti(q)
                anti_results.append(anti)
                l_ref = "✅" if anti.local_is_refusal else "❌"
                c_ref = ("✅" if anti.cloud_is_refusal else "❌") if self._cloud_available else "—"
                print(f"    LOCAL: refusal={l_ref}  CLOUD: refusal={c_ref}")

            print(f"\n  Тест стабильности ({len(STABILITY_QUESTIONS)} вопроса × 3 прогона × 2 стека)...\n")
            for q in STABILITY_QUESTIONS:
                print(f"  [{q['id']}] «{q['question']}»")
                stab = self._run_stability(q)
                stability_results.append(stab)
                l_pct = f"{stab.local_stability:.0%}"
                c_pct = f"{stab.cloud_stability:.0%}" if self._cloud_available and not stab.cloud_error else "—"
                print(f"    LOCAL: {l_pct}  CLOUD: {c_pct}")

        return results, anti_results, stability_results

    def print_all_tables(
        self,
        results: list[ComparisonResult],
        anti_results: list[AntiResult],
        stability_results: list[StabilityResult],
    ) -> None:
        """Вывести все пять сравнительных таблиц."""
        self._print_quality_table(results)
        self._print_speed_table(results)
        self._print_structured_table(results, anti_results)
        if stability_results:
            self._print_stability_table(stability_results)
        self._print_tradeoffs_table(results, stability_results)

    def print_conclusions(
        self,
        results: list[ComparisonResult],
        anti_results: list[AntiResult],
        stability_results: list[StabilityResult],
    ) -> None:
        """Вывести автоматические выводы и рекомендации на основе метрик."""
        valid_local = [r for r in results if not r.local_error]
        valid_cloud = [r for r in results if not r.cloud_error] if self._cloud_available else []

        avg_kw_local = (
            sum(r.local_keyword_hit for r in valid_local) / len(valid_local)
            if valid_local else 0.0
        )
        avg_kw_cloud = (
            sum(r.cloud_keyword_hit for r in valid_cloud) / len(valid_cloud)
            if valid_cloud else 0.0
        )
        avg_total_local = (
            sum(r.local_total_ms for r in valid_local) / len(valid_local)
            if valid_local else 0.0
        )
        avg_total_cloud = (
            sum(r.cloud_total_ms for r in valid_cloud) / len(valid_cloud)
            if valid_cloud else 0.0
        )

        n = len(results)
        anti_n = len(anti_results)
        local_with_quotes = sum(1 for r in results if r.local_has_quotes)
        anti_local_ok = sum(1 for a in anti_results if a.local_is_refusal)
        anti_cloud_ok = sum(1 for a in anti_results if a.cloud_is_refusal) if self._cloud_available else 0

        # Определить bottleneck для local
        bottleneck = "—"
        if valid_local:
            avg_ret = sum(r.local_retrieval_ms for r in valid_local) / len(valid_local)
            avg_rer = sum(r.local_rerank_ms for r in valid_local) / len(valid_local)
            avg_llm = sum(r.local_llm_ms for r in valid_local) / len(valid_local)
            bottleneck = max(
                [("retrieval", avg_ret), ("reranking", avg_rer), ("LLM генерация", avg_llm)],
                key=lambda x: x[1],
            )[0]

        avg_stab_local = (
            sum(s.local_stability for s in stability_results) / len(stability_results)
            if stability_results else None
        )
        avg_stab_cloud = (
            sum(s.cloud_stability for s in stability_results) / len(stability_results)
            if stability_results and self._cloud_available else None
        )

        print("\n" + "=" * 62)
        print("📊 Выводы:")
        print()

        if self._cloud_available and valid_local and valid_cloud:
            delta_pct = (avg_kw_cloud - avg_kw_local) * 100
            direction = "точнее" if delta_pct > 0 else "хуже"
            print(f" - Cloud на {abs(delta_pct):.0f}% {direction} по keyword_hit_rate")
            if avg_total_cloud > 0:
                ratio = avg_total_local / avg_total_cloud
                print(f" - Local в {ratio:.1f}x медленнее (основной bottleneck: {bottleneck})")
        elif valid_local:
            print(f" - Local avg keyword_hit: {avg_kw_local:.2f}")
            print(f" - Local avg total time: {avg_total_local / 1000:.1f}s (bottleneck: {bottleneck})")

        if n > 0:
            print(
                f" - Local хуже следует формату [QUOTES]: "
                f"{n - local_with_quotes} из {n} ответов без цитат"
            )

        if anti_n > 0:
            print(f" - Оба стека: правильных отказов на антивопросы — "
                  f"local {anti_local_ok}/{anti_n}, cloud {anti_cloud_ok}/{anti_n}")

        if avg_stab_local is not None:
            print(f" - Стабильность local:  {avg_stab_local:.0%}")
        if avg_stab_cloud is not None:
            stab_delta = avg_stab_cloud - avg_stab_local
            print(f" - Стабильность cloud:  {avg_stab_cloud:.0%} "
                  f"(разрыв: {abs(stab_delta):.0%})")

        print()
        print(" Рекомендация:")
        print("  - Для production с высокими требованиями → cloud")
        print("  - Для приватных данных / offline / экономии → local")
        print("  - Гибрид: local для retrieval, cloud для генерации ← оптимально")
        print("=" * 62)

    # -------------------------------------------------------- внутренние методы

    @staticmethod
    def _print_progress_line(
        label: str, total_ms: float, kw_hit: float, error: Optional[str]
    ) -> None:
        """Вывести строку прогресса для одного стека."""
        status = "✅" if not error else "❌"
        print(
            f"    {label:<5}: {total_ms / 1000:5.1f}s, "
            f"kw={kw_hit:.2f} {status}"
        )

    def _run_comparison(self, q: dict) -> ComparisonResult:
        """Запустить один вопрос на обоих стеках и собрать ComparisonResult."""
        keywords = q.get("keywords", [])
        local_data = self._run_stack(self.local_pipeline, q["question"])
        cloud_data = (
            self._run_stack(self.cloud_pipeline, q["question"])
            if self._cloud_available else {}
        )

        return ComparisonResult(
            question_id=q["id"],
            question=q["question"],
            # Local
            local_keyword_hit=_keyword_hit(local_data.get("answer", ""), keywords),
            local_has_sources=local_data.get("has_sources", False),
            local_has_quotes=local_data.get("has_quotes", False),
            local_verified_ratio=local_data.get("verified_ratio", 0.0),
            local_is_refusal=local_data.get("is_refusal", False),
            local_retrieval_ms=local_data.get("retrieval_ms", 0.0),
            local_rerank_ms=local_data.get("rerank_ms", 0.0),
            local_llm_ms=local_data.get("llm_ms", 0.0),
            local_total_ms=local_data.get("total_ms", 0.0),
            local_error=local_data.get("error"),
            local_answer=local_data.get("answer", ""),
            local_sources=local_data.get("sources", []),
            # Cloud
            cloud_keyword_hit=_keyword_hit(cloud_data.get("answer", ""), keywords),
            cloud_has_sources=cloud_data.get("has_sources", False),
            cloud_has_quotes=cloud_data.get("has_quotes", False),
            cloud_verified_ratio=cloud_data.get("verified_ratio", 0.0),
            cloud_is_refusal=cloud_data.get("is_refusal", False),
            cloud_retrieval_ms=cloud_data.get("retrieval_ms", 0.0),
            cloud_rerank_ms=cloud_data.get("rerank_ms", 0.0),
            cloud_llm_ms=cloud_data.get("llm_ms", 0.0),
            cloud_total_ms=cloud_data.get("total_ms", 0.0),
            cloud_error=cloud_data.get("error"),
            cloud_answer=cloud_data.get("answer", ""),
            cloud_sources=cloud_data.get("sources", []),
        )

    def _run_anti(self, q: dict) -> AntiResult:
        """Запустить антивопрос на обоих стеках, проверить наличие отказа."""
        local_data = self._run_stack(self.local_pipeline, q["question"])
        cloud_data = (
            self._run_stack(self.cloud_pipeline, q["question"])
            if self._cloud_available else {}
        )
        return AntiResult(
            question_id=q["id"],
            question=q["question"],
            local_is_refusal=local_data.get("is_refusal", False),
            cloud_is_refusal=cloud_data.get("is_refusal", False),
            local_error=local_data.get("error"),
            cloud_error=cloud_data.get("error"),
            local_total_ms=local_data.get("total_ms", 0.0),
            cloud_total_ms=cloud_data.get("total_ms", 0.0),
        )

    def _run_stability(self, q: dict) -> StabilityResult:
        """Прогнать вопрос q["runs"] раз на каждом стеке и вычислить стабильность."""
        runs = q.get("runs", 3)
        local_answers: list[str] = []
        cloud_answers: list[str] = []
        local_err: Optional[str] = None
        cloud_err: Optional[str] = None

        for _ in range(runs):
            d = self._run_stack(self.local_pipeline, q["question"])
            if d.get("error"):
                local_err = d["error"]
                break
            local_answers.append(d.get("answer", ""))

        if self._cloud_available:
            for _ in range(runs):
                d = self._run_stack(self.cloud_pipeline, q["question"])
                if d.get("error"):
                    cloud_err = d["error"]
                    break
                cloud_answers.append(d.get("answer", ""))

        return StabilityResult(
            question_id=q["id"],
            question=q["question"],
            local_stability=_pairwise_stability(local_answers),
            cloud_stability=_pairwise_stability(cloud_answers) if cloud_answers else 0.0,
            local_error=local_err,
            cloud_error=cloud_err,
        )

    @staticmethod
    def _run_stack(pipeline: Optional[RAGPipeline], question: str) -> dict:
        """Выполнить один запрос к пайплайну и вернуть словарь метрик.

        Возвращает пустой словарь если pipeline is None.
        При ошибке возвращает словарь с полем "error".
        """
        if pipeline is None:
            return {}

        try:
            rag_answer = pipeline.answer(question, top_k=5, initial_k=20)
        except Exception as exc:
            return {
                "answer": "", "sources": [],
                "has_sources": False, "has_quotes": False,
                "verified_ratio": 0.0, "is_refusal": False,
                "retrieval_ms": 0.0, "rerank_ms": 0.0, "llm_ms": 0.0,
                "total_ms": 0.0,
                "error": str(exc)[:200],
            }

        struct = rag_answer.structured
        return {
            "answer": rag_answer.answer,
            "sources": [r.source for r in rag_answer.sources],
            "has_sources": struct.has_sources if struct else False,
            "has_quotes": struct.has_quotes if struct else False,
            "verified_ratio": struct.verified_ratio if struct else 0.0,
            "is_refusal": struct.is_refusal if struct else False,
            "retrieval_ms": rag_answer.retrieval_time_ms,
            "rerank_ms": rag_answer.rerank_time_ms,
            "llm_ms": rag_answer.llm_time_ms,
            "total_ms": (
                rag_answer.retrieval_time_ms
                + rag_answer.rerank_time_ms
                + rag_answer.llm_time_ms
            ),
            "error": None,
        }

    # ----------------------------------------------------------- таблицы

    def _print_quality_table(self, results: list[ComparisonResult]) -> None:
        """Таблица 1: Качество (keyword_hit_rate)."""
        Q, L, C, D = 31, 7, 7, 7
        inner = 4 + Q + 2 + L + 2 + C + 2 + D + 1
        hr_top = "╔" + "═" * inner + "╗"
        hr_mid = "╠" + "═" * inner + "╣"
        hr_bot = "╚" + "═" * inner + "╝"

        print(f"\n{hr_top}")
        print(f"║{'  Таблица 1: Качество (keyword_hit_rate)':<{inner}}║")
        print(hr_mid)
        print(f"║  {'#':<3}│ {'Вопрос':<{Q}}│ {'Local':^{L}}│ {'Cloud':^{C}}│ {'Δ':^{D}}║")
        print(hr_mid)

        for r in results:
            q_s = r.question[:Q]
            l_v = f"{r.local_keyword_hit:.2f}" if not r.local_error else "ERR"
            c_v = (
                f"{r.cloud_keyword_hit:.2f}"
                if self._cloud_available and not r.cloud_error
                else ("—" if not self._cloud_available else "ERR")
            )
            if self._cloud_available and not r.local_error and not r.cloud_error:
                d_v = f"{r.local_keyword_hit - r.cloud_keyword_hit:+.2f}"
            else:
                d_v = "—"
            print(f"║  {r.question_id:<3}│ {q_s:<{Q}}│ {l_v:^{L}}│ {c_v:^{C}}│ {d_v:^{D}}║")

        print(hr_mid)
        vl = [r for r in results if not r.local_error]
        vc = [r for r in results if not r.cloud_error] if self._cloud_available else []
        al = sum(r.local_keyword_hit for r in vl) / len(vl) if vl else 0.0
        ac = sum(r.cloud_keyword_hit for r in vc) / len(vc) if vc else 0.0
        al_s = f"{al:.2f}"
        ac_s = f"{ac:.2f}" if self._cloud_available else "—"
        ad_s = f"{al - ac:+.2f}" if self._cloud_available else "—"
        print(f"║  {'AVG':<3}│ {'':<{Q}}│ {al_s:^{L}}│ {ac_s:^{C}}│ {ad_s:^{D}}║")
        print(hr_bot)

    def _print_speed_table(self, results: list[ComparisonResult]) -> None:
        """Таблица 2: Скорость (секунды)."""
        vl = [r for r in results if not r.local_error]
        vc = [r for r in results if not r.cloud_error] if self._cloud_available else []

        def _avg(lst: list, attr: str) -> float:
            return (
                sum(getattr(r, attr) for r in lst) / len(lst) / 1000.0
                if lst else 0.0
            )

        stages = [
            ("Retrieval (avg)",   _avg(vl, "local_retrieval_ms"), _avg(vc, "cloud_retrieval_ms")),
            ("Rerank (avg)",      _avg(vl, "local_rerank_ms"),    _avg(vc, "cloud_rerank_ms")),
            ("LLM gen (avg)",     _avg(vl, "local_llm_ms"),       _avg(vc, "cloud_llm_ms")),
            ("Total (avg)",       _avg(vl, "local_total_ms"),     _avg(vc, "cloud_total_ms")),
        ]

        S, L, C, R = 21, 9, 9, 7
        inner = S + 2 + L + 2 + C + 2 + R + 3
        hr = lambda ch: ch * inner  # noqa: E731

        print(f"\n╔{hr('═')}╗")
        print(f"║{'  Таблица 2: Скорость (секунды)':<{inner}}║")
        print(f"╠{hr('═')}╣")
        print(f"║ {'Этап':<{S}}│ {'Local':^{L}}│ {'Cloud':^{C}}│ {'Ratio':^{R}}║")
        print(f"╠{hr('═')}╣")
        for name, ls, cs in stages:
            l_s = f"{ls:.2f}s"
            c_s = f"{cs:.2f}s" if self._cloud_available else "—"
            r_s = f"{ls / cs:.1f}x" if self._cloud_available and cs > 0 else "—"
            print(f"║ {name:<{S}}│ {l_s:^{L}}│ {c_s:^{C}}│ {r_s:^{R}}║")
        print(f"╚{hr('═')}╝")

    def _print_structured_table(
        self,
        results: list[ComparisonResult],
        anti_results: list[AntiResult],
    ) -> None:
        """Таблица 3: Completeness structured responses."""
        n = len(results)
        an = len(anti_results)
        ca = self._cloud_available

        l_src = sum(1 for r in results if r.local_has_sources)
        c_src = sum(1 for r in results if r.cloud_has_sources) if ca else 0
        l_qts = sum(1 for r in results if r.local_has_quotes)
        c_qts = sum(1 for r in results if r.cloud_has_quotes) if ca else 0
        l_vr  = sum(r.local_verified_ratio for r in results) / n if n else 0.0
        c_vr  = sum(r.cloud_verified_ratio for r in results) / n if ca and n else 0.0
        l_ref = sum(1 for a in anti_results if a.local_is_refusal)
        c_ref = sum(1 for a in anti_results if a.cloud_is_refusal) if ca else 0
        l_fr  = sum(1 for r in results if r.local_is_refusal)
        c_fr  = sum(1 for r in results if r.cloud_is_refusal) if ca else 0

        rows = [
            ("Ответы с sources",      f"{l_src}/{n}",  f"{c_src}/{n}" if ca else "—"),
            ("Ответы с quotes",       f"{l_qts}/{n}",  f"{c_qts}/{n}" if ca else "—"),
            ("Avg verified_ratio",    f"{l_vr:.2f}",   f"{c_vr:.2f}"  if ca else "—"),
            ("Антивопросы refuse",    f"{l_ref}/{an}", f"{c_ref}/{an}" if ca else "—"),
            ("Ложные отказы",         f"{l_fr}/{n}",   f"{c_fr}/{n}"  if ca else "—"),
        ]

        M, L, C = 23, 9, 9
        inner = M + 2 + L + 2 + C + 3
        hr = lambda ch: ch * inner  # noqa: E731

        print(f"\n╔{hr('═')}╗")
        print(f"║{'  Таблица 3: Structured response completeness':<{inner}}║")
        print(f"╠{hr('═')}╣")
        print(f"║ {'Метрика':<{M}}│ {'Local':^{L}}│ {'Cloud':^{C}}║")
        print(f"╠{hr('═')}╣")
        for name, lv, cv in rows:
            print(f"║ {name:<{M}}│ {lv:^{L}}│ {cv:^{C}}║")
        print(f"╚{hr('═')}╝")

    def _print_stability_table(self, stability_results: list[StabilityResult]) -> None:
        """Таблица 4: Стабильность (3 прогона одного вопроса)."""
        Q, L, C = 27, 9, 9
        inner = Q + 2 + L + 2 + C + 3
        hr = lambda ch: ch * inner  # noqa: E731
        ca = self._cloud_available

        print(f"\n╔{hr('═')}╗")
        print(f"║{'  Таблица 4: Стабильность (3 прогона)':<{inner}}║")
        print(f"╠{hr('═')}╣")
        print(f"║ {'Вопрос':<{Q}}│ {'Local':^{L}}│ {'Cloud':^{C}}║")
        print(f"╠{hr('═')}╣")
        for s in stability_results:
            q_s = s.question[:Q]
            l_s = f"{s.local_stability:.0%}" if not s.local_error else "ERR"
            c_s = (
                f"{s.cloud_stability:.0%}"
                if ca and not s.cloud_error
                else "—"
            )
            print(f"║ {q_s:<{Q}}│ {l_s:^{L}}│ {c_s:^{C}}║")
        print(f"╚{hr('═')}╝")
        print("  (% = среднее попарное пересечение Jaccard keywords)")

    def _print_tradeoffs_table(
        self,
        results: list[ComparisonResult],
        stability_results: list[StabilityResult],
    ) -> None:
        """Таблица 5: Trade-offs (качество / скорость / приватность / стоимость)."""
        ca = self._cloud_available
        vl = [r for r in results if not r.local_error]
        vc = [r for r in results if not r.cloud_error] if ca else []

        avg_kw_l = sum(r.local_keyword_hit for r in vl) / len(vl) if vl else 0.0
        avg_kw_c = sum(r.cloud_keyword_hit for r in vc) / len(vc) if vc else 0.0
        # Скорость: инвертируем (меньше = лучше); max ~30s → 0 stars, ~1s → 5 stars
        avg_t_l  = sum(r.local_total_ms for r in vl) / len(vl) if vl else 30_000
        avg_t_c  = sum(r.cloud_total_ms for r in vc) / len(vc) if vc else 30_000
        speed_l  = max(0.0, 1.0 - avg_t_l / 30_000)
        speed_c  = max(0.0, 1.0 - avg_t_c / 30_000)

        avg_stab_l = (
            sum(s.local_stability for s in stability_results) / len(stability_results)
            if stability_results else 0.5
        )
        avg_stab_c = (
            sum(s.cloud_stability for s in stability_results) / len(stability_results)
            if stability_results and ca else 0.5
        )

        rows = [
            ("Качество",         _stars(avg_kw_l),    _stars(avg_kw_c) if ca else "—"),
            ("Скорость",         _stars(speed_l),     _stars(speed_c)  if ca else "—"),
            ("Стабильность",     _stars(avg_stab_l),  _stars(avg_stab_c) if ca else "—"),
            ("Стоимость/запрос", "0 ₽",               "~0.3-1 ₽"      if ca else "—"),
            ("Приватность",      _stars(1.0),         _stars(0.4)      if ca else "—"),
            ("Offline",          "✅ да",              "❌ нет"         if ca else "—"),
            ("Structured resp.", "частично",           "стабильно"     if ca else "—"),
        ]

        M, L, C = 21, 15, 15
        inner = M + 2 + L + 2 + C + 3
        hr = lambda ch: ch * inner  # noqa: E731

        print(f"\n╔{hr('═')}╗")
        print(f"║{'  Таблица 5: Trade-offs':<{inner}}║")
        print(f"╠{hr('═')}╣")
        print(f"║ {'':<{M}}│ {'Local':^{L}}│ {'Cloud':^{C}}║")
        print(f"╠{hr('═')}╣")
        for name, lv, cv in rows:
            print(f"║ {name:<{M}}│ {lv:^{L}}│ {cv:^{C}}║")
        print(f"╚{hr('═')}╝")
