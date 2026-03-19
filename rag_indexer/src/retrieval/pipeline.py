"""RAG Pipeline с реранкингом и query rewriting.

Цепочка:
    Вопрос → [Query Rewrite] → Поиск (initial_k кандидатов)
           → [Реранкинг]     → top_k чанков
           → Промпт → LLM → Ответ

Ключевое: initial_k (20) > top_k (5) — реранкер выбирает лучших
из широкого набора кандидатов, что улучшает качество.

Режим структурированного ответа (use_structured=True):
    Добавляет к цепочке: ConfidenceScorer → StructuredRAGPrompt → ResponseParser
    LLM возвращает ответ в формате [ANSWER]/[SOURCES]/[QUOTES].

Использование:
    from src.retrieval.pipeline import RAGPipeline, PipelineEvaluator

    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        llm_fn=my_llm_fn,
        reranker=CohereReranker(),
        query_rewriter=QueryRewriter(llm_fn=my_llm_fn),
        use_structured=True,
    )
    rag_answer = pipeline.answer("Как установить podkop?", top_k=5, initial_k=20)
    print(rag_answer.answer)
    if rag_answer.structured:
        print(rag_answer.structured.verified_ratio)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .retriever import RetrievalStrategy, RetrievalResult
from .rag_query import RAGQueryBuilder
from .reranker import Reranker
from .query_rewrite import QueryRewriter
from .evaluator import EVAL_QUESTIONS
from .confidence import ConfidenceScorer
from .structured_prompt import StructuredRAGPrompt
from .response_parser import ResponseParser, StructuredResponse


@dataclass
class RAGAnswer:
    """Полный результат ответа RAG-пайплайна с метаданными."""

    question: str
    answer: str
    mode: str                       # имя конфигурации: "hybrid+rewrite+cohere"
    sources: list[RetrievalResult]  # чанки, попавшие в контекст LLM
    rewrite_variants: list[str]     # варианты запроса (если rewrite включён)
    initial_results_count: int      # сколько нашёл поиск до реранкинга
    final_results_count: int        # сколько осталось после реранкинга
    retrieval_time_ms: float
    rerank_time_ms: float
    llm_time_ms: float
    structured: Optional[StructuredResponse] = None  # заполнен при use_structured=True
    confidence: float = 0.0         # средний score контекста


class RAGPipeline:
    """Полный RAG-пайплайн: rewrite → retrieve → rerank → build prompt → LLM.

    Каждый компонент опционален:
        - без rewriter: поиск идёт по исходному вопросу
        - без reranker: берём top_k из результатов поиска напрямую

    При use_structured=True добавляет:
        ConfidenceScorer → StructuredRAGPrompt → ResponseParser → верификация цитат
    """

    def __init__(
        self,
        retriever: RetrievalStrategy,
        llm_fn: Callable[[str, str], str],
        reranker: Optional[Reranker] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        query_builder: Optional[RAGQueryBuilder] = None,
        use_structured: bool = False,
    ):
        """
        Args:
            retriever:       Стратегия поиска (vector/bm25/hybrid).
            llm_fn:          Функция (system_prompt, user_prompt) -> ответ строкой.
            reranker:        Опциональный реранкер (Cohere, Ollama, ThresholdFilter).
            query_rewriter:  Опциональный переформулировщик запросов.
            query_builder:   Builder промптов (по умолчанию RAGQueryBuilder()).
            use_structured:  True — использовать structured prompt + parser + верификацию.
        """
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.query_builder = query_builder or RAGQueryBuilder()
        self.use_structured = use_structured
        self._confidence_scorer = ConfidenceScorer()
        self._structured_prompt = StructuredRAGPrompt()
        self._response_parser = ResponseParser()

    @property
    def name(self) -> str:
        """Имя конфигурации для логирования: "hybrid", "hybrid+cohere", etc."""
        parts = [self.retriever.name]
        if self.query_rewriter:
            parts.append("rewrite")
        if self.reranker:
            parts.append(self.reranker.name)
        return "+".join(parts)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        initial_k: int = 20,
        max_tokens: int | None = None,
    ) -> RAGAnswer:
        """Выполнить полный пайплайн и вернуть ответ.

        Args:
            question:  Вопрос пользователя.
            top_k:     Сколько чанков подать в LLM.
            initial_k: Сколько кандидатов получить из поиска до реранкинга.
                       Должен быть > top_k, чтобы реранкеру было из чего выбирать.
        """
        rewrite_variants: list[str] = []

        # 1. Query rewrite: генерация вариантов запроса
        if self.query_rewriter:
            rewrite_variants = self.query_rewriter.rewrite(question)
            queries = rewrite_variants
        else:
            queries = [question]

        # 2. Поиск по всем вариантам с дедупликацией по chunk_id
        t0 = time.monotonic()
        fetch_k = initial_k if self.reranker else top_k
        best_by_id: dict[str, RetrievalResult] = {}
        for q in queries:
            for r in self.retriever.search(q, top_k=fetch_k):
                # Оставляем чанк с лучшим score среди всех вариантов запроса
                if r.chunk_id not in best_by_id or r.score > best_by_id[r.chunk_id].score:
                    best_by_id[r.chunk_id] = r

        candidates = sorted(best_by_id.values(), key=lambda r: r.score, reverse=True)
        retrieval_time_ms = (time.monotonic() - t0) * 1000
        initial_count = len(candidates)

        # 3. Реранкинг
        t1 = time.monotonic()
        if self.reranker and candidates:
            results = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]
        rerank_time_ms = (time.monotonic() - t1) * 1000

        # 4. Формирование промпта и вызов LLM
        t2 = time.monotonic()
        structured_response: Optional[StructuredResponse] = None
        confidence = self._confidence_scorer.score(results)

        if self.use_structured:
            # Структурированный режим: [ANSWER]/[SOURCES]/[QUOTES]
            prompt = self._structured_prompt.build(question, results, confidence)
            raw_answer = self.llm_fn(prompt.system, prompt.user)
            structured_response = self._response_parser.parse(
                raw_answer, results, confidence
            )
            answer_text = structured_response.answer
        else:
            # Обычный режим (совместимость)
            build_kwargs = {} if max_tokens is None else {"max_tokens": max_tokens}
            ctx = self.query_builder.build(question, results, **build_kwargs)
            answer_text = self.llm_fn(ctx.system_prompt, ctx.user_prompt)

        llm_time_ms = (time.monotonic() - t2) * 1000

        return RAGAnswer(
            question=question,
            answer=answer_text,
            mode=self.name,
            sources=results,
            rewrite_variants=rewrite_variants,
            initial_results_count=initial_count,
            final_results_count=len(results),
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            llm_time_ms=llm_time_ms,
            structured=structured_response,
            confidence=confidence,
        )


@dataclass
class PipelineEvalResult:
    """Результат оценки одного вопроса по одному пайплайну."""

    question_id: int
    question: str
    pipeline_name: str
    source_hit: bool
    keyword_hits: int
    keyword_total: int
    keyword_rate: float
    answer_preview: str
    retrieval_time_ms: float
    rerank_time_ms: float
    llm_time_ms: float
    initial_count: int
    final_count: int


class PipelineEvaluator:
    """Оценщик RAG-пайплайнов на контрольных вопросах.

    Запускает каждый пайплайн на всех вопросах и собирает метрики:
        - source_hit: найден ли ожидаемый файл-источник
        - keyword_rate: доля ключевых слов в ответе
        - времена: retrieval, rerank, llm
    """

    def __init__(
        self,
        pipelines: list[RAGPipeline],
        top_k: int = 5,
        initial_k: int = 20,
        eval_max_tokens: int = 1000,
    ):
        """
        Args:
            pipelines:        Список пайплайнов для сравнения.
            top_k:            Количество чанков в контексте LLM.
            initial_k:        Количество кандидатов для реранкинга.
            eval_max_tokens:  Лимит токенов контекста при eval (меньше = быстрее).
        """
        self.pipelines = pipelines
        self.top_k = top_k
        self.initial_k = initial_k
        self.eval_max_tokens = eval_max_tokens

    def run(self) -> list[PipelineEvalResult]:
        """Прогнать все пайплайны на контрольных вопросах."""
        results = []
        for q in EVAL_QUESTIONS:
            for pipeline in self.pipelines:
                rag_answer = pipeline.answer(
                    q["question"],
                    top_k=self.top_k,
                    initial_k=self.initial_k,
                    max_tokens=self.eval_max_tokens,
                )
                sources = [r.source for r in rag_answer.sources]
                source_hit = any(q["expected_source"] in s for s in sources)

                combined_text = " ".join(r.text.lower() for r in rag_answer.sources)
                keyword_hits = sum(
                    1 for kw in q["keywords"] if kw.lower() in combined_text
                )
                keyword_rate = keyword_hits / len(q["keywords"]) if q["keywords"] else 0.0

                results.append(
                    PipelineEvalResult(
                        question_id=q["id"],
                        question=q["question"],
                        pipeline_name=pipeline.name,
                        source_hit=source_hit,
                        keyword_hits=keyword_hits,
                        keyword_total=len(q["keywords"]),
                        keyword_rate=keyword_rate,
                        answer_preview=rag_answer.answer[:300],
                        retrieval_time_ms=rag_answer.retrieval_time_ms,
                        rerank_time_ms=rag_answer.rerank_time_ms,
                        llm_time_ms=rag_answer.llm_time_ms,
                        initial_count=rag_answer.initial_results_count,
                        final_count=rag_answer.final_results_count,
                    )
                )
        return results

    def print_comparison_table(self, results: list[PipelineEvalResult]) -> None:
        """Вывести сравнительную таблицу keyword_rate по всем пайплайнам."""
        pipeline_names = list(dict.fromkeys(r.pipeline_name for r in results))
        question_ids = list(dict.fromkeys(r.question_id for r in results))

        # Таблица (q_id, pipeline_name) -> keyword_rate
        lookup: dict[tuple, float] = {}
        for r in results:
            lookup[(r.question_id, r.pipeline_name)] = r.keyword_rate

        q_col = 26
        p_col = 11

        # Ширина строки
        total_width = 4 + q_col + 2 + len(pipeline_names) * (p_col + 3)
        inner = total_width - 2

        q_texts = {q["id"]: q["question"] for q in EVAL_QUESTIONS}

        print("\n╔" + "═" * inner + "╗")
        title = "  СРАВНИТЕЛЬНАЯ ТАБЛИЦА RAG ПАЙПЛАЙНОВ (keyword_rate)"
        print(f"║{title:<{inner}}║")
        print("╠" + "═" * inner + "╣")

        # Заголовок
        header = f"║ {'#':<3} │ {'Вопрос':<{q_col}} "
        for name in pipeline_names:
            short = name[: p_col - 1]
            header += f"│ {short:^{p_col - 1}} "
        header += "║"
        print(header)
        print("╠" + "═" * inner + "╣")

        # Строки вопросов
        for qid in question_ids:
            q_short = q_texts.get(qid, "")[:q_col - 1]
            row = f"║ {qid:<3} │ {q_short:<{q_col}} "
            for name in pipeline_names:
                rate = lookup.get((qid, name), 0.0)
                cell = f"{rate:.2f}"
                row += f"│ {cell:^{p_col - 1}} "
            row += "║"
            print(row)

        print("╠" + "═" * inner + "╣")

        # Средние значения
        avg_row = f"║ {'AVG':<3} │ {'':<{q_col}} "
        for name in pipeline_names:
            rates = [lookup.get((qid, name), 0.0) for qid in question_ids]
            avg = sum(rates) / len(rates) if rates else 0.0
            cell = f"{avg:.2f}"
            avg_row += f"│ {cell:^{p_col - 1}} "
        avg_row += "║"
        print(avg_row)
        print("╚" + "═" * inner + "╝")

    def print_timing_summary(self, results: list[PipelineEvalResult]) -> None:
        """Вывести сводку по времени выполнения каждого пайплайна."""
        pipeline_names = list(dict.fromkeys(r.pipeline_name for r in results))
        print("\n  Среднее время (мс):")
        print(f"  {'Пайплайн':<35} {'поиск':>8}  {'реранк':>8}  {'llm':>8}  {'всего':>8}")
        print("  " + "─" * 73)
        for name in pipeline_names:
            p_results = [r for r in results if r.pipeline_name == name]
            avg_ret = sum(r.retrieval_time_ms for r in p_results) / len(p_results)
            avg_rer = sum(r.rerank_time_ms for r in p_results) / len(p_results)
            avg_llm = sum(r.llm_time_ms for r in p_results) / len(p_results)
            total = avg_ret + avg_rer + avg_llm
            print(
                f"  {name:<35} {avg_ret:>7.0f}ms {avg_rer:>7.0f}ms "
                f"{avg_llm:>7.0f}ms {total:>7.0f}ms"
            )
