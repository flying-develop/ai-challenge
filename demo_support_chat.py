"""Демо-скрипт: Support Chat (День 25).

Прогоняет два тестовых сценария через DialogManager без интерактивного ввода:
    Сценарий 1 — «Свободный разговор» (12 сообщений, прыжки между темами)
    Сценарий 2 — «Пошаговая помощь» (11 сообщений, накопление task state)

После каждого сообщения выводит:
    - вопрос и краткий ответ
    - источники
    - текущий task_state

Итоговый отчёт — таблица метрик по двум сценариям.

Запуск:
    python demo_support_chat.py
    python demo_support_chat.py --quick  # только первые 5 сообщений каждого сценария
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_indexer.src.embedding.provider import EmbeddingProvider
from rag_indexer.src.storage.index_store import IndexStore
from rag_indexer.src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever
from rag_indexer.src.retrieval.pipeline import RAGPipeline
from rag_indexer.src.retrieval.reranker import ThresholdFilter
from llm_agent.memory.manager import MemoryManager

from src.chat.dialog_manager import DialogManager
from src.chat.scenarios import SCENARIO_FREE, SCENARIO_STEPWISE
from src.llm_helper import make_llm_fn

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

_DB_PATH = Path("./output/index.db")
_MEMORY_DIR = Path("./output/demo_memory")


# ---------------------------------------------------------------------------
# Dataclass для сбора метрик
# ---------------------------------------------------------------------------

@dataclass
class MessageResult:
    """Результат обработки одного сообщения."""
    step: int
    question: str
    answer_preview: str
    has_sources: bool
    has_quotes: bool
    task_state: dict
    confidence: float
    enriched_query: str = ""


@dataclass
class ScenarioResult:
    """Результаты одного сценария."""
    name: str
    messages: list[MessageResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.messages)

    @property
    def with_sources(self) -> int:
        return sum(1 for m in self.messages if m.has_sources)

    @property
    def with_quotes(self) -> int:
        return sum(1 for m in self.messages if m.has_quotes)

    @property
    def task_state_updates(self) -> int:
        """Количество изменений task state."""
        prev = {}
        updates = 0
        for m in self.messages:
            if m.task_state != prev:
                updates += 1
            prev = m.task_state.copy()
        return updates

    @property
    def avg_confidence(self) -> float:
        confidences = [m.confidence for m in self.messages if m.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0

    @property
    def false_refusals(self) -> int:
        """Количество ответов 'не знаю' при наличии источников."""
        count = 0
        for m in self.messages:
            if m.has_sources and _is_refusal(m.answer_preview):
                count += 1
        return count


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _is_refusal(text: str) -> bool:
    """Проверить — это отказ ('не знаю') или реальный ответ."""
    refusal_markers = ["не знаю", "не могу найти", "нет информации",
                       "не нашел", "не нашёл", "отсутствует"]
    text_lower = text.lower()
    return any(m in text_lower for m in refusal_markers)


def _build_rag_pipeline(llm_fn, db_path: Path) -> RAGPipeline:
    """Построить RAG-пайплайн."""
    embedding_provider = EmbeddingProvider.create("qwen")
    store = IndexStore(db_path)

    # Проверяем что индекс не пустой
    stats = store.get_stats()
    if stats["chunks"] == 0:
        print(f"[WARN] Индекс пуст: {db_path}")
        print("  Запустите: python -m rag_indexer.main ./podkop-wiki/content")
        print("  Или: python bot.py → /index path ./podkop-wiki/content")

    vector_retriever = VectorRetriever(store=store, embedder=embedding_provider)
    bm25_retriever = BM25Retriever(store=store)
    retriever = HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)
    reranker = ThresholdFilter(threshold=0.0)

    return RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        reranker=reranker,
        use_structured=True,
    )


def _create_dialog_manager(user_id: str, rag_pipeline, llm_fn) -> DialogManager:
    """Создать DialogManager с изолированной памятью."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_path = _MEMORY_DIR / f"user_{user_id}.db"
    memory = MemoryManager(str(memory_path))
    return DialogManager(
        user_id=user_id,
        memory_manager=memory,
        rag_pipeline=rag_pipeline,
        llm_fn=llm_fn,
    )


def _process_step(
    dialog: DialogManager,
    question: str,
    step: int,
    verbose: bool = True,
) -> MessageResult:
    """Обработать одно сообщение и вернуть результат.

    Args:
        dialog:   DialogManager пользователя.
        question: Вопрос пользователя.
        step:     Номер шага (для вывода).
        verbose:  Выводить детали в консоль.

    Returns:
        MessageResult с метриками.
    """
    if verbose:
        print(f"\n  [{step}] Q: {question}")

    rag_answer = dialog.process_message(question)
    task_state = dialog.get_task_state()

    # Извлекаем метрики
    structured = getattr(rag_answer, "structured", None)
    has_sources = bool(
        (structured and structured.sources) or
        (not structured and rag_answer.sources)
    )
    has_quotes = bool(
        structured and getattr(structured, "verified_quotes", None)
    )
    confidence = getattr(rag_answer, "confidence", 0.0)

    answer_text = (structured.answer if structured else rag_answer.answer) or ""
    answer_preview = answer_text[:120].replace("\n", " ")

    if verbose:
        print(f"     A: {answer_preview}...")
        if has_sources:
            sources = structured.sources if structured else rag_answer.sources
            for i, src in enumerate(sources[:2], 1):
                print(f"     [{i}] {src.source} — {src.section[:50]}")
        if task_state:
            goal = task_state.get("goal", "")
            constraints = task_state.get("constraints", "")
            stage = task_state.get("stage", "")
            state_parts = []
            if goal:
                state_parts.append(f"goal={goal[:40]!r}")
            if constraints:
                state_parts.append(f"constraints={constraints[:40]!r}")
            if stage:
                state_parts.append(f"stage={stage!r}")
            if state_parts:
                print(f"     task_state: {', '.join(state_parts)}")

    return MessageResult(
        step=step,
        question=question,
        answer_preview=answer_preview,
        has_sources=has_sources,
        has_quotes=has_quotes,
        task_state=task_state.copy(),
        confidence=confidence,
    )


def run_scenario(
    scenario_messages: list[str],
    scenario_name: str,
    rag_pipeline,
    llm_fn,
    max_messages: int | None = None,
) -> ScenarioResult:
    """Прогнать один сценарий.

    Args:
        scenario_messages: Список сообщений сценария.
        scenario_name:     Имя сценария для вывода.
        rag_pipeline:      Общий RAG пайплайн.
        llm_fn:            LLM функция.
        max_messages:      Ограничить количество сообщений (для --quick режима).

    Returns:
        ScenarioResult с метриками всех шагов.
    """
    result = ScenarioResult(name=scenario_name)
    messages = scenario_messages[:max_messages] if max_messages else scenario_messages

    # Создаём изолированный DialogManager для этого сценария
    user_id = f"demo_{scenario_name.lower().replace(' ', '_')}"
    dialog = _create_dialog_manager(user_id, rag_pipeline, llm_fn)

    print(f"\n{'=' * 60}")
    print(f"  {scenario_name}")
    print(f"  {len(messages)} сообщений")
    print(f"{'=' * 60}")

    for i, message in enumerate(messages, 1):
        msg_result = _process_step(dialog, message, step=i, verbose=True)
        result.messages.append(msg_result)
        time.sleep(0.5)  # небольшая пауза между запросами

    return result


# ---------------------------------------------------------------------------
# Шаг 4: Проверка памяти
# ---------------------------------------------------------------------------

def check_memory_retention(stepwise_result: ScenarioResult) -> dict:
    """Проверить что task state накапливался в Сценарии 2.

    Args:
        stepwise_result: Результаты Сценария 2.

    Returns:
        dict с результатами проверок.
    """
    checks = {}

    # Проверка 1: TP-Link упоминается с шага 2, должен быть в task state на шаге 8
    if len(stepwise_result.messages) >= 8:
        step8_state = stepwise_result.messages[7].task_state
        constraints = step8_state.get("constraints", "").lower()
        tp_link_remembered = "tp-link" in constraints or "tp link" in constraints
        checks["tp_link_at_step8"] = {
            "passed": tp_link_remembered,
            "detail": f"constraints на шаге 8: {constraints[:80]}",
        }

    # Проверка 2: task state накапливается (количество обновлений)
    updates = stepwise_result.task_state_updates
    checks["task_state_updates"] = {
        "passed": updates >= 3,
        "detail": f"Обновлений task state: {updates}",
    }

    # Проверка 3: к шагу 4+ goal должен содержать "vless"
    if len(stepwise_result.messages) >= 4:
        step4_state = stepwise_result.messages[3].task_state
        goal = step4_state.get("goal", "").lower()
        vless_in_goal = "vless" in goal or "протокол" in goal
        checks["vless_goal_at_step4"] = {
            "passed": vless_in_goal,
            "detail": f"goal на шаге 4: {goal[:80]}",
        }

    return checks


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------

def print_report(
    free_result: ScenarioResult,
    stepwise_result: ScenarioResult,
    memory_checks: dict,
) -> None:
    """Вывести итоговый отчёт в виде ASCII-таблицы.

    Args:
        free_result:      Результаты Сценария 1.
        stepwise_result:  Результаты Сценария 2.
        memory_checks:    Результаты проверок памяти.
    """
    def pct(n, total):
        return f"{n} ({int(100*n/total)}%)" if total > 0 else f"{n} (0%)"

    w1, w2, w3 = 22, 14, 16

    def row(label, v1, v2):
        return f"║ {label:<{w1}} ║ {v1:^{w2}} ║ {v2:^{w3}} ║"

    def sep(left="╠", mid="╬", right="╣"):
        return left + "═" * (w1 + 2) + mid + "═" * (w2 + 2) + mid + "═" * (w3 + 2) + right

    s1 = free_result
    s2 = stepwise_result

    print(f"\n{'╔' + '═'*(w1+w2+w3+8) + '╗'}")
    title = "Support Chat Test Report"
    print(f"║{title:^{w1+w2+w3+8}}║")
    print(sep("╠", "╦", "╣"))
    print(f"║ {'Метрика':<{w1}} ║ {'Сценарий 1':^{w2}} ║ {'Сценарий 2':^{w3}} ║")
    print(sep())
    print(row("Сообщений", s1.total, s2.total))
    print(row("С источниками", pct(s1.with_sources, s1.total), pct(s2.with_sources, s2.total)))
    print(row("С цитатами", pct(s1.with_quotes, s1.total), pct(s2.with_quotes, s2.total)))
    print(row("Task state updates", s1.task_state_updates, s2.task_state_updates))
    print(row("Ложных отказов", s1.false_refusals, s2.false_refusals))
    print(row("Avg confidence", f"{s1.avg_confidence:.2f}", f"{s2.avg_confidence:.2f}"))
    print(sep("╚", "╩", "╝"))

    # Проверки памяти
    print(f"\n  Проверки накопления контекста:")
    for check_name, check in memory_checks.items():
        status = "✅" if check["passed"] else "❌"
        print(f"  {status} {check_name}: {check['detail']}")

    # Task state Сценария 2 по шагам
    print(f"\n  Task State — Сценарий 2 (накопление):")
    for msg in stepwise_result.messages:
        state = msg.task_state
        if state:
            goal = state.get("goal", "")[:45]
            constraints = state.get("constraints", "")[:45]
            print(f"  [{msg.step:2d}] goal={goal!r}")
            if constraints:
                print(f"       constraints={constraints!r}")


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    """Запустить демо-скрипт."""
    parser = argparse.ArgumentParser(description="Support Chat Demo — День 25")
    parser.add_argument("--quick", action="store_true",
                        help="Прогнать только первые 5 сообщений каждого сценария")
    args = parser.parse_args()

    max_messages = 5 if args.quick else None

    print("=" * 60)
    print("  Support Chat Demo — День 25")
    print("=" * 60)

    # Шаг 1: Инициализация
    print("\n[Шаг 1] Инициализация...")
    llm_fn = make_llm_fn(timeout=60.0)
    rag_pipeline = _build_rag_pipeline(llm_fn, _DB_PATH)

    # Показываем статистику индекса
    if _DB_PATH.exists():
        with IndexStore(_DB_PATH) as store:
            stats = store.get_stats()
            strategies = store.get_all_strategies()
        print(f"  Индекс: {stats['chunks']} чанков, стратегии: {strategies}")
        print(f"  RAG режим: hybrid + structured responses")
    else:
        print(f"  [WARN] Индекс не найден: {_DB_PATH}")
        print(f"  Некоторые ответы могут быть пустыми.")

    # Шаг 2: Сценарий 1
    print("\n[Шаг 2] Сценарий 1: Свободный разговор")
    free_result = run_scenario(
        SCENARIO_FREE, "Сценарий 1: Свободный разговор",
        rag_pipeline, llm_fn, max_messages=max_messages,
    )

    # Шаг 3: Сценарий 2
    print("\n[Шаг 3] Сценарий 2: Пошаговая помощь")
    stepwise_result = run_scenario(
        SCENARIO_STEPWISE, "Сценарий 2: Пошаговая помощь",
        rag_pipeline, llm_fn, max_messages=max_messages,
    )

    # Шаг 4: Проверка памяти
    print("\n[Шаг 4] Проверка накопления контекста...")
    memory_checks = check_memory_retention(stepwise_result)

    # Шаг 5: Итоговый отчёт
    print("\n[Шаг 5] Итоговый отчёт:")
    print_report(free_result, stepwise_result, memory_checks)

    print("\n✅ Демо завершено")


if __name__ == "__main__":
    main()
