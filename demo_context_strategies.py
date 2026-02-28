"""Демонстрация и сравнение 3 стратегий управления контекстом.

Скрипт прогоняет один и тот же сценарий (сбор ТЗ, ~12 сообщений) на каждой
из трёх стратегий:
    1. Sliding Window  — хранит только последние N сообщений
    2. Sticky Facts     — facts (ключ-значение) + последние N сообщений
    3. Branching        — checkpoint + 2 ветки диалога

После каждого прогона задаются контрольные вопросы для проверки памяти.
В конце выводится сравнительная таблица:
    - качество ответов
    - стабильность (не теряет ли важные детали)
    - расход токенов
    - удобство для пользователя

Запуск:
    export QWEN_API_KEY=sk-...
    python demo_context_strategies.py

Модель: Qwen (через QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL).
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.context_strategies import (
    BranchingStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)
from llm_agent.application.strategy_agent import StrategyAgent
from llm_agent.config import get_config
from llm_agent.domain.models import TokenUsage
from llm_agent.infrastructure.qwen_client import QwenHttpClient
from llm_agent.infrastructure.token_counter import TiktokenCounter

# ===========================================================================
# ПАРАМЕТРЫ
# ===========================================================================

WINDOW_SIZE = 10            # размер окна для Sliding Window
FACTS_WINDOW_SIZE = 6       # размер окна для Facts (меньше, т.к. есть facts)
WIDTH = 78                  # ширина вывода

SYSTEM_PROMPT = (
    "Ты опытный бизнес-аналитик и системный архитектор. "
    "Помогаешь составлять техническое задание (ТЗ) для разработки. "
    "Давай краткие, структурированные ответы на русском языке. "
    "Помни всё, что было сказано в нашем разговоре."
)

# ===========================================================================
# Сценарий: сбор ТЗ для мобильного приложения (12 сообщений)
# ===========================================================================

SCENARIO_MESSAGES = [
    "Привет! Мне нужно составить ТЗ для мобильного приложения. Это будет трекер привычек.",
    "Целевая аудитория — молодые люди 18-35 лет, которые хотят выработать полезные привычки.",
    "Приложение должно работать на iOS и Android. Предпочитаем Flutter для кросс-платформы.",
    "Основные функции: создание привычек, отслеживание выполнения, статистика и напоминания.",
    "Для авторизации используем email + пароль и вход через Google/Apple ID.",
    "Данные хранить локально с синхронизацией в облако. Бэкенд — Firebase.",
    "Дизайн должен быть минималистичным, в тёмных тонах. Основной цвет — фиолетовый (#7C3AED).",
    "Нужна геймификация: очки за выполнение, серии (streaks), уровни и достижения.",
    "Бюджет проекта — 2 миллиона рублей. Срок разработки — 4 месяца.",
    "Команда: 2 Flutter-разработчика, 1 дизайнер, 1 тестировщик, 1 бэкенд-разработчик.",
    "MVP должен включать: создание привычек, напоминания, базовую статистику. Геймификация — во второй версии.",
    "Есть ли у тебя вопросы по ТЗ? Что бы ты добавил или уточнил?",
]

# Контрольные вопросы — проверяем, помнит ли агент ключевые детали
RECALL_QUESTIONS = [
    "Какой бюджет и сроки проекта мы обсуждали?",
    "Какой стек технологий мы выбрали (фреймворк, бэкенд)?",
    "Опиши целевую аудиторию приложения.",
    "Что должно войти в MVP, а что отложено на вторую версию?",
    "Составь краткое резюме всего ТЗ, которое мы обсудили.",
]

# ===========================================================================
# Дополнительные вопросы для веток (только для Branching)
# ===========================================================================

BRANCH_A_QUESTIONS = [
    "Давай обсудим вариант с подпиской. Какие тарифные планы предложишь?",
    "Какие функции будут платными, а какие бесплатными?",
]

BRANCH_B_QUESTIONS = [
    "Давай обсудим вариант с рекламой вместо подписки. Какие форматы рекламы подойдут?",
    "Как монетизация через рекламу повлияет на UX приложения?",
]


# ===========================================================================
# Структуры данных
# ===========================================================================

@dataclass
class TurnStats:
    turn: int
    question: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class StrategyRunResult:
    strategy_name: str
    turns: list[TurnStats] = field(default_factory=list)
    recall_turns: list[TurnStats] = field(default_factory=list)
    extra_info: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def total_dialog_tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    @property
    def total_recall_tokens(self) -> int:
        return sum(t.total_tokens for t in self.recall_turns)

    @property
    def grand_total_tokens(self) -> int:
        overhead = self.extra_info.get("overhead_tokens", 0)
        return self.total_dialog_tokens + self.total_recall_tokens + overhead

    @property
    def avg_prompt_tokens(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.prompt_tokens for t in self.turns) / len(self.turns)

    @property
    def last_prompt_tokens(self) -> int:
        return self.turns[-1].prompt_tokens if self.turns else 0


# ===========================================================================
# Вспомогательный вывод
# ===========================================================================

def sep(char: str = "=", width: int = WIDTH) -> None:
    print(char * width)

def header(title: str) -> None:
    print()
    sep()
    print(f"  {title}")
    sep()

def show_q(text: str) -> None:
    prefix = "  Q: "
    indent = " " * len(prefix)
    lines = textwrap.wrap(text, width=WIDTH - len(prefix))
    for i, line in enumerate(lines):
        print(f"{prefix if i == 0 else indent}{line}")

def show_a(text: str, max_chars: int = 250) -> None:
    short = text[:max_chars] + ("..." if len(text) > max_chars else "")
    prefix = "  A: "
    indent = " " * len(prefix)
    lines = textwrap.wrap(short, width=WIDTH - len(prefix))
    for i, line in enumerate(lines):
        print(f"{prefix if i == 0 else indent}{line}")

def show_tokens(prompt_t: int, completion_t: int, label: str = "") -> None:
    total = prompt_t + completion_t
    tag = f"  {label}" if label else ""
    print(f"  [tokens: prompt={prompt_t:>5}, response={completion_t:>4}, total={total:>5}]{tag}")

def make_turn(turn: int, question: str, answer: str, usage: TokenUsage | None) -> TurnStats:
    if usage:
        return TurnStats(
            turn=turn, question=question, answer=answer,
            prompt_tokens=usage.history_tokens,
            completion_tokens=usage.response_tokens,
            total_tokens=usage.total_tokens,
        )
    return TurnStats(turn=turn, question=question, answer=answer,
                     prompt_tokens=0, completion_tokens=0, total_tokens=0)


# ===========================================================================
# Прогон стратегии
# ===========================================================================

def run_strategy(
    strategy_name: str,
    agent: StrategyAgent,
    questions: list[str],
    recall_questions: list[str],
) -> StrategyRunResult:
    """Прогнать сценарий на одной стратегии."""
    result = StrategyRunResult(strategy_name=strategy_name)

    print(f"\n  Диалог ({len(questions)} сообщений):\n")
    for i, q in enumerate(questions, 1):
        print(f"  --- Ход {i}/{len(questions)} ---")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")
            result.error = str(exc)
            break
        show_a(answer)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
        )
        result.turns.append(make_turn(i, q, answer, usage))
        # Небольшая пауза для вежливости к API
        time.sleep(0.5)

    # Контрольные вопросы
    print(f"\n  --- Контрольные вопросы (проверка памяти) ---\n")
    for i, q in enumerate(recall_questions, 1):
        print(f"  [Recall {i}]")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")
            answer = f"[ERROR: {exc}]"
        show_a(answer, max_chars=400)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
            label="(recall)",
        )
        result.recall_turns.append(
            make_turn(len(questions) + i, q, answer, usage)
        )
        time.sleep(0.5)

    return result


# ===========================================================================
# Прогон 1: Sliding Window
# ===========================================================================

def run_sliding_window(client: QwenHttpClient) -> StrategyRunResult:
    header(f"СТРАТЕГИЯ 1: Sliding Window (окно = {WINDOW_SIZE} сообщений)")
    print(f"  Хранит только последние {WINDOW_SIZE} сообщений, остальное отбрасывает.")

    strategy = SlidingWindowStrategy(window_size=WINDOW_SIZE)
    agent = StrategyAgent(
        llm_client=client,
        strategy=strategy,
        system_prompt=SYSTEM_PROMPT,
        token_counter=TiktokenCounter(),
    )

    result = run_strategy(
        strategy_name=strategy.name,
        agent=agent,
        questions=SCENARIO_MESSAGES,
        recall_questions=RECALL_QUESTIONS,
    )
    result.extra_info = strategy.get_stats()
    return result


# ===========================================================================
# Прогон 2: Sticky Facts
# ===========================================================================

def run_sticky_facts(client: QwenHttpClient) -> StrategyRunResult:
    header(f"СТРАТЕГИЯ 2: Sticky Facts (окно = {FACTS_WINDOW_SIZE}, + facts)")
    print(f"  Извлекает ключевые факты после каждого обмена.")
    print(f"  В запрос отправляет: facts + последние {FACTS_WINDOW_SIZE} сообщений.")

    strategy = StickyFactsStrategy(
        window_size=FACTS_WINDOW_SIZE,
        llm_client=client,
    )
    agent = StrategyAgent(
        llm_client=client,
        strategy=strategy,
        system_prompt=SYSTEM_PROMPT,
        token_counter=TiktokenCounter(),
    )

    result = run_strategy(
        strategy_name=strategy.name,
        agent=agent,
        questions=SCENARIO_MESSAGES,
        recall_questions=RECALL_QUESTIONS,
    )

    # Показываем извлечённые факты
    print(f"\n  Извлечённые факты ({len(strategy.facts)} шт.):")
    for k, v in strategy.facts.items():
        print(f"    {k}: {v}")

    stats = strategy.get_stats()
    result.extra_info = stats
    result.extra_info["overhead_tokens"] = stats.get("facts_update_tokens", 0)
    return result


# ===========================================================================
# Прогон 3: Branching
# ===========================================================================

def run_branching(client: QwenHttpClient) -> StrategyRunResult:
    header("СТРАТЕГИЯ 3: Branching (ветки диалога)")
    print("  Основной диалог → checkpoint → 2 ветки (подписка vs реклама).")

    strategy = BranchingStrategy()
    agent = StrategyAgent(
        llm_client=client,
        strategy=strategy,
        system_prompt=SYSTEM_PROMPT,
        token_counter=TiktokenCounter(),
    )

    # Основной диалог
    result = run_strategy(
        strategy_name=strategy.name,
        agent=agent,
        questions=SCENARIO_MESSAGES,
        recall_questions=[],  # recall после веток
    )

    # Сохраняем checkpoint
    print(f"\n  --- Сохраняем checkpoint 'base' ---")
    strategy.save_checkpoint("base", description="После основного ТЗ")
    print(f"  Checkpoint сохранён: {len(strategy.messages)} сообщений.\n")

    # Ветка A: подписка
    print(f"  {'='*40}")
    print(f"  ВЕТКА A: Монетизация через подписку")
    print(f"  {'='*40}")
    strategy.create_branch("subscription", "base", description="Подписка")
    strategy.switch_branch("subscription")

    for i, q in enumerate(BRANCH_A_QUESTIONS, 1):
        print(f"\n  [Ветка A, ход {i}]")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")
            break
        show_a(answer)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
            label="(branch-A)",
        )
        result.turns.append(
            make_turn(len(SCENARIO_MESSAGES) + i, q, answer, usage)
        )
        time.sleep(0.5)

    # Ветка B: реклама
    print(f"\n  {'='*40}")
    print(f"  ВЕТКА B: Монетизация через рекламу")
    print(f"  {'='*40}")
    strategy.create_branch("ads", "base", description="Реклама")
    strategy.switch_branch("ads")

    for i, q in enumerate(BRANCH_B_QUESTIONS, 1):
        print(f"\n  [Ветка B, ход {i}]")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")
            break
        show_a(answer)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
            label="(branch-B)",
        )
        result.turns.append(
            make_turn(len(SCENARIO_MESSAGES) + len(BRANCH_A_QUESTIONS) + i, q, answer, usage)
        )
        time.sleep(0.5)

    # Переключаемся на main и задаём recall-вопросы
    print(f"\n  --- Переключаемся на main для контрольных вопросов ---")
    strategy.switch_branch("main")

    print(f"\n  --- Контрольные вопросы (проверка памяти) ---\n")
    for i, q in enumerate(RECALL_QUESTIONS, 1):
        print(f"  [Recall {i}]")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")
            answer = f"[ERROR: {exc}]"
        show_a(answer, max_chars=400)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
            label="(recall)",
        )
        result.recall_turns.append(
            make_turn(100 + i, q, answer, usage)
        )
        time.sleep(0.5)

    result.extra_info = strategy.get_stats()
    result.extra_info["branches_info"] = {
        b: strategy.get_branch_info(b) for b in strategy.branches
    }
    return result


# ===========================================================================
# Сравнительный анализ
# ===========================================================================

def print_comparison(
    r_sliding: StrategyRunResult,
    r_facts: StrategyRunResult,
    r_branching: StrategyRunResult,
) -> None:
    header("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ТРЁХ СТРАТЕГИЙ")

    results = [r_sliding, r_facts, r_branching]
    labels = ["Sliding Window", "Sticky Facts", "Branching"]
    short_labels = ["SlidWin", "Facts", "Branch"]

    # --- Таблица: общие метрики ---
    print("\n  Общие метрики:\n")
    col = 14
    print(f"  {'Показатель':<35} {short_labels[0]:>{col}} {short_labels[1]:>{col}} {short_labels[2]:>{col}}")
    sep("-")

    def row(label: str, vals: list[str]) -> None:
        print(f"  {label:<35} {vals[0]:>{col}} {vals[1]:>{col}} {vals[2]:>{col}}")

    row("Ходов (диалог)", [str(len(r.turns)) for r in results])
    row("Avg prompt токенов/ход", [f"{r.avg_prompt_tokens:,.0f}" for r in results])
    row("Последний prompt токенов", [f"{r.last_prompt_tokens:,}" for r in results])
    row("Токены диалога (всего)", [f"{r.total_dialog_tokens:,}" for r in results])
    row("Токены recall (всего)", [f"{r.total_recall_tokens:,}" for r in results])

    overhead_labels = []
    for r in results:
        overhead = r.extra_info.get("overhead_tokens", 0)
        if overhead:
            overhead_labels.append(f"{overhead:,}")
        else:
            overhead_labels.append("0")
    row("Накладные расходы (overhead)", overhead_labels)
    row("ИТОГО ТОКЕНОВ", [f"{r.grand_total_tokens:,}" for r in results])
    sep("-")

    # --- Рост контекста ---
    print("\n\n  Рост prompt-токенов: первый ход → последний ход:\n")
    for label, r in zip(labels, results):
        if r.turns:
            first = r.turns[0].prompt_tokens
            # Берём последний основной ход (до recall)
            dialog_turns = [t for t in r.turns if t.turn <= len(SCENARIO_MESSAGES)]
            last = dialog_turns[-1].prompt_tokens if dialog_turns else r.turns[-1].prompt_tokens
            print(f"    {label:<20}: {first:>5} -> {last:>5} токенов (+{last - first:,})")

    # --- Ответы на контрольные вопросы ---
    print("\n\n  Ответы на контрольные вопросы (качество памяти):")
    sep("-")

    for i in range(len(RECALL_QUESTIONS)):
        print(f"\n  [Recall {i+1}] {RECALL_QUESTIONS[i]}\n")
        for label, r in zip(labels, results):
            if i < len(r.recall_turns):
                t = r.recall_turns[i]
                short = t.answer[:200] + ("..." if len(t.answer) > 200 else "")
                print(f"  {label} ({t.prompt_tokens} prompt-tok.):")
                for line in textwrap.wrap(short, width=WIDTH - 4):
                    print(f"    {line}")
                print()
        sep("-")

    # --- Дополнительная инфо для Branching ---
    if r_branching.extra_info.get("branches_info"):
        print("\n  Информация о ветках (Branching):")
        for b_id, info in r_branching.extra_info["branches_info"].items():
            total = info.get("total_messages", info.get("messages_count", "?"))
            desc = info.get("description", "")
            print(f"    [{b_id}] {total} сообщ. — {desc}")

    # --- Дополнительная инфо для Facts ---
    if r_facts.extra_info.get("facts"):
        print("\n  Извлечённые факты (Sticky Facts):")
        for k, v in r_facts.extra_info["facts"].items():
            short_v = v[:80] + ("..." if len(v) > 80 else "")
            print(f"    {k}: {short_v}")

    # --- ВЫВОДЫ ---
    print("\n")
    header("ВЫВОДЫ")

    min_tokens = min(r.grand_total_tokens for r in results)
    max_tokens = max(r.grand_total_tokens for r in results)

    print(f"""
  1. РАСХОД ТОКЕНОВ:
     - Sliding Window : {r_sliding.grand_total_tokens:>8,} токенов {"(минимум)" if r_sliding.grand_total_tokens == min_tokens else ""}
     - Sticky Facts   : {r_facts.grand_total_tokens:>8,} токенов {"(минимум)" if r_facts.grand_total_tokens == min_tokens else ""}
     - Branching      : {r_branching.grand_total_tokens:>8,} токенов {"(максимум — полная история + ветки)" if r_branching.grand_total_tokens == max_tokens else ""}

  2. КАЧЕСТВО ОТВЕТОВ (recall):
     - Sliding Window : теряет ранние детали при выходе за окно N={WINDOW_SIZE}
     - Sticky Facts   : сохраняет ключевые факты даже из давних сообщений
     - Branching      : полная история — максимальное качество recall

  3. СТАБИЛЬНОСТЬ:
     - Sliding Window : может потерять контекст если окно слишком мало
     - Sticky Facts   : стабильно за счёт facts, но зависит от качества извлечения
     - Branching      : полная стабильность в рамках каждой ветки

  4. УДОБСТВО ДЛЯ ПОЛЬЗОВАТЕЛЯ:
     - Sliding Window : прозрачен, не требует дополнительных действий
     - Sticky Facts   : автоматический, пользователь может проверять /facts
     - Branching      : требует явного управления (checkpoint, branch, switch)

  5. РЕКОМЕНДАЦИИ:
     - Sliding Window — для коротких диалогов, экономия токенов
     - Sticky Facts   — для длинных диалогов с важными деталями
     - Branching      — для исследования альтернативных решений
""")
    sep("=")


# ===========================================================================
# Точка входа
# ===========================================================================

def main() -> None:
    try:
        config = get_config()
    except ValueError as exc:
        print(f"\n  ОШИБКА: {exc}")
        return

    print("\n" + "#" * WIDTH)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ УПРАВЛЕНИЯ КОНТЕКСТОМ")
    print(f"  Модель: {config['model']}")
    print(f"  Сценарий: сбор ТЗ для мобильного приложения ({len(SCENARIO_MESSAGES)} сообщений)")
    print(f"  Контрольные вопросы: {len(RECALL_QUESTIONS)}")
    print(f"  Sliding Window: N={WINDOW_SIZE}")
    print(f"  Sticky Facts: window={FACTS_WINDOW_SIZE}")
    print(f"  Branching: основной диалог + 2 ветки")
    print("#" * WIDTH)

    client = QwenHttpClient(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        timeout=config["timeout"],
    )

    try:
        r_sliding = run_sliding_window(client)
        r_facts = run_sticky_facts(client)
        r_branching = run_branching(client)
        print_comparison(r_sliding, r_facts, r_branching)
    except KeyboardInterrupt:
        print("\n\n  Прервано пользователем.")
    except Exception as exc:
        print(f"\n  КРИТИЧЕСКАЯ ОШИБКА: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    main()
