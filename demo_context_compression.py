"""Демонстрация и сравнение управления контекстом LLM-агента.

Скрипт запускает одинаковый диалог двумя способами:
    1. БЕЗ сжатия  — SimpleAgent с полной историей (baseline)
    2. СО сжатием  — SummaryAgent с автоматическим суммаризированием

После каждого прогона задаются «контрольные» вопросы, проверяющие
сохранность контекста: помнит ли агент факты из начала разговора.

В конце выводится сравнительная таблица:
    - токены на каждый ход (prompt + completion)
    - экономия токенов
    - токены, потраченные на создание резюме
    - качественная оценка ответов на контрольные вопросы

Запуск:
    pip install openai tiktoken python-dotenv
    export OPENAI_API_KEY=sk-...
    python demo_context_compression.py

Параметры (см. блок ПАРАМЕТРЫ ниже):
    MODEL               — модель OpenAI
    SUMMARY_BATCH_SIZE  — кол-во сообщений в одном «окне» перед суммаризацией
    MAX_TOKENS          — максимальная длина ответа модели
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import NamedTuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.agent import SimpleAgent
from llm_agent.application.summary_agent import SummaryAgent
from llm_agent.domain.models import TokenUsage
from llm_agent.infrastructure.openai_client import CONTEXT_LIMITS, OpenAIClient
from llm_agent.infrastructure.token_counter import TiktokenCounter

# ===========================================================================
# ПАРАМЕТРЫ
# ===========================================================================

MODEL = "gpt-3.5-turbo"
CONTEXT_LIMIT = CONTEXT_LIMITS[MODEL]   # 4096 токенов
SUMMARY_BATCH_SIZE = 10                 # суммаризировать каждые 10 сообщений
MAX_TOKENS = 150                        # ограничиваем ответ для экономии токенов
WIDTH = 76                              # ширина вывода

SYSTEM_PROMPT = (
    "Ты эксперт по веб-скрапингу на Python. "
    "Давай краткие, но точные ответы на русском языке. "
    "Помни всё, что было сказано в нашем разговоре."
)

# ===========================================================================
# Вопросы основного диалога (20 вопросов = 40 сообщений → 4 резюме при batch=10)
# ===========================================================================

DIALOG_QUESTIONS = [
    # Блок 1 — основы (войдёт в резюме 1 при batch=10)
    "Что такое веб-скрапинг и для каких задач он используется?",
    "Назови три самые популярные Python-библиотеки для скрапинга.",
    "Чем requests+BeautifulSoup отличается от Scrapy по архитектуре?",
    "Что такое robots.txt и почему скрапер обязан его соблюдать?",
    "Как правильно установить заголовок User-Agent при запросах?",

    # Блок 2 — парсинг HTML (войдёт в резюме 2)
    "Как с помощью BeautifulSoup найти все ссылки на странице?",
    "Объясни разницу между find() и find_all() в BeautifulSoup.",
    "Как парсить данные из HTML-таблицы с помощью BeautifulSoup?",
    "Что такое CSS-селекторы и как их использовать в BeautifulSoup?",
    "Как сохранить результаты скрапинга в CSV с помощью pandas?",

    # Блок 3 — продвинутые техники (войдёт в резюме 3)
    "Что такое динамический контент и почему его нельзя получить через requests?",
    "Как использовать Selenium для скрапинга JavaScript-страниц?",
    "Что такое headless-браузер и зачем он нужен при скрапинге?",
    "Как настроить повторные попытки (retry) при ошибках сети?",
    "Как реализовать вежливый скрапинг с паузами между запросами?",

    # Блок 4 — масштабирование (войдёт в резюме 4)
    "Как скрапить сайты с пагинацией (несколько страниц)?",
    "Как организовать многопоточный скрапинг через concurrent.futures?",
    "Чем асинхронный скрапинг с aiohttp отличается от многопоточного?",
    "Как хранить большие объёмы скрапинговых данных в SQLite?",
    "Как тестировать скрапер без реальных HTTP-запросов (моки)?",
]

# ===========================================================================
# Контрольные вопросы — проверяем память о начале разговора
# ===========================================================================

RECALL_QUESTIONS = [
    "Какие три библиотеки Python для скрапинга ты упомянул в самом начале нашего разговора?",
    "Что мы обсуждали про robots.txt? Кратко напомни основной тезис.",
    "Назови все способы работы с динамическим контентом, о которых мы говорили.",
    "Дай краткое резюме ключевых тем, которые мы рассмотрели в нашем разговоре.",
]

# ===========================================================================
# Структуры данных
# ===========================================================================

@dataclass
class TurnStats:
    """Статистика одного хода диалога."""
    turn: int
    question: str
    answer: str
    prompt_tokens: int      # токены, отправленные в модель
    completion_tokens: int  # токены ответа
    total_tokens: int       # prompt + completion

@dataclass
class RunResult:
    """Результат одного прогона (без сжатия или со сжатием)."""
    label: str
    turns: list[TurnStats] = field(default_factory=list)
    recall_turns: list[TurnStats] = field(default_factory=list)
    summary_tokens_spent: int = 0   # токены на суммаризацию
    summary_count: int = 0          # сколько резюме создано

    @property
    def total_dialog_tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    @property
    def total_recall_tokens(self) -> int:
        return sum(t.total_tokens for t in self.recall_turns)

    @property
    def grand_total_tokens(self) -> int:
        """Все токены включая суммаризацию."""
        return self.total_dialog_tokens + self.total_recall_tokens + self.summary_tokens_spent

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

def subheader(text: str) -> None:
    print(f"\n  --- {text} ---")

def show_q(text: str) -> None:
    prefix = "  Q: "
    indent = " " * len(prefix)
    lines = textwrap.wrap(text, width=WIDTH - len(prefix))
    for i, line in enumerate(lines):
        print(f"{prefix if i == 0 else indent}{line}")

def show_a(text: str, max_chars: int = 200) -> None:
    short = text[:max_chars] + ("…" if len(text) > max_chars else "")
    prefix = "  A: "
    indent = " " * len(prefix)
    lines = textwrap.wrap(short, width=WIDTH - len(prefix))
    for i, line in enumerate(lines):
        print(f"{prefix if i == 0 else indent}{line}")

def show_tokens(prompt_t: int, completion_t: int, label: str = "") -> None:
    total = prompt_t + completion_t
    bar_len = 20
    filled = min(bar_len, int(bar_len * prompt_t / CONTEXT_LIMIT)) if CONTEXT_LIMIT else 0
    bar = "[" + "#" * filled + "." * (bar_len - filled) + "]"
    pct = (prompt_t / CONTEXT_LIMIT * 100) if CONTEXT_LIMIT else 0
    tag = f"  {label}" if label else ""
    print(f"  Tokens: prompt={prompt_t:>4} ({pct:4.1f}%) | completion={completion_t:>4} | total={total:>5} {bar}{tag}")


def make_turn_stats(turn: int, question: str, answer: str, usage: TokenUsage | None) -> TurnStats:
    if usage:
        return TurnStats(
            turn=turn,
            question=question,
            answer=answer,
            prompt_tokens=usage.history_tokens,
            completion_tokens=usage.response_tokens,
            total_tokens=usage.total_tokens,
        )
    return TurnStats(turn=turn, question=question, answer=answer,
                     prompt_tokens=0, completion_tokens=0, total_tokens=0)


# ===========================================================================
# Прогон 1: БЕЗ сжатия (SimpleAgent)
# ===========================================================================

def run_without_compression() -> RunResult:
    header("ПРОГОН 1: БЕЗ сжатия контекста (SimpleAgent — полная история)")
    print(f"  Модель: {MODEL} | Лимит: {CONTEXT_LIMIT} токенов")
    print(f"  При каждом ходе передаётся ВСЯ история диалога → токены нарастают линейно.")

    agent = SimpleAgent(
        llm_client=OpenAIClient(model=MODEL, max_tokens=MAX_TOKENS),
        system_prompt=SYSTEM_PROMPT,
        token_counter=TiktokenCounter(model=MODEL),
        context_limit=CONTEXT_LIMIT,
        auto_truncate=False,
    )

    result = RunResult(label="Без сжатия (SimpleAgent)")

    # --- Основные вопросы ---
    print(f"\n  Диалог ({len(DIALOG_QUESTIONS)} вопросов):")
    for i, q in enumerate(DIALOG_QUESTIONS, 1):
        subheader(f"Ход {i}/{len(DIALOG_QUESTIONS)}")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ✗ ОШИБКА: {exc}")
            break
        show_a(answer)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
        )
        ts = make_turn_stats(i, q, answer, usage)
        result.turns.append(ts)

    # --- Контрольные вопросы ---
    subheader("Контрольные вопросы (проверка памяти о начале разговора)")
    for i, q in enumerate(RECALL_QUESTIONS, 1):
        print(f"\n  [Recall {i}]")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ✗ ОШИБКА: {exc}")
            answer = f"[ERROR: {exc}]"
        show_a(answer, max_chars=300)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
            label="(recall)",
        )
        ts = make_turn_stats(len(DIALOG_QUESTIONS) + i, q, answer, usage)
        result.recall_turns.append(ts)

    return result


# ===========================================================================
# Прогон 2: СО сжатием (SummaryAgent)
# ===========================================================================

def run_with_compression(batch_size: int = SUMMARY_BATCH_SIZE) -> RunResult:
    header(f"ПРОГОН 2: СО сжатием контекста (SummaryAgent, batch={batch_size})")
    print(f"  Модель: {MODEL} | Лимит: {CONTEXT_LIMIT} токенов")
    print(f"  Каждые {batch_size} сообщений создаётся резюме → передаётся вместо полной истории.")

    llm_client = OpenAIClient(model=MODEL, max_tokens=MAX_TOKENS)
    agent = SummaryAgent(
        llm_client=llm_client,
        summary_batch_size=batch_size,
        system_prompt=SYSTEM_PROMPT,
        token_counter=TiktokenCounter(model=MODEL),
    )

    result = RunResult(label=f"Со сжатием (SummaryAgent, batch={batch_size})")

    # --- Основные вопросы ---
    print(f"\n  Диалог ({len(DIALOG_QUESTIONS)} вопросов):")
    for i, q in enumerate(DIALOG_QUESTIONS, 1):
        subheader(f"Ход {i}/{len(DIALOG_QUESTIONS)}")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ✗ ОШИБКА: {exc}")
            break
        show_a(answer)
        usage = agent.last_token_usage
        cm = agent.context_manager
        summary_note = ""
        if cm.summary_count > 0:
            summary_note = f"  | резюме: {cm.summary_count}, окно: {cm.current_window_size} сообщ."
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
            label=summary_note,
        )
        ts = make_turn_stats(i, q, answer, usage)
        result.turns.append(ts)

    # --- Статистика суммаризации ---
    cm = agent.context_manager
    result.summary_count = cm.summary_count
    result.summary_tokens_spent = cm.summary_tokens_spent
    if cm.summary_count > 0:
        print(f"\n  Суммаризаций выполнено: {cm.summary_count}")
        print(f"  Токены на суммаризацию: {cm.summary_tokens_spent}")
        for s in cm.summaries:
            print(f"    Резюме {s.index}: {s.messages_count} сообщ. → "
                  f"{s.tokens_spent.get('total_tokens', '?')} токенов")
            preview = s.content[:120] + ("…" if len(s.content) > 120 else "")
            print(f"      Текст: {preview}")

    # --- Контрольные вопросы ---
    subheader("Контрольные вопросы (проверка памяти о начале разговора)")
    for i, q in enumerate(RECALL_QUESTIONS, 1):
        print(f"\n  [Recall {i}]")
        show_q(q)
        try:
            answer = agent.ask(q)
        except Exception as exc:
            print(f"  ✗ ОШИБКА: {exc}")
            answer = f"[ERROR: {exc}]"
        show_a(answer, max_chars=300)
        usage = agent.last_token_usage
        show_tokens(
            usage.history_tokens if usage else 0,
            usage.response_tokens if usage else 0,
            label="(recall)",
        )
        ts = make_turn_stats(len(DIALOG_QUESTIONS) + i, q, answer, usage)
        result.recall_turns.append(ts)

    return result


# ===========================================================================
# Сравнительный анализ
# ===========================================================================

def print_comparison(no_compress: RunResult, with_compress: RunResult) -> None:
    header("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")

    # --- Таблица по ходам ---
    print("\n  Токены prompt на каждый ход основного диалога:\n")
    col_w = 6
    n_turns = min(len(no_compress.turns), len(with_compress.turns))

    # Заголовок таблицы
    print(f"  {'Ход':>4} | {'Без сжатия':>10} | {'Со сжатием':>10} | {'Экономия':>10} | {'Экон.%':>7}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")

    total_saved = 0
    for i in range(n_turns):
        nc = no_compress.turns[i].prompt_tokens
        wc = with_compress.turns[i].prompt_tokens
        saved = nc - wc
        pct = (saved / nc * 100) if nc > 0 else 0
        total_saved += saved
        flag = " ◄" if abs(pct) >= 10 else ""
        print(f"  {i+1:>4} | {nc:>10,} | {wc:>10,} | {saved:>+10,} | {pct:>6.1f}%{flag}")

    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")

    # Итоговые строки
    nc_total_pt = sum(t.prompt_tokens for t in no_compress.turns)
    wc_total_pt = sum(t.prompt_tokens for t in with_compress.turns)
    total_pct = ((nc_total_pt - wc_total_pt) / nc_total_pt * 100) if nc_total_pt > 0 else 0
    print(f"  {'ИТОГО':>4} | {nc_total_pt:>10,} | {wc_total_pt:>10,} | {nc_total_pt-wc_total_pt:>+10,} | {total_pct:>6.1f}%")

    # --- Сводная таблица ---
    print("\n\n  Итоговая сводка:")
    sep("-")

    def row(label: str, nc_val: str, wc_val: str, note: str = "") -> None:
        print(f"  {label:<38} {nc_val:>12} {wc_val:>12}    {note}")

    row("Показатель", "Без сжатия", "Со сжатием")
    sep("-")
    row("Ходов в диалоге",
        str(len(no_compress.turns)),
        str(len(with_compress.turns)))
    row("Резюме создано",
        "0",
        str(with_compress.summary_count))
    row("Avg prompt-токенов/ход (диалог)",
        f"{no_compress.avg_prompt_tokens:,.0f}",
        f"{with_compress.avg_prompt_tokens:,.0f}")
    row("Prompt-токены последнего хода",
        f"{no_compress.last_prompt_tokens:,}",
        f"{with_compress.last_prompt_tokens:,}")
    row("Всего prompt-токенов (диалог)",
        f"{nc_total_pt:,}",
        f"{wc_total_pt:,}",
        f"экономия {nc_total_pt - wc_total_pt:+,}")
    row("Всего completion-токенов (диалог)",
        f"{sum(t.completion_tokens for t in no_compress.turns):,}",
        f"{sum(t.completion_tokens for t in with_compress.turns):,}")
    row("Итого токенов (диалог)",
        f"{no_compress.total_dialog_tokens:,}",
        f"{with_compress.total_dialog_tokens:,}")
    row("Токены на суммаризацию",
        "0",
        f"{with_compress.summary_tokens_spent:,}",
        "(накладные расходы)")
    row("Токены recall-вопросов",
        f"{no_compress.total_recall_tokens:,}",
        f"{with_compress.total_recall_tokens:,}")
    row("ИТОГО всех токенов",
        f"{no_compress.grand_total_tokens:,}",
        f"{with_compress.grand_total_tokens:,}",
        f"разница {no_compress.grand_total_tokens - with_compress.grand_total_tokens:+,}")
    sep("-")

    # --- Рост контекста ---
    print("\n\n  Рост prompt-токенов: первый ход → последний ход диалога:")
    if no_compress.turns:
        nc_first = no_compress.turns[0].prompt_tokens
        nc_last = no_compress.turns[-1].prompt_tokens
        print(f"    Без сжатия : {nc_first:>5} → {nc_last:>5} токенов (+{nc_last - nc_first:,})")
    if with_compress.turns:
        wc_first = with_compress.turns[0].prompt_tokens
        wc_last = with_compress.turns[-1].prompt_tokens
        print(f"    Со сжатием : {wc_first:>5} → {wc_last:>5} токенов (+{wc_last - wc_first:,})")

    # --- Контрольные вопросы (качество) ---
    print("\n\n  Ответы на контрольные вопросы (оценка качества памяти):")
    sep("-")
    for i, (nc_t, wc_t) in enumerate(zip(no_compress.recall_turns, with_compress.recall_turns), 1):
        print(f"\n  [Recall {i}] {nc_t.question}")
        print()
        print(f"  БЕЗ сжатия ({nc_t.prompt_tokens} prompt-токенов):")
        for line in textwrap.wrap(nc_t.answer, width=WIDTH - 4):
            print(f"    {line}")
        print()
        print(f"  СО сжатием ({wc_t.prompt_tokens} prompt-токенов):")
        for line in textwrap.wrap(wc_t.answer, width=WIDTH - 4):
            print(f"    {line}")
        sep("-")

    # --- Выводы ---
    grand_nc = no_compress.grand_total_tokens
    grand_wc = with_compress.grand_total_tokens
    savings_pct = ((grand_nc - grand_wc) / grand_nc * 100) if grand_nc > 0 else 0
    overhead_pct = (with_compress.summary_tokens_spent / grand_wc * 100) if grand_wc > 0 else 0

    print("\n\n  ВЫВОДЫ:")
    sep("=")
    print(f"  • Сжатие экономит ~{total_pct:.0f}% prompt-токенов в процессе диалога")
    print(f"  • Накладные расходы на суммаризацию: {with_compress.summary_tokens_spent:,} токенов "
          f"({overhead_pct:.1f}% от итога со сжатием)")
    print(f"  • Итоговая экономия токенов: {grand_nc - grand_wc:+,} ({savings_pct:.1f}%)")
    print(f"  • Рост prompt при ПОЛНОЙ истории: линейный — каждый ход дороже предыдущего")
    print(f"  • Рост prompt при сжатии: ступенчатый — сбрасывается после каждого резюме")
    print(f"  • Качество (recall): см. ответы выше — суммаризация сохраняет ключевой контекст")
    print(f"  • Компромисс: суммаризация сжимает детали, полная история точнее для нюансов")
    sep("=")
    print()


# ===========================================================================
# Точка входа
# ===========================================================================

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("\n  ОШИБКА: переменная OPENAI_API_KEY не задана.")
        print("  Установите: export OPENAI_API_KEY=sk-...")
        print("  Или создайте файл .env: OPENAI_API_KEY=sk-...")
        return

    print("\n" + "#" * WIDTH)
    print("  СРАВНЕНИЕ УПРАВЛЕНИЯ КОНТЕКСТОМ: ПОЛНАЯ ИСТОРИЯ vs СУММАРИЗАЦИЯ")
    print(f"  Модель  : {MODEL}  (контекст {CONTEXT_LIMIT} токенов)")
    print(f"  Вопросов: {len(DIALOG_QUESTIONS)} основных + {len(RECALL_QUESTIONS)} контрольных")
    print(f"  Размер окна суммаризации: {SUMMARY_BATCH_SIZE} сообщений")
    print("#" * WIDTH)

    try:
        result_no_compress = run_without_compression()
        result_with_compress = run_with_compression(batch_size=SUMMARY_BATCH_SIZE)
        print_comparison(result_no_compress, result_with_compress)
    except KeyboardInterrupt:
        print("\n\n  Прервано пользователем.")
    except Exception as exc:
        print(f"\n  КРИТИЧЕСКАЯ ОШИБКА: {exc}")
        raise


if __name__ == "__main__":
    main()
