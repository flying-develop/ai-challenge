#!/usr/bin/env python3
"""demo_task_fsm.py — демонстрация Task Orchestrator (конечный автомат задач).

Запуск: python demo_task_fsm.py

Скрипт без интерактивного ввода демонстрирует полный цикл:
  Шаг 1 — Создание задачи → вход в planning
  Шаг 2 — Planning: LLM строит план (2 exchange-а)
  Шаг 3 — Пауза в planning, перезагрузка из БД
  Шаг 4 — Resume → продолжение без повтора контекста
  Шаг 5 — /task next → переход в execution
  Шаг 6 — Execution: выполнение с отслеживанием [STEP_DONE:N]
  Шаг 7 — Пауза в execution, resume
  Шаг 8 — /task next → validation
  Шаг 9 — Validation → /task next → done
  Шаг 10 — /task history + /task status
"""

from __future__ import annotations

import json
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.context_strategies import SlidingWindowStrategy
from llm_agent.application.strategy_agent import StrategyAgent
from llm_agent.infrastructure.llm_factory import build_client, current_provider_from_env
from llm_agent.infrastructure.token_counter import TiktokenCounter
from llm_agent.memory.manager import MemoryManager
from llm_agent.tasks.orchestrator import TaskOrchestrator

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

DEMO_DB = os.path.join(os.path.expanduser("~"), ".llm-agent", "demo_task_fsm.db")

TASK_TITLE = "Написать утилиту подсчёта слов в тексте на Python"

# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def banner(text: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {text}")
    print(f"{'=' * 62}\n")


def step_header(num: int, title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  Шаг {num}: {title}")
    print(f"{'─' * 62}\n")


def show_task_info(orch: TaskOrchestrator) -> None:
    print(orch.get_status())


def ask_and_show(orch: TaskOrchestrator, msg: str) -> str:
    print(f"  >> Пользователь: {msg}")
    reply = orch.handle_message(msg)
    # Показываем первые 500 символов ответа
    preview = reply[:500] + ("..." if len(reply) > 500 else "")
    print(f"  << Агент: {preview}")
    print()
    return reply


def cleanup_db() -> None:
    """Удалить демо-БД для чистого запуска."""
    if os.path.exists(DEMO_DB):
        os.remove(DEMO_DB)
        # Удалить WAL/SHM файлы если есть
        for suffix in ("-wal", "-shm"):
            path = DEMO_DB + suffix
            if os.path.exists(path):
                os.remove(path)


def create_components():
    """Создать все необходимые компоненты."""
    provider = current_provider_from_env()
    client = build_client(provider)
    token_counter = TiktokenCounter()
    strategy = SlidingWindowStrategy(window_size=20)
    memory_manager = MemoryManager(DEMO_DB)

    agent = StrategyAgent(
        llm_client=client,
        strategy=strategy,
        system_prompt="Ты полезный ассистент. Отвечай на русском языке.",
        token_counter=token_counter,
        provider_name=provider,
    )

    orchestrator = TaskOrchestrator(
        db_path=DEMO_DB,
        agent=agent,
        memory_manager=memory_manager,
    )

    return agent, memory_manager, orchestrator


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------

def main() -> None:
    banner("ДЕМО: Task Orchestrator — конечный автомат задач")

    # Чистый запуск
    cleanup_db()

    agent, memory_manager, orchestrator = create_components()

    try:
        # ── Шаг 1: Создание задачи ───────────────────────────────────────
        step_header(1, "Создание задачи")
        task = orchestrator.create_task(TASK_TITLE)
        print(f"  Задача #{task.id} создана: \"{task.title}\"")
        print(f"  Статус: {task.status.value}")
        print(f"  Ожидается: {task.expected_action.value}")
        show_task_info(orchestrator)

        # ── Шаг 2: Planning — LLM строит план ────────────────────────────
        step_header(2, "Planning: LLM строит план")
        print("  System prompt для фазы planning:")
        print(f"  {agent._system_prompt[:200]}...\n")

        reply = ask_and_show(
            orchestrator,
            "Нужна консольная утилита на Python, которая считает количество слов "
            "в текстовом файле. Вход: путь к файлу. Выход: общее число слов и "
            "топ-5 самых частых слов. Составь план."
        )

        time.sleep(1)

        # Если артефакт не найден с первого раза — попросить формализовать
        if not orchestrator.active_task.artifact:
            reply = ask_and_show(
                orchestrator,
                "Сформулируй готовый план. Выведи его в формате "
                "[ARTIFACT:PLAN] с JSON: {\"steps\": [{\"id\": N, \"title\": \"...\", "
                "\"description\": \"...\"}]}."
            )
            time.sleep(1)

        print("  Артефакт planning:")
        print(f"  {json.dumps(orchestrator.active_task.artifact, ensure_ascii=False, indent=2)[:500]}")

        # ── Шаг 3: Пауза в planning ─────────────────────────────────────
        step_header(3, "Пауза в planning")
        msg = orchestrator.pause_task()
        print(f"  {msg}")

        # Имитация перезапуска: пересоздаём компоненты
        print("\n  [Имитация перезапуска: пересоздаём компоненты из БД]")
        orchestrator.close()
        memory_manager.close()

        agent, memory_manager, orchestrator = create_components()
        print("  Компоненты пересозданы.")

        # ── Шаг 4: Resume ────────────────────────────────────────────────
        step_header(4, "Resume: продолжение без повтора контекста")

        # Загружаем задачу
        tasks = orchestrator.list_tasks()
        print(f"  Найдено задач в БД: {len(tasks)}")
        task = orchestrator.load_task(tasks[0].id)
        print(f"  Загружена: #{task.id} \"{task.title}\" ({task.status.value})")

        msg = orchestrator.resume_task()
        print(f"  {msg}")
        print(f"\n  System prompt после resume (первые 300 символов):")
        print(f"  {agent._system_prompt[:300]}...")

        # Проверяем: если артефакта плана ещё нет, дозапрашиваем
        if not orchestrator.active_task.artifact:
            reply = ask_and_show(
                orchestrator,
                "Мы обсуждали утилиту подсчёта слов. Сформулируй план. "
                "Выведи [ARTIFACT:PLAN] с JSON."
            )
            time.sleep(1)

        # ── Шаг 5: Переход в execution ───────────────────────────────────
        step_header(5, "Переход в execution (/task next)")
        try:
            msg = orchestrator.next_phase()
            print(f"  {msg}")
        except ValueError as e:
            print(f"  Не удалось перейти: {e}")
            print("  Запрашиваем план ещё раз...")
            reply = ask_and_show(
                orchestrator,
                "Выведи план в формате [ARTIFACT:PLAN] с JSON: "
                "{\"steps\": [{\"id\": 1, \"title\": \"...\", \"description\": \"...\"}]}"
            )
            time.sleep(1)
            msg = orchestrator.next_phase()
            print(f"  {msg}")

        show_task_info(orchestrator)
        print(f"  System prompt для execution (первые 300 символов):")
        print(f"  {agent._system_prompt[:300]}...")

        # ── Шаг 6: Execution — выполнение ────────────────────────────────
        step_header(6, "Execution: выполнение с отслеживанием шагов")

        reply = ask_and_show(
            orchestrator,
            "Начинай выполнение с первого шага. После каждого выводи [STEP_DONE:N]."
        )
        time.sleep(1)

        print(f"  current_step: {orchestrator.active_task.current_step}")

        reply = ask_and_show(
            orchestrator,
            "Продолжай выполнение остальных шагов. Когда закончишь все — "
            "выведи [ARTIFACT:RESULT] с JSON: {\"summary\": \"...\", \"outputs\": [...]}."
        )
        time.sleep(1)

        print(f"  current_step: {orchestrator.active_task.current_step}")
        print(f"  Артефакт execution: {json.dumps(orchestrator.active_task.artifact, ensure_ascii=False)[:300]}")

        # ── Шаг 7: Пауза в execution, resume ─────────────────────────────
        step_header(7, "Пауза/Resume в execution")

        if not orchestrator.active_task.artifact:
            # Если LLM ещё не выдал ARTIFACT:RESULT — попросим
            reply = ask_and_show(
                orchestrator,
                "Заверши выполнение. Выведи [ARTIFACT:RESULT] с JSON."
            )
            time.sleep(1)

        msg = orchestrator.pause_task()
        print(f"  Пауза: {msg}")

        # Перезагрузка
        orchestrator.close()
        memory_manager.close()
        agent, memory_manager, orchestrator = create_components()

        tasks = orchestrator.list_tasks()
        orchestrator.load_task(tasks[0].id)
        msg = orchestrator.resume_task()
        print(f"  Resume: {msg}")

        # ── Шаг 8: Переход в validation ──────────────────────────────────
        step_header(8, "Переход в validation (/task next)")

        # Если артефакта execution нет — запросим
        if not orchestrator.active_task.artifact:
            reply = ask_and_show(
                orchestrator,
                "Выведи итоговый результат: [ARTIFACT:RESULT] с JSON: "
                "{\"summary\": \"...\", \"outputs\": [\"...\"]}"
            )
            time.sleep(1)

        try:
            msg = orchestrator.next_phase()
            print(f"  {msg}")
        except ValueError as e:
            print(f"  Не удалось перейти: {e}")
            reply = ask_and_show(
                orchestrator,
                "Выведи [ARTIFACT:RESULT] с JSON: {\"summary\": \"результат\", \"outputs\": []}"
            )
            time.sleep(1)
            msg = orchestrator.next_phase()
            print(f"  {msg}")

        show_task_info(orchestrator)

        # ── Шаг 9: Validation → done ─────────────────────────────────────
        step_header(9, "Validation → Done")

        reply = ask_and_show(
            orchestrator,
            "Проведи валидацию результата. Выведи "
            "[ARTIFACT:VALIDATION] с JSON: {\"passed\": true, \"issues\": [], "
            "\"recommendations\": []}."
        )
        time.sleep(1)

        try:
            msg = orchestrator.next_phase()
            print(f"  {msg}")
        except ValueError as e:
            print(f"  Не удалось перейти: {e}")
            reply = ask_and_show(
                orchestrator,
                "Выведи [ARTIFACT:VALIDATION] с JSON: "
                "{\"passed\": true, \"issues\": [], \"recommendations\": []}"
            )
            time.sleep(1)
            msg = orchestrator.next_phase()
            print(f"  {msg}")

        # ── Шаг 10: История и финальный статус ────────────────────────────
        step_header(10, "Финальный вывод: history + list")

        # Загрузим задачу для просмотра истории
        tasks = orchestrator.list_tasks()
        if tasks:
            # active_task уже None после done, загрузим для просмотра
            task_data = orchestrator.get_task(tasks[0].id)
            if task_data:
                print(f"  Задача #{task_data.id}: \"{task_data.title}\"")
                print(f"  Статус: {task_data.status.value}")
                print(f"  Завершённых фаз: {len(task_data.history)}")
                for i, entry in enumerate(task_data.history, 1):
                    phase = entry.get("phase", "?")
                    completed = entry.get("completed_at", "?")
                    artifact_preview = json.dumps(
                        entry.get("artifact", {}), ensure_ascii=False
                    )[:200]
                    print(f"\n  [{i}] Фаза: {phase}")
                    print(f"      Завершена: {completed}")
                    print(f"      Артефакт: {artifact_preview}")

        # Список всех задач
        print("\n  Список задач:")
        for t in tasks:
            print(f"    [#{t.id}] {t.title} ({t.status.value})")

        banner("ДЕМО ЗАВЕРШЕНО")

    except Exception as e:
        print(f"\n  ОШИБКА: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    finally:
        orchestrator.close()
        memory_manager.close()


if __name__ == "__main__":
    main()
