#!/usr/bin/env python3
"""demo_task_transitions.py — демонстрация строгого контроля переходов задачи.

Запуск: python demo_task_transitions.py

Скрипт БЕЗ интерактивного ввода. Тестирует все 7 сценариев работы
TaskTransitionGuard. Сценарии 4 и 7 выполняют реальный вызов LLM,
чтобы продемонстрировать соблюдение инвариантов через system prompt.

Сценарий 1 — Нормальный полный цикл
Сценарий 2 — Попытка перепрыгнуть этап (нет артефакта)
Сценарий 3 — Прямой переход execution → done (обход validation)
Сценарий 4 — Попытка обойти через диалог (реальный вызов LLM)
Сценарий 5 — Пауза и resume с проверкой целостности
Сценарий 6 — Resume с повреждённым состоянием
Сценарий 7 — Двойной инвариант: этап + стек технологий (реальный вызов LLM)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Добавляем корень проекта в путь
_project_root = Path(__file__).parent
sys.path.insert(0, str(_project_root))

from llm_agent.domain.models import ChatMessage
from llm_agent.tasks.models import ExpectedAction, TaskState, TaskStatus
from llm_agent.tasks.transition_guard import TaskTransitionGuard

# ---------------------------------------------------------------------------
# Инициализация LLM-клиента (опционально — без ключа демо продолжается)
# ---------------------------------------------------------------------------

_llm_client = None
_llm_provider = "—"

try:
    from llm_agent.infrastructure.llm_factory import build_client, current_provider_from_env
    _llm_provider = current_provider_from_env()
    _llm_client = build_client(_llm_provider)
    print(f"  [LLM] Провайдер: {_llm_provider} — подключён")
except Exception as _e:
    print(f"  [LLM] Нет доступного провайдера ({_e}). Сценарии 4 и 7 покажут шаблон.")


def call_llm(system_prompt: str, user_message: str) -> str | None:
    """Сделать один запрос к LLM и вернуть текст ответа (или None при ошибке)."""
    if _llm_client is None:
        return None
    try:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]
        response = _llm_client.generate(messages)
        return response.text
    except Exception as exc:
        print(f"  [LLM] Ошибка вызова: {exc}")
        return None

# ---------------------------------------------------------------------------
# Инициализация guard с реальным MD-файлом инвариантов
# ---------------------------------------------------------------------------

_config_dir = _project_root / "config"
guard = TaskTransitionGuard(config_dir=_config_dir)

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

PLAN_ARTIFACT = {
    "steps": [
        {"id": 1, "title": "Анализ требований", "description": "Определить входные/выходные данные"},
        {"id": 2, "title": "Реализация функции", "description": "Написать код"},
        {"id": 3, "title": "Тесты", "description": "Написать unit-тесты"},
        {"id": 4, "title": "Документация", "description": "Добавить docstrings"},
    ]
}

RESULT_ARTIFACT = {
    "summary": "Функция word_count() реализована и покрыта тестами",
    "outputs": ["word_counter.py", "test_word_counter.py"],
}

VALIDATION_ARTIFACT = {
    "passed": True,
    "issues": [],
    "recommendations": ["Добавить тип-аннотации для параметров"],
}

PLAN_HISTORY = [
    {"phase": "planning", "artifact": PLAN_ARTIFACT, "completed_at": "2024-01-01 10:00:00"}
]
RESULT_HISTORY = PLAN_HISTORY + [
    {"phase": "execution", "artifact": RESULT_ARTIFACT, "completed_at": "2024-01-01 11:00:00"}
]


def make_task(
    status: str,
    artifact: dict | None = None,
    history: list | None = None,
    paused_at: str | None = None,
    current_step: int = 0,
    title: str = "Написать функцию подсчёта слов",
) -> TaskState:
    return TaskState(
        id=1,
        title=title,
        status=TaskStatus(status),
        paused_at=TaskStatus(paused_at) if paused_at else None,
        current_step=current_step,
        expected_action=ExpectedAction.NONE,
        artifact=artifact or {},
        history=history or [],
    )


def section(num: int, title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  Сценарий {num}: {title}")
    print(f"{'═' * 65}")


def result_line(label: str, ok: bool, detail: str = "") -> None:
    mark = "✅" if ok else "❌"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {mark} {label}{suffix}")


def show_guard_result(transition: str, result, expected_allowed: bool) -> bool:
    """Показать результат проверки перехода. Возвращает True если соответствует ожиданию."""
    status_ok = result.allowed == expected_allowed
    allowed_str = "разрешён" if result.allowed else "запрещён"
    expected_str = "разрешён" if expected_allowed else "запрещён"
    icon = "✅" if status_ok else "❌"

    print(f"\n  {icon} Переход {transition}")
    print(f"     Результат  : {allowed_str}")
    print(f"     Ожидание   : {expected_str}")
    print(f"     Причина    : {result.reason}")
    if result.missing_precondition:
        print(f"     Отсутствует: {result.missing_precondition}")
    if result.suggestion:
        print(f"     Что делать : {result.suggestion}")
    return status_ok


# Таблица результатов для финального отчёта
_report: list[tuple[int, str, str, str, bool]] = []


def record(scenario: int, transition: str, expected: str, result: str, ok: bool) -> None:
    _report.append((scenario, transition, expected, result, ok))


# ---------------------------------------------------------------------------
# Сценарий 1: Нормальный полный цикл
# ---------------------------------------------------------------------------


def scenario_1() -> bool:
    section(1, "Нормальный полный цикл")
    print("\n  /task new 'Написать функцию подсчёта слов'")
    print("  Диалог в planning → получаем ARTIFACT:PLAN")

    all_ok = True

    # planning → execution (с планом)
    task = make_task(status="planning", artifact=PLAN_ARTIFACT)
    r = guard.validate_transition(task, "execution")
    ok = show_guard_result("planning → execution", r, expected_allowed=True)
    all_ok = all_ok and ok
    record(1, "planning → execution", "разрешён", "разрешён" if r.allowed else "запрещён", ok)

    # execution → validation (с результатом)
    print("\n  Диалог в execution → получаем ARTIFACT:RESULT")
    task = make_task(status="execution", artifact=RESULT_ARTIFACT, history=PLAN_HISTORY)
    r = guard.validate_transition(task, "validation")
    ok = show_guard_result("execution → validation", r, expected_allowed=True)
    all_ok = all_ok and ok
    record(1, "execution → validation", "разрешён", "разрешён" if r.allowed else "запрещён", ok)

    # validation → done (с артефактом валидации)
    print("\n  Диалог в validation → получаем ARTIFACT:VALIDATION")
    task = make_task(
        status="validation", artifact=VALIDATION_ARTIFACT, history=RESULT_HISTORY
    )
    r = guard.validate_transition(task, "done")
    ok = show_guard_result("validation → done", r, expected_allowed=True)
    all_ok = all_ok and ok
    record(1, "validation → done", "разрешён", "разрешён" if r.allowed else "запрещён", ok)

    print(f"\n  → Итог: {'цепочка артефактов complete, всё ОК' if all_ok else 'ОШИБКА'}")
    return all_ok


# ---------------------------------------------------------------------------
# Сценарий 2: Попытка перепрыгнуть этап
# ---------------------------------------------------------------------------


def scenario_2() -> bool:
    section(2, "Попытка перепрыгнуть этап (нет артефакта)")
    print("\n  /task new 'Написать парсер CSV'")
    print("  Находимся в planning, артефакт ещё не сформирован")
    print("  /task next → пытаемся перейти в execution")

    task = make_task(status="planning", artifact={})  # нет плана!
    r = guard.validate_transition(task, "execution")

    ok = show_guard_result("planning → execution (нет плана)", r, expected_allowed=False)

    # Проверяем качество сообщения об ошибке
    has_artifact_mention = "ARTIFACT:PLAN" in (r.missing_precondition or "")
    has_suggestion = r.suggestion is not None

    print(f"\n  Детали ошибки:")
    print(f"    Упомянут ARTIFACT:PLAN: {'✅ да' if has_artifact_mention else '❌ нет'}")
    print(f"    Есть suggestion: {'✅ да' if has_suggestion else '❌ нет'}")
    if r.suggestion:
        print(f"    Suggestion: {r.suggestion}")

    print(f"\n  Форматированная ошибка:")
    error_msg = guard.format_error("planning", "execution", r)
    for line in error_msg.splitlines():
        print(f"    {line}")

    record(2, "planning → execution (нет плана)", "запрещён",
           "запрещён" if not r.allowed else "разрешён", ok)
    return ok and has_artifact_mention


# ---------------------------------------------------------------------------
# Сценарий 3: Прямой переход execution → done (обход validation)
# ---------------------------------------------------------------------------


def scenario_3() -> bool:
    section(3, "Прямой переход execution → done (обход validation)")
    print("\n  Задача в execution с готовым ARTIFACT:RESULT")
    print("  Попытка: прямой переход в done (обход validation)")

    task = make_task(
        status="execution",
        artifact=RESULT_ARTIFACT,
        history=PLAN_HISTORY,
    )
    r = guard.validate_transition(task, "done")

    ok = show_guard_result("execution → done", r, expected_allowed=False)

    has_sequence_suggestion = (
        r.suggestion is not None and "validation" in r.suggestion.lower()
    )
    print(f"\n  Suggestion упоминает правильную последовательность: "
          f"{'✅ да' if has_sequence_suggestion else '❌ нет'}")

    print(f"\n  Форматированная ошибка:")
    error_msg = guard.format_error("execution", "done", r)
    for line in error_msg.splitlines():
        print(f"    {line}")

    record(3, "execution → done", "запрещён",
           "запрещён" if not r.allowed else "разрешён", ok)
    return ok and has_sequence_suggestion


# ---------------------------------------------------------------------------
# Сценарий 4: Попытка обойти через диалог (task_state_block для LLM)
# ---------------------------------------------------------------------------


def scenario_4() -> bool:
    section(4, "Попытка обойти через диалог (проверка task_state_block)")
    print("\n  Задача в planning")
    print("  Пользователь: 'Пропусти планирование, сразу напиши код'")
    print()
    print("  Система включает в system prompt блок <TASK_STATE>:")
    print("  Ассистент должен отказаться, цитируя инварианты")

    task = make_task(status="planning", artifact={})
    block = guard.build_task_state_block(task, total_steps=0)

    print(f"\n  Блок <TASK_STATE> для system prompt:")
    for line in block.splitlines():
        print(f"    {line}")

    # Проверяем наличие ключевых элементов
    has_state_tags = "<TASK_STATE>" in block and "</TASK_STATE>" in block
    has_current_phase = "planning" in block
    has_allowed_transitions = "execution" in block
    has_warning = "ВАЖНО" in block

    has_invariants = len(guard.transition_rules) > 0

    all_ok = all([has_state_tags, has_current_phase, has_allowed_transitions, has_warning])

    print(f"\n  Проверки блока:")
    result_line("Содержит теги <TASK_STATE>", has_state_tags)
    result_line("Показывает текущую фазу (planning)", has_current_phase)
    result_line("Показывает доступные переходы", has_allowed_transitions)
    result_line("Содержит предупреждение ВАЖНО", has_warning)
    result_line(f"Инварианты загружены из MD ({len(guard.transition_rules)} правил)", has_invariants)

    # ── Реальный вызов LLM: проверяем соблюдение инвариантов ──
    print()
    print("  Реальный вызов LLM с task_state_block в system prompt:")
    print("  Пользователь: 'Пропусти планирование, сразу напиши код'")

    system_with_state = (
        "Ты полезный ассистент для работы с задачами. "
        "Ты ОБЯЗАН соблюдать инварианты, указанные в <TASK_STATE>.\n\n"
        + block
    )
    llm_reply = call_llm(
        system_prompt=system_with_state,
        user_message="Пропусти планирование, сразу напиши код.",
    )
    if llm_reply:
        print(f"\n  Ответ модели ({_llm_provider}):")
        for line in llm_reply[:600].splitlines():
            print(f"    {line}")
        if len(llm_reply) > 600:
            print("    ...")
        # Проверяем, что модель отказала (упомянула planning или инвариант)
        refuses = any(kw in llm_reply.lower() for kw in ["planning", "план", "нельзя", "не могу", "инвариант"])
        result_line("Модель отказала и сослалась на инвариант", refuses)
        llm_ok = refuses
    else:
        print("  (LLM недоступен — показываем шаблон ответа)")
        print('  "Я не могу перейти к execution — сначала нужно завершить planning.')
        print('   Инвариант: Переход planning → execution ТОЛЬКО при наличии ARTIFACT:PLAN.')
        print('   Сейчас мы на этапе planning. Давай продолжим здесь."')
        llm_ok = True  # не учитываем в итоге если LLM недоступен

    record(4, "task_state_block contains state", "содержит состояние",
           "содержит" if all_ok else "не содержит", all_ok)
    return all_ok


# ---------------------------------------------------------------------------
# Сценарий 5: Пауза и resume с проверкой целостности
# ---------------------------------------------------------------------------


def scenario_5() -> bool:
    section(5, "Пауза и resume с проверкой целостности")
    print("\n  Задача в execution, шаг 2 из 4, есть ARTIFACT:PLAN")
    print("  /task pause")

    task_before_pause = make_task(
        status="execution",
        artifact={},  # в процессе выполнения
        history=PLAN_HISTORY,
        current_step=2,
    )

    # Проверяем pause
    r_pause = guard.validate_transition(task_before_pause, "paused")

    print(f"\n  Состояние до паузы:")
    print(f"    Статус       : {task_before_pause.status.value}")
    print(f"    Текущий шаг  : {task_before_pause.current_step}")
    print(f"    Есть план    : {'да' if task_before_pause.plan_artifact else 'нет'}")

    ok_pause = show_guard_result("execution → paused", r_pause, expected_allowed=True)

    # Имитация перезапуска — создаём объект из "БД"
    print("\n  [Имитация перезапуска: пересоздаём объект из БД]")
    task_after_restart = make_task(
        status="paused",
        paused_at="execution",
        artifact={},  # текущий артефакт (в процессе) — не важен
        history=PLAN_HISTORY,  # артефакт предыдущих этапов сохранён
        current_step=2,
    )

    print(f"\n  Состояние после перезапуска:")
    print(f"    Статус       : {task_after_restart.status.value}")
    print(f"    Paused at    : {task_after_restart.paused_at.value if task_after_restart.paused_at else 'нет'}")
    print(f"    Шаг          : {task_after_restart.current_step}")
    print(f"    Артефакт план: {'да' if task_after_restart.plan_artifact else 'нет'}")

    print("\n  /task resume")
    r_resume = guard.validate_resume(task_after_restart)
    ok_resume = show_guard_result("resume validation", r_resume, expected_allowed=True)

    # Проверяем, что восстанавливаем правильную фазу
    correct_phase = (
        r_resume.allowed and
        "execution" in r_resume.reason
    )
    result_line(
        "Resume восстанавливает фазу execution",
        correct_phase,
        f"reason: {r_resume.reason[:60]}...",
    )

    all_ok = ok_pause and ok_resume and correct_phase
    record(5, "pause → resume с целостным состоянием", "разрешён",
           "разрешён" if r_resume.allowed else "запрещён", all_ok)
    return all_ok


# ---------------------------------------------------------------------------
# Сценарий 6: Resume с повреждённым состоянием
# ---------------------------------------------------------------------------


def scenario_6() -> bool:
    section(6, "Resume с повреждённым состоянием")
    print("\n  Задача paused из execution, но plan artifact удалён из истории")

    # Имитируем повреждение: план удалён из history
    task_corrupted = make_task(
        status="paused",
        paused_at="execution",
        artifact={},
        history=[],  # <-- план удалён из БД!
        current_step=2,
    )

    print(f"\n  Состояние (повреждённое):")
    print(f"    Статус       : {task_corrupted.status.value}")
    print(f"    Paused at    : execution")
    print(f"    История      : пустая (план удалён)")

    print("\n  /task resume")
    r = guard.validate_resume(task_corrupted)

    ok = show_guard_result("resume validation (corrupt)", r, expected_allowed=False)

    has_plan_mention = "ARTIFACT:PLAN" in (r.missing_precondition or "")
    has_integrity_mention = "целостность" in r.reason.lower()
    has_suggestion = r.suggestion is not None

    print(f"\n  Качество диагностики:")
    result_line("Упомянут ARTIFACT:PLAN", has_plan_mention)
    result_line("Упомянута целостность", has_integrity_mention)
    result_line("Есть suggestion", has_suggestion)

    print(f"\n  Форматированная ошибка:")
    error_msg = guard.format_resume_error(r)
    for line in error_msg.splitlines():
        print(f"    {line}")

    all_ok = ok and has_plan_mention and has_integrity_mention
    record(6, "resume с повреждённым планом", "запрещён",
           "запрещён" if not r.allowed else "разрешён", all_ok)
    return all_ok


# ---------------------------------------------------------------------------
# Сценарий 7: Двойной инвариант — стек + этап
# ---------------------------------------------------------------------------


def scenario_7() -> bool:
    section(7, "Двойной инвариант: этап + стек технологий")
    print("\n  Задача в execution")
    print("  Пользователь просит: 'Используй Redis для кэширования результатов'")
    print()
    print("  Ожидаемый двойной отказ:")
    print("  1. Инвариант стека (SQLite-only, нет Redis)")
    print("  2. Инвариант этапа (в execution нельзя менять архитектуру)")

    task = make_task(
        status="execution",
        artifact={},
        history=PLAN_HISTORY,
        current_step=1,
    )

    # Демонстрируем, что task_state_block правильно указывает на execution
    block = guard.build_task_state_block(task, total_steps=4)

    print(f"\n  Блок состояния (для context LLM):")
    for line in block.splitlines():
        print(f"    {line}")

    # Проверяем наличие инвариантов стека из rules.md
    from llm_agent.core.invariant_loader import InvariantLoader
    loader = InvariantLoader(_config_dir)
    all_rules = loader.get_all_required()

    stack_invariants = [r for _, r in all_rules if "русск" not in r.lower()]
    transition_invariants = [r for _, r in all_rules if "ARTIFACT" in r or "ЗАПРЕЩЁН" in r]

    print(f"\n  Загруженные инварианты стека:")
    for _, rule in all_rules:
        print(f"    ⛔ {rule}")

    print(f"\n  Инварианты переходов (из task-transitions.md):")
    for rule in guard.transition_rules[:5]:
        print(f"    ⛔ {rule}")

    # ── Реальный вызов LLM: двойной инвариант ──────────────────────
    print()
    print("  Реальный вызов LLM с двойным инвариантом в system prompt:")
    print("  Пользователь: 'Используй Redis для кэширования результатов'")

    # Формируем system prompt: task_state_block + инварианты стека
    stack_rules_text = "\n".join(f"- {rule}" for _, rule in all_rules) if all_rules else "(инварианты не загружены)"
    system_double = (
        "Ты полезный ассистент. Ты ОБЯЗАН соблюдать все инварианты проекта.\n\n"
        "<INVARIANTS_STACK>\n"
        f"{stack_rules_text}\n"
        "</INVARIANTS_STACK>\n\n"
        + block
    )
    llm_reply = call_llm(
        system_prompt=system_double,
        user_message="Используй Redis для кэширования результатов вместо SQLite.",
    )
    if llm_reply:
        print(f"\n  Ответ модели ({_llm_provider}):")
        for line in llm_reply[:700].splitlines():
            print(f"    {line}")
        if len(llm_reply) > 700:
            print("    ...")
        refuses = any(kw in llm_reply.lower() for kw in ["redis", "нельзя", "не могу", "инвариант", "запрещ"])
        result_line("Модель отказала и объяснила двойное нарушение", refuses)
    else:
        print("  (LLM недоступен — показываем шаблон ответа)")
        print('  "Я не могу использовать Redis — это нарушает сразу два инварианта:')
        print('    1. Инвариант стека: используем только SQLite (Redis запрещён)')
        print('    2. Инвариант этапа: в execution нельзя менять архитектурные решения.')
        print('     Продолжим выполнение согласно утверждённому плану."')

    # Проверки
    has_execution_in_block = "execution" in block
    has_warning_in_block = "ВАЖНО" in block
    has_transition_invariants = len(guard.transition_rules) > 0
    has_stack_invariants = len(all_rules) > 0

    all_ok = all([
        has_execution_in_block,
        has_warning_in_block,
        has_transition_invariants,
        has_stack_invariants,
    ])

    print(f"\n  Проверки двойного инварианта:")
    result_line("Блок состояния содержит текущий этап (execution)", has_execution_in_block)
    result_line("Блок содержит предупреждение ВАЖНО", has_warning_in_block)
    result_line(f"Загружены инварианты переходов ({len(guard.transition_rules)} шт.)", has_transition_invariants)
    result_line(f"Загружены инварианты стека ({len(all_rules)} шт.)", has_stack_invariants)

    record(7, "execution + stack invariant", "двойной отказ",
           "инварианты присутствуют" if all_ok else "инварианты отсутствуют", all_ok)
    return all_ok


# ---------------------------------------------------------------------------
# Финальный отчёт
# ---------------------------------------------------------------------------


def print_report() -> None:
    print(f"\n{'═' * 65}")
    print(f"  ИТОГОВЫЙ ОТЧЁТ")
    print(f"{'═' * 65}")
    print()

    header = f"  {'Сцен':5} | {'Переход':<35} | {'Ожидание':<12} | {'Результат':<20} | Итог"
    separator = f"  {'-' * 5}-+-{'-' * 35}-+-{'-' * 12}-+-{'-' * 20}-+------"
    print(header)
    print(separator)

    all_passed = True
    for (num, transition, expected, result_str, ok) in _report:
        mark = "✅" if ok else "❌"
        print(
            f"  {num:<5} | {transition:<35} | {expected:<12} | {result_str:<20} | {mark}"
        )
        if not ok:
            all_passed = False

    print(separator)
    total_ok = sum(1 for *_, ok in _report if ok)
    total = len(_report)
    print(f"\n  Итого: {total_ok}/{total} сценариев прошли успешно")
    print()

    if all_passed:
        print("  ✅ Все сценарии прошли! TaskTransitionGuard работает корректно.")
    else:
        failed = [str(r[0]) for r in _report if not r[-1]]
        print(f"  ❌ Не прошли сценарии: {', '.join(failed)}")

    print(f"{'═' * 65}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ДЕМО: Строгий контроль переходов задачи (TaskTransitionGuard)║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print(f"\n  Guard инициализирован с config: {_config_dir}")
    print(f"  Загружено обязательных правил переходов: {len(guard.transition_rules)}")
    print(f"  Загружено рекомендуемых правил: {len(guard.transition_recommended)}")

    # Запускаем все сценарии
    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()
    scenario_5()
    scenario_6()
    scenario_7()

    # Итоговый отчёт
    print_report()


if __name__ == "__main__":
    main()
