"""System prompt шаблоны для каждой фазы задачи."""

from __future__ import annotations

import json


def planning_prompt(title: str) -> str:
    """System prompt для фазы PLANNING."""
    return (
        f'Ты помогаешь спланировать задачу: "{title}".\n\n'
        "Твоя роль — составить структурированный план из пронумерованных шагов.\n"
        "Обсуди с пользователем детали, уточни требования.\n"
        "Когда план готов, выведи его в формате:\n\n"
        "[ARTIFACT:PLAN]\n"
        "```json\n"
        '{"steps": [{"id": 1, "title": "...", "description": "..."}, ...]}\n'
        "```\n\n"
        "НЕ выполняй задачу — только планируй."
    )


def execution_prompt(title: str, plan_artifact: dict, current_step: int) -> str:
    """System prompt для фазы EXECUTION."""
    plan_json = json.dumps(plan_artifact, ensure_ascii=False, indent=2)
    return (
        f'Ты выполняешь задачу: "{title}".\n\n'
        f"План:\n{plan_json}\n\n"
        f"Текущий шаг: {current_step}\n\n"
        "Выполняй шаги последовательно. После завершения каждого шага выведи:\n"
        "[STEP_DONE:{step_id}]\n\n"
        "Когда все шаги выполнены, выведи итоговый результат:\n\n"
        "[ARTIFACT:RESULT]\n"
        "```json\n"
        '{"summary": "...", "outputs": [...]}\n'
        "```"
    )


def validation_prompt(title: str, plan_artifact: dict, result_artifact: dict) -> str:
    """System prompt для фазы VALIDATION."""
    plan_json = json.dumps(plan_artifact, ensure_ascii=False, indent=2)
    result_json = json.dumps(result_artifact, ensure_ascii=False, indent=2)
    return (
        f'Ты валидируешь результат выполнения задачи: "{title}".\n\n'
        f"План:\n{plan_json}\n\n"
        f"Результат:\n{result_json}\n\n"
        "Проверь соответствие результата плану, найди проблемы, оцени качество.\n"
        "Завершив анализ, выведи:\n\n"
        "[ARTIFACT:VALIDATION]\n"
        "```json\n"
        '{"passed": true, "issues": ["..."], "recommendations": ["..."]}\n'
        "```"
    )


def done_prompt(title: str) -> str:
    """System prompt для фазы DONE."""
    return (
        f'Задача "{title}" завершена.\n'
        "Дай краткое резюме: что было сделано, какие решения приняты, "
        "что стоит запомнить."
    )


def resume_prompt(title: str, phase: str, artifact_summary: str) -> str:
    """Дополнение к system prompt при возобновлении задачи."""
    return (
        "\n\n--- ВОЗОБНОВЛЕНИЕ ЗАДАЧИ ---\n"
        f'Задача: "{title}"\n'
        f"Фаза: {phase}\n"
        f"Контекст: {artifact_summary}\n"
        "Продолжай с того места, где остановились. "
        "Не повторяй то, что уже было сделано."
    )
