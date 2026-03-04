"""TaskStateMachine: чистая логика FSM-переходов между фазами задачи."""

from __future__ import annotations

from llm_agent.tasks.models import ExpectedAction, TaskStatus

# Допустимые переходы: from_status → list[to_status]
VALID_TRANSITIONS: dict[TaskStatus, list[TaskStatus]] = {
    TaskStatus.PLANNING: [TaskStatus.EXECUTION, TaskStatus.PAUSED],
    TaskStatus.EXECUTION: [TaskStatus.VALIDATION, TaskStatus.PAUSED],
    TaskStatus.VALIDATION: [TaskStatus.DONE, TaskStatus.PAUSED],
    TaskStatus.DONE: [],
    TaskStatus.PAUSED: [TaskStatus.PLANNING, TaskStatus.EXECUTION, TaskStatus.VALIDATION],
}

# Следующая фаза (для /task next)
NEXT_PHASE: dict[TaskStatus, TaskStatus] = {
    TaskStatus.PLANNING: TaskStatus.EXECUTION,
    TaskStatus.EXECUTION: TaskStatus.VALIDATION,
    TaskStatus.VALIDATION: TaskStatus.DONE,
}

# Ожидаемое действие пользователя при входе в фазу
PHASE_EXPECTED_ACTIONS: dict[TaskStatus, ExpectedAction] = {
    TaskStatus.PLANNING: ExpectedAction.APPROVE_PLAN,
    TaskStatus.EXECUTION: ExpectedAction.APPROVE_RESULT,
    TaskStatus.VALIDATION: ExpectedAction.APPROVE_VALIDATION,
    TaskStatus.DONE: ExpectedAction.NONE,
    TaskStatus.PAUSED: ExpectedAction.NONE,
}


class TaskStateMachine:
    """Чистый конечный автомат для фазовых переходов задачи.

    Валидирует переходы и проверяет требования к артефактам.
    Не взаимодействует с LLM и хранилищем напрямую.
    """

    @staticmethod
    def can_transition(from_status: TaskStatus, to_status: TaskStatus) -> bool:
        """Проверить, допустим ли переход."""
        return to_status in VALID_TRANSITIONS.get(from_status, [])

    @staticmethod
    def validate_transition(from_status: TaskStatus, to_status: TaskStatus) -> None:
        """Валидировать переход; бросает ValueError если недопустим."""
        if not TaskStateMachine.can_transition(from_status, to_status):
            allowed = [s.value for s in VALID_TRANSITIONS.get(from_status, [])]
            raise ValueError(
                f"Невозможен переход {from_status.value} -> {to_status.value}. "
                f"Допустимые: {allowed}"
            )

    @staticmethod
    def validate_artifact_for_next(status: TaskStatus, artifact: dict) -> None:
        """Проверить, что артефакт текущей фазы достаточен для перехода.

        Raises:
            ValueError: если артефакт отсутствует или неполон.
        """
        if status == TaskStatus.PLANNING:
            if not artifact or "steps" not in artifact:
                raise ValueError(
                    "План ещё не сформирован. Продолжайте диалог, пока LLM "
                    "не выдаст [ARTIFACT:PLAN] с JSON-планом."
                )
            if not artifact["steps"]:
                raise ValueError("План пуст — нет шагов для выполнения.")

        elif status == TaskStatus.EXECUTION:
            if not artifact:
                raise ValueError(
                    "Результат выполнения ещё не получен. Продолжайте диалог, "
                    "пока LLM не выдаст [ARTIFACT:RESULT]."
                )

        elif status == TaskStatus.VALIDATION:
            if not artifact or "passed" not in artifact:
                raise ValueError(
                    "Валидация ещё не завершена. Продолжайте диалог, "
                    "пока LLM не выдаст [ARTIFACT:VALIDATION]."
                )

    @staticmethod
    def get_next_phase(current: TaskStatus) -> TaskStatus:
        """Получить следующую фазу для /task next."""
        if current not in NEXT_PHASE:
            if current == TaskStatus.DONE:
                raise ValueError("Задача уже завершена.")
            raise ValueError(
                f"Нет следующей фазы для статуса '{current.value}'. "
                "Снимите задачу с паузы командой /task resume."
            )
        return NEXT_PHASE[current]

    @staticmethod
    def get_expected_action(status: TaskStatus) -> ExpectedAction:
        """Получить ожидаемое действие пользователя для фазы."""
        return PHASE_EXPECTED_ACTIONS.get(status, ExpectedAction.NONE)
