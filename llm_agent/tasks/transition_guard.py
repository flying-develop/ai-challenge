"""TaskTransitionGuard: строгий программный контроль переходов между фазами задачи.

Уровень 1 (программный) — этот модуль:
  Класс TaskTransitionGuard запрещает невалидные переходы на уровне кода.
  Никакой prompt-инженерией не обойти.

Уровень 2 (prompt) — InvariantLoader:
  Правила из config/invariants/task-transitions.md автоматически загружаются
  InvariantLoader и включаются в <INVARIANTS> блок system prompt.

Граф переходов (ALLOWED_TRANSITIONS) — hardcoded в коде как первичная защита.
MD-файл дублирует правила для system prompt (объяснение для LLM и пользователя).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from llm_agent.tasks.models import TaskState, TaskStatus

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Результат проверки перехода
# ---------------------------------------------------------------------------


@dataclass
class TransitionResult:
    """Результат проверки допустимости перехода.

    Attributes:
        allowed: True если переход разрешён.
        reason: Объяснение решения (почему разрешён или запрещён).
        missing_precondition: Чего не хватает для перехода (если запрещён).
        suggestion: Что нужно сделать для разблокировки.
    """

    allowed: bool
    reason: str
    missing_precondition: str | None = None
    suggestion: str | None = None


# ---------------------------------------------------------------------------
# Граф переходов (программная защита — первична)
# ---------------------------------------------------------------------------

# Допустимые переходы из каждого состояния
ALLOWED_TRANSITIONS: dict[str, list[str]] = {
    "planning":   ["execution", "paused"],
    "execution":  ["validation", "paused"],
    "validation": ["done", "paused"],
    "paused":     [],   # восстанавливается в paused_at через resume, не напрямую
    "done":       [],
}

# Явно запрещённые прямые переходы — для понятных сообщений об ошибках
_FORBIDDEN_DIRECT: dict[tuple[str, str], str] = {
    ("planning", "validation"): "нельзя валидировать без выполнения",
    ("planning", "done"):       "нельзя завершить без выполнения и валидации",
    ("execution", "done"):      "нельзя завершить без валидации",
    ("execution", "planning"):  "обратный переход запрещён",
    ("validation", "planning"): "обратный переход запрещён",
    ("validation", "execution"):"обратный переход запрещён",
    ("done", "planning"):       "задача уже завершена",
    ("done", "execution"):      "задача уже завершена",
    ("done", "validation"):     "задача уже завершена",
}


# ---------------------------------------------------------------------------
# Проверка артефактов
# ---------------------------------------------------------------------------


def _has_plan_artifact(task: TaskState) -> bool:
    """Проверить наличие корректного артефакта плана."""
    artifact = task.plan_artifact
    return bool(artifact) and "steps" in artifact and bool(artifact["steps"])


def _has_result_artifact(task: TaskState) -> bool:
    """Проверить наличие артефакта результата выполнения."""
    return bool(task.result_artifact)


def _has_validation_artifact(task: TaskState) -> bool:
    """Проверить наличие артефакта валидации."""
    # Текущий артефакт (если в фазе validation)
    if task.status == TaskStatus.VALIDATION and task.artifact and "passed" in task.artifact:
        return True
    # В истории завершённых фаз
    for entry in task.history:
        if entry.get("phase") == "validation":
            a = entry.get("artifact", {})
            return bool(a) and "passed" in a
    return False


# ---------------------------------------------------------------------------
# Предусловия для переходов
# ---------------------------------------------------------------------------

# Тип: (TaskState) -> (ok: bool, missing: str | None, suggestion: str | None)
_PrecondFn = Callable[[TaskState], tuple[bool, "str | None", "str | None"]]


def _check_planning_to_execution(task: TaskState) -> tuple[bool, str | None, str | None]:
    if not _has_plan_artifact(task):
        return (
            False,
            "Артефакт плана не сформирован (ARTIFACT:PLAN отсутствует)",
            "Продолжите диалог в фазе planning, пока LLM не сгенерирует "
            "[ARTIFACT:PLAN] с JSON-планом",
        )
    return (True, None, None)


def _check_execution_to_validation(task: TaskState) -> tuple[bool, str | None, str | None]:
    if not _has_result_artifact(task):
        return (
            False,
            "Артефакт результата выполнения не сформирован (ARTIFACT:RESULT отсутствует)",
            "Продолжите диалог в фазе execution, пока LLM не сгенерирует [ARTIFACT:RESULT]",
        )
    return (True, None, None)


def _check_validation_to_done(task: TaskState) -> tuple[bool, str | None, str | None]:
    if not _has_validation_artifact(task):
        return (
            False,
            "Артефакт валидации не сформирован (ARTIFACT:VALIDATION отсутствует)",
            "Продолжите диалог в фазе validation, пока LLM не сгенерирует [ARTIFACT:VALIDATION]",
        )
    return (True, None, None)


PRECONDITIONS: dict[tuple[str, str], _PrecondFn] = {
    ("planning", "execution"):   _check_planning_to_execution,
    ("execution", "validation"): _check_execution_to_validation,
    ("validation", "done"):      _check_validation_to_done,
}


# ---------------------------------------------------------------------------
# TaskTransitionGuard
# ---------------------------------------------------------------------------


class TaskTransitionGuard:
    """Строгий контроль переходов между фазами задачи.

    Уровень 1 (программный): запрещает невалидные переходы на уровне кода.
    Уровень 2 (prompt): правила из MD-файла включаются в system prompt через
                        InvariantLoader — этот класс их читает для задачи
                        build_task_state_block().

    Не зависит от LLM — все проверки синхронные, на уровне данных.

    Args:
        config_dir: Путь к директории config/ для загрузки инвариантов.
                    Если None — используются только программные правила.
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self._transition_rules: list[str] = []
        self._transition_recommended: list[str] = []
        if config_dir is not None:
            self._load_invariants(Path(config_dir))

    def _load_invariants(self, config_dir: Path) -> None:
        """Загрузить правила переходов из task-transitions.md через InvariantLoader."""
        try:
            from llm_agent.core.invariant_loader import InvariantLoader
            loader = InvariantLoader(config_dir)
            for cat in loader.categories:
                if cat.name == "task-transitions":
                    self._transition_rules = list(cat.required)
                    self._transition_recommended = list(cat.recommended)
                    break
        except Exception:
            pass

    @property
    def transition_rules(self) -> list[str]:
        """Обязательные правила переходов из MD-файла (для system prompt)."""
        return list(self._transition_rules)

    @property
    def transition_recommended(self) -> list[str]:
        """Рекомендуемые правила из MD-файла."""
        return list(self._transition_recommended)

    # ------------------------------------------------------------------
    # Основные методы валидации
    # ------------------------------------------------------------------

    def validate_transition(self, task: TaskState, target_status: str) -> TransitionResult:
        """Проверить допустимость перехода из текущего состояния задачи.

        Args:
            task: Текущее состояние задачи.
            target_status: Целевой статус (строка, например "execution").

        Returns:
            TransitionResult с полями allowed, reason, missing_precondition, suggestion.
        """
        current = task.status.value

        # Пауза доступна из любого активного состояния
        if target_status == "paused":
            if current in ("planning", "execution", "validation"):
                return TransitionResult(
                    allowed=True,
                    reason=f"Пауза доступна из любого активного состояния",
                )
            return TransitionResult(
                allowed=False,
                reason=f"Нельзя поставить на паузу задачу в статусе '{current}'",
                suggestion="Пауза доступна из: planning, execution, validation",
            )

        # Проверяем явно запрещённые прямые переходы
        forbidden_reason = _FORBIDDEN_DIRECT.get((current, target_status))
        if forbidden_reason:
            return TransitionResult(
                allowed=False,
                reason=f"Прямой переход {current} → {target_status} запрещён: {forbidden_reason}",
                suggestion="Правильная последовательность: planning → execution → validation → done",
            )

        # Проверяем граф допустимых переходов
        allowed = ALLOWED_TRANSITIONS.get(current, [])
        if target_status not in allowed:
            if current == "paused":
                return TransitionResult(
                    allowed=False,
                    reason="Задача на паузе — нельзя переходить напрямую",
                    suggestion="Сначала выполните /task resume для восстановления состояния",
                )
            if current == "done":
                return TransitionResult(
                    allowed=False,
                    reason="Задача уже завершена — новые переходы недоступны",
                    suggestion="Создайте новую задачу: /task new",
                )
            return TransitionResult(
                allowed=False,
                reason=f"Переход {current} → {target_status} не предусмотрен",
                suggestion=f"Из '{current}' доступны переходы: {allowed}",
            )

        # Проверяем предусловия (наличие артефактов)
        precond_fn = PRECONDITIONS.get((current, target_status))
        if precond_fn is not None:
            ok, missing, suggestion = precond_fn(task)
            if not ok:
                return TransitionResult(
                    allowed=False,
                    reason=missing or "Предусловие перехода не выполнено",
                    missing_precondition=missing,
                    suggestion=suggestion,
                )

        return TransitionResult(
            allowed=True,
            reason=f"Все предусловия выполнены — переход {current} → {target_status} разрешён",
        )

    def validate_resume(self, task: TaskState) -> TransitionResult:
        """Проверить целостность состояния задачи при resume.

        Args:
            task: Задача в статусе PAUSED.

        Returns:
            TransitionResult — allowed=True если можно делать resume.
        """
        if task.status != TaskStatus.PAUSED:
            return TransitionResult(
                allowed=False,
                reason=f"Задача не на паузе (статус: {task.status.value})",
                suggestion="Команда resume работает только для задач со статусом 'paused'",
            )

        if task.paused_at is None:
            return TransitionResult(
                allowed=False,
                reason="Не удалось определить фазу при паузе (paused_at отсутствует)",
                missing_precondition="paused_at",
                suggestion="Состояние задачи повреждено — создайте новую задачу",
            )

        paused_at = task.paused_at.value

        # Если пауза была в execution или позже — план обязан быть в истории
        if paused_at in ("execution", "validation", "done"):
            if not _has_plan_artifact(task):
                return TransitionResult(
                    allowed=False,
                    reason=(
                        "Артефакт плана отсутствует — целостность состояния нарушена. "
                        f"Задача была на паузе в фазе '{paused_at}', "
                        "но артефакт planning в истории не найден"
                    ),
                    missing_precondition="ARTIFACT:PLAN в истории фазы planning",
                    suggestion=(
                        "Артефакт плана был удалён из базы данных. "
                        "Целостность нарушена — рекомендуется создать новую задачу: /task new"
                    ),
                )

        # Если пауза была в validation или позже — результат обязан быть в истории
        if paused_at in ("validation", "done"):
            if not _has_result_artifact(task):
                return TransitionResult(
                    allowed=False,
                    reason=(
                        "Артефакт результата выполнения отсутствует — целостность нарушена. "
                        f"Задача была на паузе в фазе '{paused_at}', "
                        "но артефакт execution в истории не найден"
                    ),
                    missing_precondition="ARTIFACT:RESULT в истории фазы execution",
                    suggestion=(
                        "Артефакт результата был удалён из базы данных. "
                        "Целостность нарушена — рекомендуется создать новую задачу: /task new"
                    ),
                )

        return TransitionResult(
            allowed=True,
            reason=f"Состояние целостно — resume в фазу '{paused_at}' разрешён",
        )

    # ------------------------------------------------------------------
    # Форматирование ошибок
    # ------------------------------------------------------------------

    def format_error(
        self,
        from_status: str,
        to_status: str,
        result: TransitionResult,
    ) -> str:
        """Форматировать ошибку перехода в читаемый вид.

        Формат:
          ⛔ Переход {from} → {to} запрещён
          ├─ Причина: ...
          ├─ Текущее состояние: ...
          ├─ Отсутствует: ... (если есть)
          └─ Что делать: ... (если есть)
        """
        lines = [
            f"⛔ Переход {from_status} → {to_status} запрещён",
            f"├─ Причина: {result.reason}",
            f"├─ Текущее состояние: {from_status}",
        ]
        if result.missing_precondition:
            lines.append(f"├─ Отсутствует: {result.missing_precondition}")
        if result.suggestion:
            lines.append(f"└─ Что делать: {result.suggestion}")
        else:
            # Последняя строка должна завершаться └─
            lines[-1] = lines[-1].replace("├─", "└─", 1)
        return "\n".join(lines)

    def format_resume_error(self, result: TransitionResult) -> str:
        """Форматировать ошибку resume в читаемый вид."""
        lines = [
            "⛔ Resume невозможен — нарушена целостность состояния",
            f"├─ Причина: {result.reason}",
        ]
        if result.missing_precondition:
            lines.append(f"├─ Отсутствует: {result.missing_precondition}")
        if result.suggestion:
            lines.append(f"└─ Что делать: {result.suggestion}")
        else:
            lines[-1] = lines[-1].replace("├─", "└─", 1)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Блок состояния для system prompt
    # ------------------------------------------------------------------

    def build_task_state_block(
        self,
        task: TaskState,
        total_steps: int = 0,
    ) -> str:
        """Построить блок <TASK_STATE> для включения в system prompt LLM.

        Блок информирует LLM о текущей фазе, доступных переходах и артефактах,
        чтобы ассистент понимал ограничения и объяснял их пользователю.

        Args:
            task: Текущее состояние задачи.
            total_steps: Общее количество шагов в плане.

        Returns:
            Строка с блоком <TASK_STATE>.
        """
        status = task.status.value
        available = ALLOWED_TRANSITIONS.get(status, [])

        # Собираем список присутствующих артефактов
        artifacts_present: list[str] = []
        if _has_plan_artifact(task):
            artifacts_present.append("PLAN")
        if _has_result_artifact(task):
            artifacts_present.append("RESULT")
        if _has_validation_artifact(task):
            artifacts_present.append("VALIDATION")

        artifact_str = ", ".join(artifacts_present) if artifacts_present else "нет"
        transitions_str = ", ".join(available) if available else "нет доступных переходов"
        step_str = (
            f"{task.current_step} из {total_steps}"
            if total_steps else str(task.current_step)
        )

        expected_map = {
            "planning":   "Составь структурированный план [ARTIFACT:PLAN]",
            "execution":  "Выполни шаги плана и выдай итог [ARTIFACT:RESULT]",
            "validation": "Проверь результат на соответствие плану [ARTIFACT:VALIDATION]",
            "done":       "Задача завершена, дай резюме",
            "paused":     "Ожидание resume",
        }

        lines = [
            "<TASK_STATE>",
            f"Текущая задача: {task.title}",
            f"Текущий этап: {status}",
            f"Текущий шаг: {step_str}",
            f"Ожидаемое действие: {expected_map.get(status, status)}",
            f"Доступные переходы: {transitions_str}",
            f"Артефакты: {artifact_str}",
            "",
            f"ВАЖНО: Ты находишься в этапе {status}. Не выходи за рамки этого этапа.",
        ]

        # Добавляем ключевые инварианты из MD-файла как напоминание
        if self._transition_rules:
            lines.append("")
            lines.append("Инварианты переходов (обязательны к соблюдению):")
            for rule in self._transition_rules[:5]:
                lines.append(f"  - {rule}")

        lines.append("</TASK_STATE>")
        return "\n".join(lines)
