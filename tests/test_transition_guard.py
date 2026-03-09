"""tests/test_transition_guard.py — тесты для TaskTransitionGuard.

Покрываем все 7 сценариев из демо-скрипта:
  1. Нормальный полный цикл (planning → execution → validation → done)
  2. Попытка перейти без артефакта (planning → execution без ARTIFACT:PLAN)
  3. Прямой переход execution → done (обход validation)
  4. Попытка обойти через диалог (проверка через task_state_block)
  5. Пауза и resume с проверкой целостности
  6. Resume с повреждённым состоянием (план удалён из истории)
  7. Проверка всех инвариантов графа переходов

Дополнительно:
  - Тесты TransitionResult dataclass
  - Тесты format_error / format_resume_error
  - Тесты build_task_state_block
  - Тесты загрузки инвариантов из MD-файла
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_agent.tasks.models import ExpectedAction, TaskState, TaskStatus
from llm_agent.tasks.transition_guard import (
    ALLOWED_TRANSITIONS,
    TransitionResult,
    TaskTransitionGuard,
    _has_plan_artifact,
    _has_result_artifact,
    _has_validation_artifact,
)


# ---------------------------------------------------------------------------
# Фикстуры и вспомогательные функции
# ---------------------------------------------------------------------------


def make_task(
    status: str = "planning",
    artifact: dict | None = None,
    history: list[dict] | None = None,
    paused_at: str | None = None,
    current_step: int = 0,
    title: str = "Тестовая задача",
) -> TaskState:
    """Создать TaskState для тестов без обращения к БД."""
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


PLAN_ARTIFACT = {"steps": [
    {"id": 1, "title": "Шаг 1", "description": "Описание шага 1"},
    {"id": 2, "title": "Шаг 2", "description": "Описание шага 2"},
]}

RESULT_ARTIFACT = {"summary": "Задача выполнена", "outputs": ["файл.py"]}

VALIDATION_ARTIFACT = {"passed": True, "issues": [], "recommendations": []}

PLAN_HISTORY = [{"phase": "planning", "artifact": PLAN_ARTIFACT, "completed_at": "2024-01-01"}]
RESULT_HISTORY = PLAN_HISTORY + [
    {"phase": "execution", "artifact": RESULT_ARTIFACT, "completed_at": "2024-01-02"}
]
FULL_HISTORY = RESULT_HISTORY + [
    {"phase": "validation", "artifact": VALIDATION_ARTIFACT, "completed_at": "2024-01-03"}
]


@pytest.fixture()
def guard() -> TaskTransitionGuard:
    return TaskTransitionGuard()


@pytest.fixture()
def guard_with_config(tmp_path: Path) -> TaskTransitionGuard:
    """Guard с загруженными инвариантами из временного MD-файла."""
    inv_dir = tmp_path / "config" / "invariants"
    inv_dir.mkdir(parents=True)
    md_file = inv_dir / "task-transitions.md"
    md_file.write_text(
        "# Инварианты переходов задачи\n\n"
        "## Обязательные\n"
        "- Переход planning → execution ТОЛЬКО при наличии ARTIFACT:PLAN\n"
        "- Прямой переход planning → done ЗАПРЕЩЁН\n\n"
        "## Рекомендуемые\n"
        "- В planning желательно минимум 2 turn-а\n",
        encoding="utf-8",
    )
    return TaskTransitionGuard(config_dir=tmp_path / "config")


# ---------------------------------------------------------------------------
# Тесты TransitionResult
# ---------------------------------------------------------------------------


class TestTransitionResult:
    def test_allowed_result(self) -> None:
        r = TransitionResult(allowed=True, reason="OK")
        assert r.allowed is True
        assert r.reason == "OK"
        assert r.missing_precondition is None
        assert r.suggestion is None

    def test_denied_result_with_details(self) -> None:
        r = TransitionResult(
            allowed=False,
            reason="Нет артефакта",
            missing_precondition="ARTIFACT:PLAN",
            suggestion="Сгенерируйте план",
        )
        assert r.allowed is False
        assert r.missing_precondition == "ARTIFACT:PLAN"
        assert r.suggestion == "Сгенерируйте план"


# ---------------------------------------------------------------------------
# Тесты вспомогательных функций проверки артефактов
# ---------------------------------------------------------------------------


class TestArtifactChecks:
    def test_no_plan_when_empty_artifact(self) -> None:
        task = make_task(status="planning", artifact={})
        assert _has_plan_artifact(task) is False

    def test_no_plan_when_steps_empty(self) -> None:
        task = make_task(status="planning", artifact={"steps": []})
        assert _has_plan_artifact(task) is False

    def test_has_plan_in_current_artifact(self) -> None:
        task = make_task(status="planning", artifact=PLAN_ARTIFACT)
        assert _has_plan_artifact(task) is True

    def test_has_plan_in_history(self) -> None:
        task = make_task(status="execution", history=PLAN_HISTORY)
        assert _has_plan_artifact(task) is True

    def test_no_result_when_empty(self) -> None:
        task = make_task(status="execution", artifact={})
        assert _has_result_artifact(task) is False

    def test_has_result_in_current_artifact(self) -> None:
        task = make_task(status="execution", artifact=RESULT_ARTIFACT)
        assert _has_result_artifact(task) is True

    def test_has_result_in_history(self) -> None:
        task = make_task(status="validation", history=RESULT_HISTORY)
        assert _has_result_artifact(task) is True

    def test_no_validation_when_empty(self) -> None:
        task = make_task(status="validation", artifact={})
        assert _has_validation_artifact(task) is False

    def test_no_validation_without_passed_key(self) -> None:
        task = make_task(status="validation", artifact={"issues": []})
        assert _has_validation_artifact(task) is False

    def test_has_validation_in_current_artifact(self) -> None:
        task = make_task(status="validation", artifact=VALIDATION_ARTIFACT)
        assert _has_validation_artifact(task) is True

    def test_has_validation_in_history(self) -> None:
        task = make_task(status="done", history=FULL_HISTORY)
        assert _has_validation_artifact(task) is True


# ---------------------------------------------------------------------------
# Сценарий 1: Нормальный полный цикл
# ---------------------------------------------------------------------------


class TestScenario1NormalCycle:
    """Нормальный полный цикл: planning → execution → validation → done."""

    def test_planning_to_execution_with_plan(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact=PLAN_ARTIFACT)
        result = guard.validate_transition(task, "execution")
        assert result.allowed is True
        assert "execution" in result.reason

    def test_execution_to_validation_with_result(self, guard: TaskTransitionGuard) -> None:
        task = make_task(
            status="execution",
            artifact=RESULT_ARTIFACT,
            history=PLAN_HISTORY,
        )
        result = guard.validate_transition(task, "validation")
        assert result.allowed is True

    def test_validation_to_done_with_validation(self, guard: TaskTransitionGuard) -> None:
        task = make_task(
            status="validation",
            artifact=VALIDATION_ARTIFACT,
            history=RESULT_HISTORY,
        )
        result = guard.validate_transition(task, "done")
        assert result.allowed is True

    def test_full_cycle_each_transition_allowed(self, guard: TaskTransitionGuard) -> None:
        """Все переходы в полном цикле должны быть разрешены."""
        # planning → execution
        t1 = make_task(status="planning", artifact=PLAN_ARTIFACT)
        r1 = guard.validate_transition(t1, "execution")
        assert r1.allowed, f"planning→execution запрещён: {r1.reason}"

        # execution → validation
        t2 = make_task(status="execution", artifact=RESULT_ARTIFACT, history=PLAN_HISTORY)
        r2 = guard.validate_transition(t2, "validation")
        assert r2.allowed, f"execution→validation запрещён: {r2.reason}"

        # validation → done
        t3 = make_task(
            status="validation", artifact=VALIDATION_ARTIFACT, history=RESULT_HISTORY
        )
        r3 = guard.validate_transition(t3, "done")
        assert r3.allowed, f"validation→done запрещён: {r3.reason}"


# ---------------------------------------------------------------------------
# Сценарий 2: Попытка перепрыгнуть этап без артефакта
# ---------------------------------------------------------------------------


class TestScenario2NoArtifact:
    """Попытка перехода без сформированного артефакта."""

    def test_planning_to_execution_without_plan_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact={})
        result = guard.validate_transition(task, "execution")
        assert result.allowed is False
        assert result.missing_precondition is not None
        assert "ARTIFACT:PLAN" in result.missing_precondition

    def test_planning_to_execution_empty_steps_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact={"steps": []})
        result = guard.validate_transition(task, "execution")
        assert result.allowed is False

    def test_execution_to_validation_without_result_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="execution", artifact={}, history=PLAN_HISTORY)
        result = guard.validate_transition(task, "validation")
        assert result.allowed is False
        assert result.missing_precondition is not None
        assert "ARTIFACT:RESULT" in result.missing_precondition

    def test_validation_to_done_without_validation_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="validation", artifact={}, history=RESULT_HISTORY)
        result = guard.validate_transition(task, "done")
        assert result.allowed is False
        assert result.missing_precondition is not None
        assert "ARTIFACT:VALIDATION" in result.missing_precondition

    def test_denied_result_has_suggestion(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact={})
        result = guard.validate_transition(task, "execution")
        assert result.suggestion is not None
        assert "planning" in result.suggestion.lower()


# ---------------------------------------------------------------------------
# Сценарий 3: Прямые запрещённые переходы
# ---------------------------------------------------------------------------


class TestScenario3ForbiddenDirect:
    """Прямые переходы в обход обязательной последовательности."""

    def test_planning_to_done_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact=PLAN_ARTIFACT)
        result = guard.validate_transition(task, "done")
        assert result.allowed is False
        assert "planning" in result.reason.lower()
        assert "done" in result.reason.lower()

    def test_planning_to_validation_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact=PLAN_ARTIFACT)
        result = guard.validate_transition(task, "validation")
        assert result.allowed is False
        assert "validation" in result.reason.lower()

    def test_execution_to_done_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(
            status="execution", artifact=RESULT_ARTIFACT, history=PLAN_HISTORY
        )
        result = guard.validate_transition(task, "done")
        assert result.allowed is False
        assert "done" in result.reason.lower()

    def test_execution_to_planning_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="execution", artifact=RESULT_ARTIFACT, history=PLAN_HISTORY)
        result = guard.validate_transition(task, "planning")
        assert result.allowed is False

    def test_validation_to_planning_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="validation", artifact=VALIDATION_ARTIFACT, history=RESULT_HISTORY)
        result = guard.validate_transition(task, "planning")
        assert result.allowed is False

    def test_validation_to_execution_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="validation", artifact=VALIDATION_ARTIFACT, history=RESULT_HISTORY)
        result = guard.validate_transition(task, "execution")
        assert result.allowed is False

    def test_done_to_anything_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="done", history=FULL_HISTORY)
        for target in ("planning", "execution", "validation"):
            result = guard.validate_transition(task, target)
            assert result.allowed is False, f"done → {target} должен быть запрещён"

    def test_forbidden_result_has_sequence_suggestion(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact=PLAN_ARTIFACT)
        result = guard.validate_transition(task, "done")
        assert result.suggestion is not None
        assert "planning" in result.suggestion.lower()
        assert "execution" in result.suggestion.lower()
        assert "validation" in result.suggestion.lower()


# ---------------------------------------------------------------------------
# Сценарий 4: Пауза
# ---------------------------------------------------------------------------


class TestScenario4Pause:
    """Проверка пауз из различных состояний."""

    def test_pause_from_planning_allowed(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning")
        result = guard.validate_transition(task, "paused")
        assert result.allowed is True

    def test_pause_from_execution_allowed(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="execution", history=PLAN_HISTORY)
        result = guard.validate_transition(task, "paused")
        assert result.allowed is True

    def test_pause_from_validation_allowed(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="validation", history=RESULT_HISTORY)
        result = guard.validate_transition(task, "paused")
        assert result.allowed is True

    def test_pause_from_done_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="done", history=FULL_HISTORY)
        result = guard.validate_transition(task, "paused")
        assert result.allowed is False

    def test_pause_from_paused_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="paused", paused_at="planning")
        result = guard.validate_transition(task, "paused")
        assert result.allowed is False


# ---------------------------------------------------------------------------
# Сценарий 5: Resume с нормальным состоянием
# ---------------------------------------------------------------------------


class TestScenario5ResumeNormal:
    """Resume задачи с целостным состоянием."""

    def test_resume_from_planning_pause_allowed(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="paused", paused_at="planning", artifact={})
        result = guard.validate_resume(task)
        assert result.allowed is True
        assert "planning" in result.reason

    def test_resume_from_execution_pause_with_plan_allowed(
        self, guard: TaskTransitionGuard
    ) -> None:
        task = make_task(
            status="paused",
            paused_at="execution",
            artifact=RESULT_ARTIFACT,
            history=PLAN_HISTORY,
        )
        result = guard.validate_resume(task)
        assert result.allowed is True
        assert "execution" in result.reason

    def test_resume_from_validation_pause_with_artifacts_allowed(
        self, guard: TaskTransitionGuard
    ) -> None:
        task = make_task(
            status="paused",
            paused_at="validation",
            artifact=VALIDATION_ARTIFACT,
            history=RESULT_HISTORY,
        )
        result = guard.validate_resume(task)
        assert result.allowed is True

    def test_resume_preserves_correct_state_info(self, guard: TaskTransitionGuard) -> None:
        """Resume возвращает фазу из paused_at, не из status."""
        task = make_task(status="paused", paused_at="execution", history=PLAN_HISTORY)
        result = guard.validate_resume(task)
        assert result.allowed is True
        assert "execution" in result.reason

    def test_resume_non_paused_task_denied(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning")
        result = guard.validate_resume(task)
        assert result.allowed is False
        assert "не на паузе" in result.reason

    def test_resume_without_paused_at_denied(self, guard: TaskTransitionGuard) -> None:
        """paused_at = None → повреждённое состояние."""
        task = make_task(status="paused", paused_at=None)
        result = guard.validate_resume(task)
        assert result.allowed is False
        assert "paused_at" in result.reason.lower()


# ---------------------------------------------------------------------------
# Сценарий 6: Resume с повреждённым состоянием
# ---------------------------------------------------------------------------


class TestScenario6CorruptedState:
    """Resume при повреждённом состоянии (удалённые артефакты)."""

    def test_resume_execution_without_plan_denied(self, guard: TaskTransitionGuard) -> None:
        """Задача на паузе в execution, но история пустая (план удалён)."""
        task = make_task(
            status="paused",
            paused_at="execution",
            artifact={},
            history=[],  # <-- план удалён
        )
        result = guard.validate_resume(task)
        assert result.allowed is False
        assert "ARTIFACT:PLAN" in (result.missing_precondition or "")
        assert "целостность" in result.reason.lower()

    def test_resume_execution_without_plan_has_suggestion(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="paused", paused_at="execution", history=[])
        result = guard.validate_resume(task)
        assert result.suggestion is not None
        assert "новую задачу" in result.suggestion.lower()

    def test_resume_validation_without_result_denied(self, guard: TaskTransitionGuard) -> None:
        """Задача на паузе в validation, артефакт result удалён."""
        task = make_task(
            status="paused",
            paused_at="validation",
            artifact={},
            history=PLAN_HISTORY,  # есть только план, result удалён
        )
        result = guard.validate_resume(task)
        assert result.allowed is False
        assert "ARTIFACT:RESULT" in (result.missing_precondition or "")

    def test_resume_planning_doesnt_require_plan(self, guard: TaskTransitionGuard) -> None:
        """Пауза из planning — план не обязателен (ещё не создан)."""
        task = make_task(
            status="paused",
            paused_at="planning",
            artifact={},
            history=[],  # пусто — это нормально для planning
        )
        result = guard.validate_resume(task)
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Сценарий 7: Граф переходов — полная проверка
# ---------------------------------------------------------------------------


class TestScenario7TransitionGraph:
    """Проверка всего графа допустимых переходов."""

    @pytest.mark.parametrize("from_status,to_status", [
        ("planning", "execution"),
        ("planning", "paused"),
        ("execution", "validation"),
        ("execution", "paused"),
        ("validation", "done"),
        ("validation", "paused"),
    ])
    def test_allowed_transition_in_graph(
        self,
        guard: TaskTransitionGuard,
        from_status: str,
        to_status: str,
    ) -> None:
        """Все переходы из ALLOWED_TRANSITIONS должны быть разрешены (при наличии артефактов)."""
        # Выбираем подходящий артефакт для каждого статуса
        artifact_map = {
            "planning":   PLAN_ARTIFACT,
            "execution":  RESULT_ARTIFACT,
            "validation": VALIDATION_ARTIFACT,
        }
        task = make_task(
            status=from_status,
            artifact=artifact_map.get(from_status, {}),
            history=RESULT_HISTORY if from_status == "validation" else PLAN_HISTORY,
        )
        result = guard.validate_transition(task, to_status)
        assert result.allowed is True, (
            f"Переход {from_status}→{to_status} должен быть разрешён, "
            f"но запрещён: {result.reason}"
        )

    @pytest.mark.parametrize("from_status,to_status", [
        ("planning", "done"),
        ("planning", "validation"),
        ("execution", "done"),
        ("execution", "planning"),
        ("validation", "planning"),
        ("validation", "execution"),
        ("done", "planning"),
        ("done", "execution"),
        ("done", "validation"),
        ("paused", "planning"),
        ("paused", "execution"),
    ])
    def test_forbidden_transition_in_graph(
        self,
        guard: TaskTransitionGuard,
        from_status: str,
        to_status: str,
    ) -> None:
        """Переходы вне ALLOWED_TRANSITIONS должны быть запрещены."""
        task = make_task(
            status=from_status,
            paused_at="planning" if from_status == "paused" else None,
            history=FULL_HISTORY,
            artifact=PLAN_ARTIFACT,
        )
        result = guard.validate_transition(task, to_status)
        assert result.allowed is False, (
            f"Переход {from_status}→{to_status} должен быть запрещён"
        )

    def test_paused_task_cannot_transition_directly(self, guard: TaskTransitionGuard) -> None:
        """Задача на паузе — нельзя перейти напрямую (нужен resume)."""
        task = make_task(status="paused", paused_at="execution", history=FULL_HISTORY)
        for target in ("planning", "execution", "validation", "done"):
            result = guard.validate_transition(task, target)
            assert result.allowed is False, (
                f"paused→{target} должен быть запрещён"
            )


# ---------------------------------------------------------------------------
# Тесты format_error
# ---------------------------------------------------------------------------


class TestFormatError:
    def test_error_format_contains_statuses(self, guard: TaskTransitionGuard) -> None:
        result = TransitionResult(
            allowed=False,
            reason="Артефакт не готов",
            missing_precondition="ARTIFACT:PLAN",
            suggestion="Сгенерируйте план",
        )
        msg = guard.format_error("planning", "execution", result)
        assert "planning" in msg
        assert "execution" in msg
        assert "⛔" in msg
        assert "ARTIFACT:PLAN" in msg
        assert "Сгенерируйте план" in msg

    def test_error_format_without_suggestion(self, guard: TaskTransitionGuard) -> None:
        result = TransitionResult(
            allowed=False,
            reason="Запрещено",
        )
        msg = guard.format_error("planning", "done", result)
        assert "└─" in msg  # последняя строка должна завершаться └─
        assert "⛔" in msg

    def test_error_format_tree_structure(self, guard: TaskTransitionGuard) -> None:
        """Проверяем структуру дерева ├─ / └─."""
        result = TransitionResult(
            allowed=False,
            reason="Нет артефакта",
            missing_precondition="ARTIFACT:PLAN",
            suggestion="Создайте план",
        )
        msg = guard.format_error("planning", "execution", result)
        lines = msg.splitlines()
        # Первая строка: ⛔
        assert "⛔" in lines[0]
        # Последняя строка: └─
        assert "└─" in lines[-1]


class TestFormatResumeError:
    def test_resume_error_format(self, guard: TaskTransitionGuard) -> None:
        result = TransitionResult(
            allowed=False,
            reason="Артефакт плана удалён",
            missing_precondition="ARTIFACT:PLAN",
            suggestion="Создайте новую задачу",
        )
        msg = guard.format_resume_error(result)
        assert "⛔" in msg
        assert "Resume" in msg
        assert "ARTIFACT:PLAN" in msg
        assert "Создайте новую задачу" in msg


# ---------------------------------------------------------------------------
# Тесты build_task_state_block
# ---------------------------------------------------------------------------


class TestBuildTaskStateBlock:
    def test_block_contains_task_title(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", title="Моя задача")
        block = guard.build_task_state_block(task)
        assert "Моя задача" in block

    def test_block_contains_status(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="execution", history=PLAN_HISTORY)
        block = guard.build_task_state_block(task)
        assert "execution" in block

    def test_block_contains_task_state_tags(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning")
        block = guard.build_task_state_block(task)
        assert "<TASK_STATE>" in block
        assert "</TASK_STATE>" in block

    def test_block_shows_available_transitions(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning")
        block = guard.build_task_state_block(task)
        assert "execution" in block
        assert "paused" in block

    def test_block_shows_artifacts_present(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="execution", artifact=RESULT_ARTIFACT, history=PLAN_HISTORY)
        block = guard.build_task_state_block(task)
        assert "PLAN" in block
        assert "RESULT" in block

    def test_block_shows_no_artifacts_when_empty(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="planning", artifact={})
        block = guard.build_task_state_block(task)
        assert "нет" in block.lower()

    def test_block_includes_step_info(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="execution", current_step=2)
        block = guard.build_task_state_block(task, total_steps=4)
        assert "2" in block
        assert "4" in block

    def test_block_contains_warning_about_stage(self, guard: TaskTransitionGuard) -> None:
        task = make_task(status="execution")
        block = guard.build_task_state_block(task)
        assert "ВАЖНО" in block

    def test_block_with_invariants(
        self, guard_with_config: TaskTransitionGuard
    ) -> None:
        """Guard с конфигом включает инварианты в блок."""
        task = make_task(status="planning")
        block = guard_with_config.build_task_state_block(task)
        assert "ARTIFACT:PLAN" in block or "инварианты" in block.lower()


# ---------------------------------------------------------------------------
# Тесты загрузки инвариантов из MD-файла
# ---------------------------------------------------------------------------


class TestInvariantLoading:
    def test_guard_without_config_has_no_rules(self) -> None:
        guard = TaskTransitionGuard()
        assert guard.transition_rules == []
        assert guard.transition_recommended == []

    def test_guard_loads_required_rules(
        self, guard_with_config: TaskTransitionGuard
    ) -> None:
        rules = guard_with_config.transition_rules
        assert len(rules) == 2
        assert any("ARTIFACT:PLAN" in r for r in rules)

    def test_guard_loads_recommended_rules(
        self, guard_with_config: TaskTransitionGuard
    ) -> None:
        rec = guard_with_config.transition_recommended
        assert len(rec) == 1
        assert any("turn" in r.lower() for r in rec)

    def test_guard_with_nonexistent_config(self) -> None:
        """Несуществующий config_dir → guard инициализируется без ошибок."""
        guard = TaskTransitionGuard(config_dir="/nonexistent/path")
        assert guard.transition_rules == []

    def test_guard_with_no_task_transitions_file(self, tmp_path: Path) -> None:
        """Нет файла task-transitions.md → guard инициализируется без ошибок."""
        inv_dir = tmp_path / "config" / "invariants"
        inv_dir.mkdir(parents=True)
        (inv_dir / "other.md").write_text("# Другое\n\n## Обязательные\n- Правило\n")
        guard = TaskTransitionGuard(config_dir=tmp_path / "config")
        assert guard.transition_rules == []


# ---------------------------------------------------------------------------
# Тесты ALLOWED_TRANSITIONS структуры данных
# ---------------------------------------------------------------------------


class TestAllowedTransitionsData:
    def test_planning_allows_execution_and_paused(self) -> None:
        assert "execution" in ALLOWED_TRANSITIONS["planning"]
        assert "paused" in ALLOWED_TRANSITIONS["planning"]

    def test_planning_does_not_allow_done(self) -> None:
        assert "done" not in ALLOWED_TRANSITIONS["planning"]

    def test_execution_allows_validation_and_paused(self) -> None:
        assert "validation" in ALLOWED_TRANSITIONS["execution"]
        assert "paused" in ALLOWED_TRANSITIONS["execution"]

    def test_validation_allows_done_and_paused(self) -> None:
        assert "done" in ALLOWED_TRANSITIONS["validation"]
        assert "paused" in ALLOWED_TRANSITIONS["validation"]

    def test_paused_allows_nothing_directly(self) -> None:
        assert ALLOWED_TRANSITIONS["paused"] == []

    def test_done_allows_nothing(self) -> None:
        assert ALLOWED_TRANSITIONS["done"] == []


# ---------------------------------------------------------------------------
# Интеграционные тесты: полный цикл через guard
# ---------------------------------------------------------------------------


class TestIntegrationFullCycle:
    """Проверяем полную последовательность состояний через guard."""

    def test_full_cycle_allowed(self, guard: TaskTransitionGuard) -> None:
        """Полный цикл planning→execution→validation→done должен проходить."""

        # planning → execution (с планом)
        task = make_task(status="planning", artifact=PLAN_ARTIFACT)
        r = guard.validate_transition(task, "execution")
        assert r.allowed, f"planning→execution: {r.reason}"

        # execution → validation (с результатом)
        task = make_task(status="execution", artifact=RESULT_ARTIFACT, history=PLAN_HISTORY)
        r = guard.validate_transition(task, "validation")
        assert r.allowed, f"execution→validation: {r.reason}"

        # validation → done (с валидацией)
        task = make_task(
            status="validation", artifact=VALIDATION_ARTIFACT, history=RESULT_HISTORY
        )
        r = guard.validate_transition(task, "done")
        assert r.allowed, f"validation→done: {r.reason}"

    def test_pause_and_resume_cycle(self, guard: TaskTransitionGuard) -> None:
        """Пауза и resume должны корректно обрабатываться."""

        # Пауза из execution
        task = make_task(status="execution", history=PLAN_HISTORY, current_step=2)
        r = guard.validate_transition(task, "paused")
        assert r.allowed, f"pause: {r.reason}"

        # Resume с сохранённым планом
        paused_task = make_task(
            status="paused",
            paused_at="execution",
            history=PLAN_HISTORY,
            current_step=2,
        )
        r = guard.validate_resume(paused_task)
        assert r.allowed, f"resume: {r.reason}"
        assert "execution" in r.reason

    def test_skip_phase_always_denied(self, guard: TaskTransitionGuard) -> None:
        """Перепрыгивание фаз всегда должно быть запрещено."""
        skips = [
            ("planning", "validation"),
            ("planning", "done"),
            ("execution", "done"),
        ]
        for from_s, to_s in skips:
            task = make_task(
                status=from_s,
                artifact=PLAN_ARTIFACT,
                history=RESULT_HISTORY,
            )
            r = guard.validate_transition(task, to_s)
            assert r.allowed is False, f"{from_s}→{to_s} должен быть запрещён"
