"""Модели данных для Task Orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    """Фаза задачи в жизненном цикле FSM."""

    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"
    PAUSED = "paused"


class ExpectedAction(Enum):
    """Что оркестратор ожидает от пользователя."""

    APPROVE_PLAN = "approve_plan"
    CONFIRM_STEP = "confirm_step"
    APPROVE_RESULT = "approve_result"
    APPROVE_VALIDATION = "approve_validation"
    NONE = "none"


@dataclass
class PlanStep:
    """Один шаг плана задачи."""

    id: int
    title: str
    description: str


@dataclass
class TaskState:
    """Полное персистентное состояние задачи."""

    id: int
    title: str
    status: TaskStatus
    paused_at: TaskStatus | None
    current_step: int
    expected_action: ExpectedAction
    artifact: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @property
    def plan_steps(self) -> list[PlanStep]:
        """Извлечь шаги плана из артефакта или истории."""
        if self.status == TaskStatus.PLANNING and "steps" in self.artifact:
            return [PlanStep(**s) for s in self.artifact["steps"]]
        for entry in self.history:
            if entry.get("phase") == "planning" and "steps" in entry.get("artifact", {}):
                return [PlanStep(**s) for s in entry["artifact"]["steps"]]
        return []

    @property
    def plan_artifact(self) -> dict:
        """Получить артефакт фазы планирования из истории."""
        for entry in self.history:
            if entry.get("phase") == "planning":
                return entry.get("artifact", {})
        if self.status == TaskStatus.PLANNING:
            return self.artifact
        return {}

    @property
    def result_artifact(self) -> dict:
        """Получить артефакт фазы выполнения из истории."""
        for entry in self.history:
            if entry.get("phase") == "execution":
                return entry.get("artifact", {})
        if self.status == TaskStatus.EXECUTION:
            return self.artifact
        return {}
