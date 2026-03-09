"""TaskOrchestrator: управление жизненным циклом задачи, LLM-взаимодействием и персистенцией."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from llm_agent.tasks.models import ExpectedAction, TaskState, TaskStatus
from llm_agent.tasks.state_machine import TaskStateMachine
from llm_agent.tasks.transition_guard import TaskTransitionGuard
from llm_agent.tasks import prompts as task_prompts

if TYPE_CHECKING:
    from llm_agent.application.strategy_agent import StrategyAgent
    from llm_agent.memory.manager import MemoryManager


class TaskOrchestrator:
    """Управляет жизненным циклом задачи, LLM-взаимодействием и персистенцией.

    Args:
        db_path: Путь к SQLite-файлу (тот же memory.db).
        agent: Экземпляр StrategyAgent для LLM-вызовов.
        memory_manager: MemoryManager для интеграции working memory при resume.
    """

    def __init__(
        self,
        db_path: str | Path,
        agent: "StrategyAgent",
        memory_manager: "MemoryManager | None" = None,
        config_dir: str | Path | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._agent = agent
        self._memory_manager = memory_manager
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._setup_schema()

        self._active_task: TaskState | None = None
        self._original_system_prompt: str | None = None

        self._guard = TaskTransitionGuard(config_dir)

    # ------------------------------------------------------------------
    # Схема
    # ------------------------------------------------------------------

    def _setup_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS task_state (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'planning',
                paused_at       TEXT,
                current_step    INTEGER NOT NULL DEFAULT 0,
                expected_action TEXT    NOT NULL DEFAULT 'approve_plan',
                artifact        TEXT    NOT NULL DEFAULT '{}',
                history         TEXT    NOT NULL DEFAULT '[]',
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_task_state_status
                ON task_state (status);
        """)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_task(self, title: str) -> TaskState:
        """Создать новую задачу и войти в фазу PLANNING."""
        if self._active_task is not None:
            raise ValueError(
                f"Уже есть активная задача #{self._active_task.id}: "
                f'"{self._active_task.title}". '
                "Используйте /task pause, чтобы приостановить текущую."
            )

        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        cur = self._conn.execute(
            """INSERT INTO task_state (title, status, expected_action, created_at, updated_at)
               VALUES (?, 'planning', 'approve_plan', ?, ?)""",
            (title, now, now),
        )
        self._conn.commit()
        task_id = cur.lastrowid

        task = self.get_task(task_id)
        assert task is not None
        self._active_task = task

        # Подменяем system prompt на planning
        self._override_system_prompt(self._build_phase_prompt(task))
        self._agent.clear_history()

        return task

    def get_task(self, task_id: int) -> TaskState | None:
        """Загрузить задачу по ID из SQLite."""
        row = self._conn.execute(
            "SELECT * FROM task_state WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(self) -> list[TaskState]:
        """Список всех задач."""
        rows = self._conn.execute(
            "SELECT * FROM task_state ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def _save_task(self, task: TaskState) -> None:
        """Сохранить состояние задачи в SQLite."""
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        task.updated_at = now
        self._conn.execute(
            """UPDATE task_state
               SET status = ?, paused_at = ?, current_step = ?,
                   expected_action = ?, artifact = ?, history = ?,
                   updated_at = ?
               WHERE id = ?""",
            (
                task.status.value,
                task.paused_at.value if task.paused_at else None,
                task.current_step,
                task.expected_action.value,
                json.dumps(task.artifact, ensure_ascii=False),
                json.dumps(task.history, ensure_ascii=False),
                now,
                task.id,
            ),
        )
        self._conn.commit()

    def _row_to_task(self, row: sqlite3.Row) -> TaskState:
        """Конвертировать строку SQLite в TaskState."""
        paused_at_val = row["paused_at"]
        paused_at = TaskStatus(paused_at_val) if paused_at_val else None

        return TaskState(
            id=row["id"],
            title=row["title"],
            status=TaskStatus(row["status"]),
            paused_at=paused_at,
            current_step=row["current_step"],
            expected_action=ExpectedAction(row["expected_action"]),
            artifact=json.loads(row["artifact"]),
            history=json.loads(row["history"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Активная задача
    # ------------------------------------------------------------------

    @property
    def active_task(self) -> TaskState | None:
        return self._active_task

    @property
    def has_active_task(self) -> bool:
        return self._active_task is not None

    def load_task(self, task_id: int) -> TaskState:
        """Загрузить и активировать задачу по ID."""
        if self._active_task is not None:
            raise ValueError(
                f"Уже есть активная задача #{self._active_task.id}. "
                "Сначала приостановите её: /task pause"
            )
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Задача #{task_id} не найдена.")

        self._active_task = task

        if task.status != TaskStatus.PAUSED and task.status != TaskStatus.DONE:
            self._override_system_prompt(self._build_phase_prompt(task))
            self._agent.clear_history()
        elif task.status == TaskStatus.PAUSED:
            # Для paused задач — не подменяем prompt до /task resume
            pass

        return task

    # ------------------------------------------------------------------
    # Основные операции
    # ------------------------------------------------------------------

    def handle_message(self, user_input: str) -> str:
        """Обработать сообщение пользователя в контексте активной задачи.

        1. Подменить system prompt на фазовый
        2. Вызвать agent.ask()
        3. Распарсить ответ на маркеры артефактов
        4. Обновить состояние если артефакты найдены
        5. Вернуть ответ LLM (без изменений)
        """
        task = self._active_task
        if task is None:
            raise ValueError('Нет активной задачи. Используйте /task new "название".')

        if task.status == TaskStatus.PAUSED:
            raise ValueError("Задача на паузе. Используйте /task resume для продолжения.")

        if task.status == TaskStatus.DONE:
            raise ValueError("Задача завершена. Создайте новую: /task new")

        # Убеждаемся, что prompt актуален
        self._override_system_prompt(self._build_phase_prompt(task))

        # Вызов LLM
        response = self._agent.ask(user_input)

        # Парсинг ответа в зависимости от фазы
        if task.status == TaskStatus.PLANNING:
            artifact = self._parse_artifact(response, "PLAN")
            if artifact:
                task.artifact = artifact
                task.expected_action = ExpectedAction.APPROVE_PLAN
                self._save_task(task)

        elif task.status == TaskStatus.EXECUTION:
            steps_done = self._parse_step_done(response)
            if steps_done:
                task.current_step = max(steps_done)
                self._save_task(task)
            artifact = self._parse_artifact(response, "RESULT")
            if artifact:
                task.artifact = artifact
                task.expected_action = ExpectedAction.APPROVE_RESULT
                self._save_task(task)

        elif task.status == TaskStatus.VALIDATION:
            artifact = self._parse_artifact(response, "VALIDATION")
            if artifact:
                task.artifact = artifact
                task.expected_action = ExpectedAction.APPROVE_VALIDATION
                self._save_task(task)

        return response

    def next_phase(self) -> str:
        """Перейти к следующей фазе (/task next)."""
        task = self._active_task
        if task is None:
            raise ValueError("Нет активной задачи.")

        # Определяем следующую фазу
        next_status = TaskStateMachine.get_next_phase(task.status)

        # Строгая проверка через TaskTransitionGuard (программный уровень 1)
        result = self._guard.validate_transition(task, next_status.value)
        if not result.allowed:
            raise ValueError(
                self._guard.format_error(task.status.value, next_status.value, result)
            )

        # Сохраняем текущий артефакт в историю
        task.history.append({
            "phase": task.status.value,
            "artifact": task.artifact,
            "completed_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        })

        # Переход
        task.status = next_status
        task.artifact = {}
        task.current_step = 0
        task.expected_action = TaskStateMachine.get_expected_action(next_status)

        self._save_task(task)

        # Обновляем system prompt и чистим историю агента
        if next_status != TaskStatus.DONE:
            self._override_system_prompt(self._build_phase_prompt(task))
            self._agent.clear_history()
            phase_label = next_status.value.upper()
            return (
                f'Переход: → {phase_label}\n'
                f'Задача: "{task.title}"\n'
                f"Продолжайте диалог в фазе {phase_label}."
            )
        else:
            self._restore_system_prompt()
            self._active_task = None
            return (
                f'Задача "{task.title}" завершена!\n'
                "Используйте /task history для просмотра всех этапов."
            )

    def pause_task(self) -> str:
        """Приостановить активную задачу (/task pause)."""
        task = self._active_task
        if task is None:
            raise ValueError("Нет активной задачи для паузы.")
        if task.status in (TaskStatus.PAUSED, TaskStatus.DONE):
            raise ValueError(f"Задача уже в статусе '{task.status.value}'.")

        task.paused_at = task.status
        task.status = TaskStatus.PAUSED
        task.expected_action = ExpectedAction.NONE
        self._save_task(task)

        self._restore_system_prompt()
        self._active_task = None

        return (
            f'Задача #{task.id} "{task.title}" приостановлена.\n'
            f"Фаза при паузе: {task.paused_at.value}\n"
            "Для продолжения: /task load {id} → /task resume"
        )

    def resume_task(self) -> str:
        """Возобновить задачу с паузы (/task resume)."""
        task = self._active_task
        if task is None:
            raise ValueError(
                "Нет активной задачи. Сначала загрузите: /task load <id>"
            )
        if task.status != TaskStatus.PAUSED:
            raise ValueError(
                f"Задача не на паузе (статус: {task.status.value}). "
                "Команда resume работает только для паузных задач."
            )

        # Строгая проверка целостности через TaskTransitionGuard (программный уровень 1)
        resume_result = self._guard.validate_resume(task)
        if not resume_result.allowed:
            raise ValueError(self._guard.format_resume_error(resume_result))

        original_phase = task.paused_at
        if original_phase is None:
            raise ValueError("Не удалось определить фазу при паузе.")

        # Восстанавливаем фазу
        task.status = original_phase
        task.paused_at = None
        task.expected_action = TaskStateMachine.get_expected_action(original_phase)

        # Очищаем short-term (не нужен стейтный диалог)
        if self._memory_manager:
            self._memory_manager.clear_short_term()

        # Добавляем контекст задачи в working memory
        if self._memory_manager:
            artifact_summary = json.dumps(task.artifact, ensure_ascii=False)[:300]
            self._memory_manager.add_to_working(
                f"task_{task.id}_context",
                f"Задача: {task.title} | Фаза: {original_phase.value} | "
                f"Шаг: {task.current_step} | Артефакт: {artifact_summary}",
            )

        # Строим prompt фазы + resume-дополнение
        phase_prompt = self._build_phase_prompt(task)
        artifact_summary = json.dumps(task.artifact, ensure_ascii=False)[:500]
        resume_addition = task_prompts.resume_prompt(
            task.title, original_phase.value, artifact_summary,
        )
        self._override_system_prompt(phase_prompt + resume_addition)

        # Очищаем историю агента (свежий контекст)
        self._agent.clear_history()

        self._save_task(task)

        return (
            f'Задача "{task.title}" возобновлена.\n'
            f"Фаза: {task.status.value}, шаг: {task.current_step}\n"
            "Продолжайте диалог."
        )

    # ------------------------------------------------------------------
    # Отображение
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        """Форматированный статус текущей задачи (/task status)."""
        task = self._active_task
        if task is None:
            return "Нет активной задачи."

        lines = [
            f"\n  Задача #{task.id}: \"{task.title}\"",
            f"  Статус          : {task.status.value}",
        ]
        if task.paused_at:
            lines.append(f"  Фаза при паузе  : {task.paused_at.value}")
        lines.append(f"  Текущий шаг     : {task.current_step}")
        lines.append(f"  Ожидается       : {task.expected_action.value}")

        has_artifact = bool(task.artifact)
        lines.append(f"  Артефакт        : {'есть' if has_artifact else 'пусто'}")
        lines.append(f"  Завершённых фаз : {len(task.history)}")
        lines.append(f"  Создана         : {task.created_at}")
        lines.append(f"  Обновлена       : {task.updated_at}")
        lines.append("")

        return "\n".join(lines)

    def get_artifact(self) -> str:
        """Показать артефакт текущей фазы (/task artifact)."""
        task = self._active_task
        if task is None:
            return "Нет активной задачи."
        if not task.artifact:
            return "Артефакт текущей фазы пуст."

        formatted = json.dumps(task.artifact, ensure_ascii=False, indent=2)
        return f"\n  Артефакт ({task.status.value}):\n{formatted}\n"

    def get_history(self) -> str:
        """Показать историю завершённых фаз (/task history)."""
        task = self._active_task
        if task is None:
            # Попробуем показать последнюю задачу
            tasks = self.list_tasks()
            if not tasks:
                return "Задач нет."
            task = tasks[0]

        if not task.history:
            return f'Задача #{task.id} "{task.title}": история пуста.'

        lines = [f'\n  История задачи #{task.id}: "{task.title}"']
        for i, entry in enumerate(task.history, 1):
            phase = entry.get("phase", "?")
            completed = entry.get("completed_at", "?")
            artifact = entry.get("artifact", {})
            artifact_preview = json.dumps(artifact, ensure_ascii=False)[:200]
            lines.append(f"\n  [{i}] Фаза: {phase}")
            lines.append(f"      Завершена: {completed}")
            lines.append(f"      Артефакт: {artifact_preview}")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Парсинг ответов LLM
    # ------------------------------------------------------------------

    def _parse_artifact(self, response: str, artifact_type: str) -> dict | None:
        """Парсить маркер [ARTIFACT:TYPE] и извлечь JSON из ответа.

        Ищет паттерн: [ARTIFACT:TYPE] за которым следует JSON
        (опционально обёрнутый в ```json ... ```).
        """
        # Паттерн 1: маркер + блок ```json ... ```
        pattern = (
            rf'\[ARTIFACT:{artifact_type}\]\s*'
            rf'```(?:json)?\s*(\{{.*?\}})\s*```'
        )
        match = re.search(pattern, response, re.DOTALL)

        if not match:
            # Паттерн 2: маркер + голый JSON
            pattern2 = rf'\[ARTIFACT:{artifact_type}\]\s*(\{{.*?\}})'
            match = re.search(pattern2, response, re.DOTALL)

        if not match:
            return None

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def _parse_step_done(self, response: str) -> list[int]:
        """Парсить маркеры [STEP_DONE:N] из ответа."""
        matches = re.findall(r'\[STEP_DONE:(\d+)\]', response)
        return [int(m) for m in matches]

    # ------------------------------------------------------------------
    # Управление system prompt
    # ------------------------------------------------------------------

    def _override_system_prompt(self, prompt: str) -> None:
        """Сохранить оригинальный prompt и установить фазовый."""
        if self._original_system_prompt is None:
            self._original_system_prompt = self._agent._system_prompt
        self._agent._system_prompt = prompt

    def _restore_system_prompt(self) -> None:
        """Восстановить оригинальный system prompt."""
        if self._original_system_prompt is not None:
            self._agent._system_prompt = self._original_system_prompt
            self._original_system_prompt = None

    def _build_phase_prompt(self, task: TaskState) -> str:
        """Построить system prompt для текущей фазы задачи."""
        if task.status == TaskStatus.PLANNING:
            base = task_prompts.planning_prompt(task.title)
        elif task.status == TaskStatus.EXECUTION:
            base = task_prompts.execution_prompt(
                task.title, task.plan_artifact, task.current_step,
            )
        elif task.status == TaskStatus.VALIDATION:
            base = task_prompts.validation_prompt(
                task.title, task.plan_artifact, task.result_artifact,
            )
        elif task.status == TaskStatus.DONE:
            return task_prompts.done_prompt(task.title)
        else:
            return ""

        # Добавляем блок состояния для информирования LLM о текущем этапе и инвариантах
        total_steps = len(task.plan_steps)
        state_block = self._guard.build_task_state_block(task, total_steps)
        return base + "\n\n" + state_block

    # ------------------------------------------------------------------
    # Ресурсы
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Закрыть DB-соединение."""
        self._conn.close()

    def __enter__(self) -> "TaskOrchestrator":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
