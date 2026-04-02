"""
ReAct (Reasoning + Acting) цикл.
Агент думает → выбирает инструмент → получает результат → думает снова.
Цикл продолжается до вызова инструмента 'finish' (или [FINISH] блока) или MAX_STEPS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from llm_agent.application.agent import SimpleAgent
from llm_agent.domain.protocols import LLMClientProtocol
from llm_agent.file_tools import TOOLS_SCHEMA, FileSystemToolkit
from llm_agent.tool_executor import ToolExecutor

MAX_STEPS = 15

# ANSI
_R     = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_CYAN  = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"


@dataclass
class AgentResult:
    summary: str
    files_read: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    steps: int = 0
    warning: str | None = None
    elapsed_sec: float = 0.0


class AgentLoop:
    """
    Основной ReAct-цикл файлового ассистента.

    На каждом шаге:
    1. Формируется prompt (задача + накопленная история через SimpleAgent)
    2. LLM генерирует ответ
    3. Парсится [TOOL_CALL] или [FINISH]
    4. Результат добавляется в контекст через следующий ask()
    5. Повторяется до finish или MAX_STEPS
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._executor = ToolExecutor()

    def run(
        self,
        task: str,
        toolkit: FileSystemToolkit,
        llm_client: LLMClientProtocol,
    ) -> AgentResult:
        """
        Запустить ReAct-цикл для задачи task.

        Args:
            task: Текст задачи от пользователя.
            toolkit: Инициализированный FileSystemToolkit.
            llm_client: LLM-клиент (LLMClientProtocol-совместимый).

        Returns:
            AgentResult с резюме и статистикой.
        """
        system_prompt = self._build_system_prompt(TOOLS_SCHEMA)
        agent = SimpleAgent(llm_client=llm_client, system_prompt=system_prompt)

        t0 = time.time()
        step = 0
        files_read: list[str] = []
        files_modified: list[str] = []

        # Первое сообщение: задача
        current_prompt = (
            f"Задача: {task}\n\n"
            "Начни с размышления о задаче, затем используй инструменты для её выполнения."
        )

        while step < MAX_STEPS:
            step += 1
            self._print_step_header(step)

            response = agent.ask(current_prompt)

            # ── Шаг 1: проверить [FINISH] ──────────────────────────────────
            finish_data = self._executor.parse_finish(response)
            if finish_data:
                self._print_finish(step, finish_data.get("summary", ""))
                return AgentResult(
                    summary=finish_data.get("summary", "Задача выполнена."),
                    files_read=_split_list(finish_data.get("files_read", ""))
                    or files_read,
                    files_modified=_split_list(finish_data.get("files_modified", ""))
                    or files_modified,
                    steps=step,
                    elapsed_sec=time.time() - t0,
                )

            # ── Шаг 2: проверить [TOOL_CALL] ──────────────────────────────
            tool_call = self._executor.parse_tool_call(response)
            if tool_call:
                name = tool_call.get("name", "?")

                # [TOOL_CALL] name: finish → тоже финиш
                if name == "finish":
                    summary = tool_call.get("summary", "Задача выполнена.")
                    self._print_finish(step, summary)
                    return AgentResult(
                        summary=summary,
                        files_read=_split_list(tool_call.get("files_read", ""))
                        or files_read,
                        files_modified=_split_list(tool_call.get("files_modified", ""))
                        or files_modified,
                        steps=step,
                        elapsed_sec=time.time() - t0,
                    )

                self._print_tool_call(name, tool_call)

                tool_result = self._executor.execute(tool_call, toolkit)
                self._print_tool_result(tool_result)

                # Трекинг — только если выполнение прошло без ошибки
                success = "Ошибка:" not in tool_result
                if name == "read_file" and "path" in tool_call and success:
                    files_read.append(tool_call["path"])
                elif name in ("write_file", "apply_diff") and "path" in tool_call and success:
                    files_modified.append(tool_call["path"])

                current_prompt = tool_result

            else:
                # ── Шаг 3: чистое размышление без вызова инструмента ──────
                self._print_reasoning(response)
                current_prompt = (
                    "Продолжай. Вызови инструмент для выполнения задачи "
                    "или завершись через [FINISH]."
                )

        # MAX_STEPS достигнут
        elapsed = time.time() - t0
        warning = f"Достигнут максимум шагов ({MAX_STEPS}). Увеличьте MAX_STEPS если нужно."
        if self._verbose:
            print(f"\n{_YELLOW}⚠️  {warning}{_R}")
        return AgentResult(
            summary="Достигнут лимит шагов. Задача может быть выполнена частично.",
            files_read=files_read,
            files_modified=files_modified,
            steps=step,
            warning=warning,
            elapsed_sec=elapsed,
        )

    # ─── System prompt ────────────────────────────────────────────────────────

    def _build_system_prompt(self, tools_schema: list[dict]) -> str:
        """System prompt для ReAct-агента с описанием всех инструментов."""
        tools_lines: list[str] = []
        for tool in tools_schema:
            params = "\n".join(
                f"    {k}: {v}" for k, v in tool["parameters"].items()
            )
            tools_lines.append(
                f"• {tool['name']}\n"
                f"  {tool['description']}\n"
                f"  Параметры:\n{params}"
            )
        tools_text = "\n\n".join(tools_lines)

        return f"""Ты — ассистент разработчика. Выполняешь задачи работы с файлами проекта.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_text}

ПРАВИЛА:
1. Сначала исследуй структуру (list_files), потом читай конкретные файлы
2. Не читай файл целиком если нужно найти одно место — используй search_in_files
3. Перед изменением файла — покажи diff через show_diff
4. Не изменяй файлы без явной необходимости (если задача аналитическая — только читай)
5. Вызывай finish когда задача выполнена

ФОРМАТ ВЫЗОВА ИНСТРУМЕНТА:
[TOOL_CALL]
name: имя_инструмента
параметр1: значение1
параметр2: значение2
[/TOOL_CALL]

ФОРМАТ ЗАВЕРШЕНИЯ:
[FINISH]
summary: что было сделано
files_read: файл1, файл2
files_modified: (пусто если только анализ)
[/FINISH]

Начинай с размышления о задаче, потом действуй."""

    # ─── Вывод прогресса ──────────────────────────────────────────────────────

    def _print_step_header(self, step: int) -> None:
        if not self._verbose:
            return
        bar = "━" * 50
        print(f"\n{bar}")
        print(f"🤔 {_BOLD}Шаг {step}/{MAX_STEPS}{_R}")
        print(bar)

    def _print_tool_call(self, name: str, tool_call: dict) -> None:
        if not self._verbose:
            return
        print(f"\n🔧 {_CYAN}Вызов: {name}{_R}")
        for k, v in tool_call.items():
            if k == "name":
                continue
            display = str(v)
            if len(display) > 80:
                display = display[:77] + "..."
            print(f"   {k}: {display}")

    def _print_tool_result(self, result: str) -> None:
        if not self._verbose:
            return
        lines = result.split("\n")
        preview = "\n".join(lines[:6])
        suffix = f"\n   {_DIM}... (+{len(lines) - 6} строк){_R}" if len(lines) > 6 else ""
        print(f"\n📋 {_BOLD}Результат:{_R}")
        for line in preview.split("\n"):
            print(f"   {line}")
        if suffix:
            print(suffix)

    def _print_reasoning(self, response: str) -> None:
        if not self._verbose:
            return
        # Show first 3 lines of pure reasoning
        lines = [l for l in response.split("\n") if l.strip()][:3]
        print(f"\n💭 {_DIM}Размышление (без вызова инструмента):{_R}")
        for line in lines:
            print(f"   {line}")

    def _print_finish(self, step: int, summary: str) -> None:
        if not self._verbose:
            return
        print(f"\n{_GREEN}✅ Завершено на шаге {step}{_R}")
        if summary:
            print(f"   {summary}")


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def _split_list(value: str) -> list[str]:
    """Разбить строку вида 'файл1, файл2' или 'файл1\nфайл2' в список."""
    if not value:
        return []
    # Ignore placeholder values
    stripped = value.strip()
    if stripped in ("", "(пусто)", "(пусто если только анализ)", "-"):
        return []
    return [v.strip() for v in stripped.replace(",", "\n").split("\n") if v.strip()]
