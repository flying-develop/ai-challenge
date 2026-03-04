"""Интерактивный CLI с поддержкой переключения стратегий и LLM-провайдеров.

Использование:
    python chat.py                         # авто-определение провайдера
    python chat.py --provider qwen         # выбрать провайдер
    python chat.py --provider openai --model gpt-4o-mini
    python chat.py --provider claude --model claude-haiku-4-5-20251001
    python chat.py --strategy 2            # сразу запустить с нужной стратегией

Или через модуль:
    python -m llm_agent.interfaces.cli.interactive_strategies --provider openai

Команды в интерактивном режиме:
    /provider              — показать текущего провайдера
    /provider <name>       — переключить провайдер (qwen / openai / claude)
    /model <name>          — сменить модель в рамках текущего провайдера
    /providers             — список доступных провайдеров

    /strategy              — показать текущую стратегию
    /switch <номер>        — переключить стратегию (1=Sliding Window, 2=Facts, 3=Branching)
    /stats                 — показать статистику агента

    /facts                 — показать извлечённые факты (для стратегии 2)
    /checkpoint <id>       — сохранить checkpoint (для стратегии 3)
    /branch <id> <cp_id>   — создать ветку от checkpoint-а
    /switch-branch <id>    — переключиться на ветку
    /branches              — список веток и checkpoint-ов

    /clear                 — сбросить историю
    /help                  — показать справку
    exit, quit             — выход
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import os
import readline  # noqa: F401 — подключает историю ввода (стрелки вверх/вниз)
import shlex
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.context_strategies import (
    BranchingStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)
from llm_agent.application.strategy_agent import StrategyAgent
from llm_agent.infrastructure.llm_factory import (
    DEFAULT_MODELS,
    PROVIDER_LABELS,
    SUPPORTED_PROVIDERS,
    build_client,
    current_provider_from_env,
    get_available_providers,
)
from llm_agent.infrastructure.token_counter import TiktokenCounter
from llm_agent.memory.manager import MemoryManager
from llm_agent.memory.profile_manager import ProfileManager
from llm_agent.tasks.orchestrator import TaskOrchestrator


# ---------------------------------------------------------------------------
# Спиннер
# ---------------------------------------------------------------------------

@contextmanager
def spinner(text: str = "Думаю") -> Generator[None, None, None]:
    if not sys.stderr.isatty():
        yield
        return

    stop_event = threading.Event()
    frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

    def _spin() -> None:
        while not stop_event.is_set():
            frame = next(frames)
            sys.stderr.write(f"\r\033[K{frame} {text}...")
            sys.stderr.flush()
            time.sleep(0.08)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты полезный ассистент. Давай краткие, точные ответы на русском языке. "
    "Помни всё, что было сказано в нашем разговоре."
)

WINDOW_SIZE = 10
FACTS_WINDOW_SIZE = 6

STRATEGY_NAMES = {
    1: "Sliding Window",
    2: "Sticky Facts / Key-Value Memory",
    3: "Branching (ветки диалога)",
}


# ---------------------------------------------------------------------------
# Справка
# ---------------------------------------------------------------------------

HELP_TEXT = """\
═══ LLM-агент: стратегии контекста + переключение провайдеров ════════

ПРОВАЙДЕРЫ:
  /provider              — показать текущего провайдера и модель
  /provider <name>       — переключить: qwen | openai | claude
  /model <name>          — сменить модель (в рамках текущего провайдера)
  /providers             — список провайдеров с доступностью ключей

СТРАТЕГИИ:
  /strategy              — текущая стратегия
  /switch 1|2|3          — переключить стратегию
                           1 = Sliding Window
                           2 = Sticky Facts / Key-Value Memory
                           3 = Branching (ветки диалога)
  /stats                 — полная статистика агента

ПАМЯТЬ (Memory Layers):
  /memory                        — показать все 3 слоя памяти
  /memory short|working|long     — показать конкретный слой
  /remember working <текст>      — сохранить в рабочую память
  /remember long <текст>         — сохранить в долговременную память
  /forget working [id]           — удалить из рабочей (всё или по id)
  /forget long [id]              — удалить из долговременной (всё или по id)
  /promote working <id>          — переместить из рабочей в долговременную

ПРОФИЛИ (Profiles):
  /profile                              — активный профиль (все поля)
  /profiles                             — список всех профилей
  /profile show <name>                  — показать конкретный профиль
  /profile use <name>                   — переключиться на профиль
  /profile create <name> --interactive  — создать через вопросы (LLM генерирует prompt)
  /profile create <name> --describe "…" — создать из описания (LLM генерирует prompt)
  /profile edit <name>                  — редактировать system_prompt вручную
  /profile edit <name> --describe "…"   — изменить system_prompt через LLM + diff
  /profile delete <name>                — удалить (нельзя удалить активный)
  /profile export <name>                — экспорт в JSON
  /profile import                       — импорт из JSON (вводится в терминале)

STICKY FACTS (стратегия 2):
  /facts                 — показать извлечённые факты

BRANCHING (стратегия 3):
  /checkpoint <id>       — сохранить checkpoint
  /branch <id> <cp_id>   — создать ветку от checkpoint-а
  /switch-branch <id>    — переключиться на ветку
  /branches              — список веток и checkpoint-ов

ЗАДАЧИ (Task Orchestrator):
  /task new "<название>"      — создать задачу, войти в planning
  /task status                — текущее состояние (этап, шаг)
  /task pause                 — приостановить задачу
  /task resume                — продолжить с места паузы
  /task next                  — подтвердить артефакт, перейти к следующей фазе
  /task artifact              — показать артефакт текущей фазы
  /task history               — завершённые фазы и артефакты
  /task list                  — список всех задач
  /task load <id>             — загрузить задачу по id

ОБЩИЕ:
  /clear                 — сбросить историю диалога
  /help                  — эта справка
  exit / quit            — выход
"""


# ---------------------------------------------------------------------------
# Вспомогательные функции для профилей
# ---------------------------------------------------------------------------

def _safe_input(prompt: str = "") -> str:
    """input() с корректной обработкой суррогатных символов (Windows/WSL terminal)."""
    raw = input(prompt)
    return raw.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")


def _generate_system_prompt(llm_client, description: str) -> str:
    """Сгенерировать system prompt через LLM по текстовому описанию."""
    from llm_agent.domain.models import ChatMessage
    messages = [
        ChatMessage(
            role="user",
            content=(
                "Сформируй system prompt для AI-ассистента на основе этого описания. "
                "Верни только текст system prompt, без пояснений, без кавычек.\n\n"
                f"Описание: {description}"
            ),
        )
    ]
    response = llm_client.generate(messages)
    return response.text.strip()


def _modify_system_prompt(llm_client, current: str, instruction: str) -> str:
    """Изменить существующий system prompt через LLM по инструкции."""
    from llm_agent.domain.models import ChatMessage
    messages = [
        ChatMessage(
            role="user",
            content=(
                f"Измени этот system prompt согласно инструкции. "
                f"Верни только новый текст system prompt, без пояснений.\n\n"
                f"Инструкция: {instruction}\n\n"
                f"Текущий system prompt:\n{current}"
            ),
        )
    ]
    response = llm_client.generate(messages)
    return response.text.strip()


def _show_diff(old: str, new: str) -> None:
    """Показать diff между старым и новым system prompt."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile="было", tofile="стало", lineterm=""))
    if diff:
        print("\n  Изменения:")
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"    \033[32m{line}\033[0m")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"    \033[31m{line}\033[0m")
            else:
                print(f"    {line}")
    else:
        print("  (изменений нет)")


def _multiline_input(prompt_text: str = "") -> str:
    """Ввод многострочного текста. Пустая строка завершает ввод."""
    if prompt_text:
        print(prompt_text)
    print("  (введите текст; пустая строка завершает ввод)")
    lines: list[str] = []
    while True:
        try:
            line = _safe_input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def _print_profile(profile) -> None:
    """Отобразить профиль в читаемом виде."""
    active_mark = " ◄ активный" if profile.is_active else ""
    print(f"\n  Профиль: {profile.name}{active_mark}")
    print(f"  Название: {profile.display_name}")
    print(f"  Создан: {profile.created_at}")
    print(f"  Изменён: {profile.updated_at}")
    print(f"  System prompt:\n{_indent(profile.system_prompt, '    ')}")
    print()


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _handle_profile_command(
    parts: list[str],
    agent,
    profile_manager: ProfileManager,
) -> None:
    """Обработать /profile <subcommand> ..."""
    if len(parts) == 1:
        # /profile — показать активный
        active = profile_manager.get_active()
        if active:
            _print_profile(active)
        else:
            print("Нет активного профиля. Используйте /profile use <name>.")
        return

    sub = parts[1].lower()

    # /profile show <name>
    if sub == "show":
        if len(parts) < 3:
            print("Использование: /profile show <name>")
            return
        profile = profile_manager.get(parts[2])
        if profile:
            _print_profile(profile)
        else:
            print(f"Профиль '{parts[2]}' не найден.")
        return

    # /profile use <name>
    if sub == "use":
        if len(parts) < 3:
            print("Использование: /profile use <name>")
            return
        try:
            p = profile_manager.set_active(parts[2])
            print(f"Активный профиль: {p.name} ({p.display_name})")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    # /profile delete <name>
    if sub == "delete":
        if len(parts) < 3:
            print("Использование: /profile delete <name>")
            return
        try:
            profile_manager.delete(parts[2])
            print(f"Профиль '{parts[2]}' удалён.")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    # /profile export <name>
    if sub == "export":
        if len(parts) < 3:
            print("Использование: /profile export <name>")
            return
        try:
            print(profile_manager.export_json(parts[2]))
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    # /profile import
    if sub == "import":
        print("Вставьте JSON профиля (пустая строка завершает ввод):")
        lines: list[str] = []
        while True:
            try:
                line = _safe_input()
            except EOFError:
                break
            if line.strip() == "":
                break
            lines.append(line)
        json_str = "\n".join(lines)
        try:
            p = profile_manager.import_json(json_str)
            print(f"Профиль '{p.name}' импортирован.")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    # /profile create <name> --interactive | --describe "..."
    if sub == "create":
        if len(parts) < 3:
            print("Использование: /profile create <name> --interactive | --describe \"...\"")
            return
        name = parts[2]
        flags = parts[3:]

        if "--interactive" in flags:
            _profile_create_interactive(name, profile_manager, agent._llm_client)
        elif "--describe" in flags:
            idx = flags.index("--describe")
            if idx + 1 >= len(flags):
                print("Укажите описание: --describe \"текст\"")
                return
            description = flags[idx + 1]
            _profile_create_describe(name, profile_manager, agent._llm_client, description)
        else:
            print("Укажите --interactive или --describe \"описание\"")
        return

    # /profile edit <name> [--describe "..."]
    if sub == "edit":
        if len(parts) < 3:
            print("Использование: /profile edit <name> [--describe \"...\"]")
            return
        name = parts[2]
        if name == profile_manager.DEFAULT_PROFILE:
            print(f"Профиль '{profile_manager.DEFAULT_PROFILE}' является системным и не может быть изменён.")
            return
        flags = parts[3:]
        profile = profile_manager.get(name)
        if profile is None:
            print(f"Профиль '{name}' не найден.")
            return

        if "--describe" in flags:
            idx = flags.index("--describe")
            if idx + 1 >= len(flags):
                print("Укажите инструкцию: --describe \"что изменить\"")
                return
            instruction = flags[idx + 1]
            _profile_edit_describe(name, profile, profile_manager, agent._llm_client, instruction)
        else:
            _profile_edit_interactive(name, profile, profile_manager)
        return

    print(f"Неизвестная подкоманда профиля: '{sub}'")
    print("Используйте /help для списка команд.")


def _profile_create_interactive(name: str, profile_manager: ProfileManager, llm_client) -> None:
    """Создать профиль через серию вопросов."""
    print(f"\nСоздаём профиль '{name}'. Ответьте на вопросы (Enter — пропустить):\n")
    display_name = _safe_input("  Как называть этот профиль? ").strip() or name
    language = _safe_input("  На каком языке отвечать? ").strip()
    style = _safe_input("  Какой стиль общения предпочитаешь? ").strip()
    dont_do = _safe_input("  Что ассистент НЕ должен делать? ").strip()
    context = _safe_input("  Есть ли контекст о тебе, который важно знать? ").strip()

    parts_desc: list[str] = []
    if language:
        parts_desc.append(f"Язык общения: {language}")
    if style:
        parts_desc.append(f"Стиль: {style}")
    if dont_do:
        parts_desc.append(f"Запрещено: {dont_do}")
    if context:
        parts_desc.append(f"Контекст о пользователе: {context}")

    if not parts_desc:
        print("Ни одного ответа не получено, создание отменено.")
        return

    description = "\n".join(parts_desc)
    print("\n  Генерирую system prompt...")
    system_prompt = _generate_system_prompt(llm_client, description)
    print(f"\n  Сгенерированный system prompt:\n{_indent(system_prompt, '    ')}\n")

    choice = _safe_input("  Сохранить? [y=да / e=редактировать / n=отмена]: ").strip().lower()
    if choice == "y":
        try:
            profile_manager.create(name, display_name, system_prompt)
            print(f"Профиль '{name}' сохранён.")
        except ValueError as e:
            print(f"Ошибка: {e}")
    elif choice == "e":
        new_prompt = _multiline_input(f"\n  Введите новый system prompt для '{name}':")
        if new_prompt:
            try:
                profile_manager.create(name, display_name, new_prompt)
                print(f"Профиль '{name}' сохранён.")
            except ValueError as e:
                print(f"Ошибка: {e}")
        else:
            print("Пустой prompt, создание отменено.")
    else:
        print("Создание отменено.")


def _profile_create_describe(
    name: str,
    profile_manager: ProfileManager,
    llm_client,
    description: str,
) -> None:
    """Создать профиль по текстовому описанию через LLM."""
    display_name = _safe_input(f"  Название профиля (Enter = '{name}'): ").strip() or name
    print(f"\n  Генерирую system prompt для '{name}'...")
    system_prompt = _generate_system_prompt(llm_client, description)
    print(f"\n  Сгенерированный system prompt:\n{_indent(system_prompt, '    ')}\n")

    choice = _safe_input("  Сохранить? [y=да / e=редактировать / n=отмена]: ").strip().lower()
    if choice == "y":
        try:
            profile_manager.create(name, display_name, system_prompt)
            print(f"Профиль '{name}' сохранён.")
        except ValueError as e:
            print(f"Ошибка: {e}")
    elif choice == "e":
        new_prompt = _multiline_input(f"\n  Введите новый system prompt для '{name}':")
        if new_prompt:
            try:
                profile_manager.create(name, display_name, new_prompt)
                print(f"Профиль '{name}' сохранён.")
            except ValueError as e:
                print(f"Ошибка: {e}")
        else:
            print("Пустой prompt, создание отменено.")
    else:
        print("Создание отменено.")


def _profile_edit_interactive(name: str, profile, profile_manager: ProfileManager) -> None:
    """Редактировать system_prompt профиля вручную."""
    print(f"\n  Текущий system prompt для '{name}':\n{_indent(profile.system_prompt, '    ')}\n")
    new_prompt = _multiline_input(f"  Введите новый system prompt:")
    if new_prompt:
        profile_manager.update(name, system_prompt=new_prompt)
        print(f"Профиль '{name}' обновлён.")
    else:
        print("Пустой prompt, изменение отменено.")


def _profile_edit_describe(
    name: str,
    profile,
    profile_manager: ProfileManager,
    llm_client,
    instruction: str,
) -> None:
    """Изменить system_prompt профиля через LLM по инструкции."""
    print(f"\n  Изменяю system prompt для '{name}'...")
    new_prompt = _modify_system_prompt(llm_client, profile.system_prompt, instruction)
    _show_diff(profile.system_prompt, new_prompt)
    print(f"\n  Новый system prompt:\n{_indent(new_prompt, '    ')}\n")

    choice = _safe_input("  Сохранить изменения? [y/n]: ").strip().lower()
    if choice == "y":
        profile_manager.update(name, system_prompt=new_prompt)
        print(f"Профиль '{name}' обновлён.")
    else:
        print("Изменение отменено.")


# ---------------------------------------------------------------------------
# Обработка команд задач (Task Orchestrator)
# ---------------------------------------------------------------------------

def _handle_task_command(parts: list[str], orchestrator: TaskOrchestrator) -> None:
    """Обработка /task <подкоманда> ..."""
    if len(parts) == 1:
        if orchestrator.has_active_task:
            print(orchestrator.get_status())
        else:
            print('Нет активной задачи. Используйте /task new "название".')
        return

    sub = parts[1].lower()

    if sub == "new":
        if len(parts) < 3:
            print('Использование: /task new "название задачи"')
            return
        title = " ".join(parts[2:])
        try:
            task = orchestrator.create_task(title)
            print(f'\n  Задача #{task.id} создана: "{task.title}"')
            print(f"  Фаза: {task.status.value}")
            print("  Опишите задачу подробнее, чтобы LLM составил план.\n")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    if sub == "status":
        print(orchestrator.get_status())
        return

    if sub == "pause":
        try:
            msg = orchestrator.pause_task()
            print(msg)
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    if sub == "resume":
        try:
            msg = orchestrator.resume_task()
            print(msg)
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    if sub == "next":
        try:
            msg = orchestrator.next_phase()
            print(msg)
        except ValueError as e:
            print(f"Ошибка: {e}")
        return

    if sub == "artifact":
        print(orchestrator.get_artifact())
        return

    if sub == "history":
        print(orchestrator.get_history())
        return

    if sub == "list":
        tasks = orchestrator.list_tasks()
        if not tasks:
            print("Задач нет.")
        else:
            print(f"\n  Задачи ({len(tasks)} шт.):")
            for t in tasks:
                active_mark = ""
                if orchestrator.active_task and orchestrator.active_task.id == t.id:
                    active_mark = " << активная"
                print(f"    [#{t.id}] {t.title} ({t.status.value}){active_mark}")
            print()
        return

    if sub == "load":
        if len(parts) < 3:
            print("Использование: /task load <id>")
            return
        try:
            task_id = int(parts[2])
            task = orchestrator.load_task(task_id)
            print(f'Загружена задача #{task.id}: "{task.title}" ({task.status.value})')
        except (ValueError, TypeError) as e:
            print(f"Ошибка: {e}")
        return

    print(f"Неизвестная подкоманда задачи: '{sub}'")
    print("Используйте /help для списка команд.")


# ---------------------------------------------------------------------------
# Обработка команд
# ---------------------------------------------------------------------------

def _show_memory_layer(title: str, items: list, fmt_fn) -> None:
    """Вспомогательная функция для отображения слоя памяти."""
    print(f"\n  {title}:")
    if not items:
        print("    (пусто)")
    else:
        for item in items:
            print(f"    {fmt_fn(item)}")
    print()


def handle_command(
    cmd: str,
    agent: StrategyAgent,
    strategies: dict,
    current_strategy_num: int,
    memory_manager: MemoryManager | None = None,
    profile_manager: ProfileManager | None = None,
    task_orchestrator: TaskOrchestrator | None = None,
) -> tuple[int, bool]:
    """Обработать команду. Возвращает (strategy_num, was_handled)."""
    try:
        parts = shlex.split(cmd.strip())
    except ValueError:
        # Не сбалансированные кавычки — откатываемся к простому сплиту
        parts = cmd.strip().split()
    if not parts:
        return current_strategy_num, False
    command = parts[0].lower()

    # ---- Справка ----
    if command == "/help":
        print(HELP_TEXT)
        return current_strategy_num, True

    # ---- Провайдеры ----
    if command == "/providers":
        print("\n  Доступные провайдеры:")
        for info in get_available_providers():
            status = "✓" if info["available"] else "✗ нет ключа"
            marker = " ◄ текущий" if info["provider"] == agent.provider_name else ""
            print(
                f"    [{info['provider']:8}] {info['label']:<28} "
                f"{status}{marker}"
            )
            if not info["available"]:
                print(f"              Нужна переменная: {info['key_var']}")
        print()
        return current_strategy_num, True

    if command == "/provider":
        if len(parts) == 1:
            model_str = f"  модель: {agent.model_name}" if agent.model_name else ""
            print(f"Провайдер: {agent.provider_name}  ({PROVIDER_LABELS.get(agent.provider_name, '')}){model_str}")
            return current_strategy_num, True

        new_provider = parts[1].lower()
        if new_provider not in SUPPORTED_PROVIDERS:
            print(f"Неизвестный провайдер: {new_provider!r}")
            print(f"Доступные: {', '.join(SUPPORTED_PROVIDERS)}")
            return current_strategy_num, True

        model = parts[2] if len(parts) > 2 else None
        try:
            new_client = build_client(new_provider, model=model)
            model_used = model or DEFAULT_MODELS[new_provider]
            change = agent.switch_client(
                new_client,
                provider_name=new_provider,
                model_name=model_used,
            )
            print(f"Провайдер переключён: {change}")
            print(f"Модель: {model_used}")
            print(f"История сохранена — продолжаем диалог с новым провайдером.\n")
        except ValueError as exc:
            print(f"Ошибка: {exc}")
        return current_strategy_num, True

    if command == "/model":
        if len(parts) < 2:
            print(f"Использование: /model <название-модели>")
            print(f"Текущий провайдер: {agent.provider_name}, модель: {agent.model_name or '(по умолчанию)'}")
            return current_strategy_num, True
        model = parts[1]
        try:
            new_client = build_client(agent.provider_name, model=model)
            change = agent.switch_client(
                new_client,
                provider_name=agent.provider_name,
                model_name=model,
            )
            print(f"Модель переключена: {change}")
            print(f"История сохранена — продолжаем диалог.\n")
        except ValueError as exc:
            print(f"Ошибка: {exc}")
        return current_strategy_num, True

    # ---- Стратегии ----
    if command == "/strategy":
        print(f"Текущая стратегия: [{current_strategy_num}] {agent.strategy_name}")
        return current_strategy_num, True

    if command == "/switch":
        if len(parts) < 2 or parts[1] not in ("1", "2", "3"):
            print("Использование: /switch 1|2|3")
            for num, name in STRATEGY_NAMES.items():
                marker = " <-- текущая" if num == current_strategy_num else ""
                print(f"  {num} = {name}{marker}")
            return current_strategy_num, True
        new_num = int(parts[1])
        if new_num == current_strategy_num:
            print(f"Стратегия «{STRATEGY_NAMES[new_num]}» уже активна.")
            return current_strategy_num, True
        new_strategy = strategies[new_num]
        new_strategy.reset()
        # Если переключаемся на StickyFacts — прокидываем текущий клиент
        if isinstance(new_strategy, StickyFactsStrategy):
            new_strategy._llm_client = agent._llm_client
        change = agent.switch_strategy(new_strategy)
        current_strategy_num = new_num
        print(f"Стратегия переключена: {change}")
        print(f"История сброшена. Начинаем новый диалог.\n")
        return current_strategy_num, True

    if command == "/stats":
        stats = agent.get_stats()
        print("Статистика:")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
        return current_strategy_num, True

    # ---- Sticky Facts ----
    if command == "/facts":
        strategy = agent.strategy
        if not isinstance(strategy, StickyFactsStrategy):
            print("Команда /facts доступна только для стратегии Sticky Facts (2).")
            return current_strategy_num, True
        facts = strategy.facts
        if not facts:
            print("Фактов пока нет.")
        else:
            print(f"Извлечённые факты ({len(facts)} шт.):")
            for k, v in facts.items():
                print(f"  {k}: {v}")
        return current_strategy_num, True

    # ---- Branching ----
    if command == "/checkpoint":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /checkpoint доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        if len(parts) < 2:
            print("Использование: /checkpoint <id> [описание]")
            return current_strategy_num, True
        cp_id = parts[1]
        desc = " ".join(parts[2:]) if len(parts) > 2 else ""
        try:
            cp = strategy.save_checkpoint(cp_id, description=desc)
            print(f"Checkpoint '{cp_id}' сохранён (ход {cp.turn}, {len(cp.messages)} сообщений).")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    if command == "/branch":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /branch доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        if len(parts) < 3:
            print("Использование: /branch <branch_id> <checkpoint_id> [описание]")
            return current_strategy_num, True
        branch_id, cp_id = parts[1], parts[2]
        desc = " ".join(parts[3:]) if len(parts) > 3 else ""
        try:
            strategy.create_branch(branch_id, cp_id, description=desc)
            print(f"Ветка '{branch_id}' создана от checkpoint '{cp_id}'.")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    if command == "/switch-branch":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /switch-branch доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        if len(parts) < 2:
            print(f"Использование: /switch-branch <id>")
            print(f"Доступные ветки: {', '.join(strategy.branches)}")
            return current_strategy_num, True
        try:
            strategy.switch_branch(parts[1])
            info = strategy.get_branch_info(parts[1])
            total = info.get("total_messages", info.get("messages_count", "?"))
            print(f"Переключено на ветку '{parts[1]}' ({total} сообщений).")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    if command == "/branches":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /branches доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        print(f"Текущая ветка: {strategy.current_branch_id}")
        print("\nCheckpoints:")
        if not strategy.checkpoints:
            print("  (нет)")
        else:
            for cp_id in strategy.checkpoints:
                cp = strategy._checkpoints[cp_id]
                print(f"  [{cp_id}] ход {cp.turn}, {len(cp.messages)} сообщ. {cp.description}")
        print("\nВетки:")
        for b_id in strategy.branches:
            info = strategy.get_branch_info(b_id)
            marker = " ◄ текущая" if b_id == strategy.current_branch_id else ""
            total = info.get("total_messages", info.get("messages_count", "?"))
            desc = info.get("description", "")
            print(f"  [{b_id}] {total} сообщ. {desc}{marker}")
        return current_strategy_num, True

    # ---- Профили ----
    if command == "/profiles":
        if profile_manager is None:
            print("Менеджер профилей не подключён.")
            return current_strategy_num, True
        profiles = profile_manager.list_all()
        if not profiles:
            print("Профилей нет. Создайте: /profile create <name> --describe \"...\"")
        else:
            print(f"\n  Профили ({len(profiles)} шт.):")
            for p in profiles:
                marker = " ◄ активный" if p.is_active else ""
                print(f"    [{p.name}] {p.display_name}{marker}")
        print()
        return current_strategy_num, True

    if command == "/profile":
        if profile_manager is None:
            print("Менеджер профилей не подключён.")
            return current_strategy_num, True
        _handle_profile_command(parts, agent, profile_manager)
        return current_strategy_num, True

    # ---- Память (Memory Layers) ----
    if command == "/memory":
        if memory_manager is None:
            print("Менеджер памяти не подключён.")
            return current_strategy_num, True

        layer = parts[1].lower() if len(parts) > 1 else None

        if layer is None or layer == "short":
            entries = memory_manager.get_short_term()
            _show_memory_layer(
                "SHORT-TERM (краткосрочная — текущий диалог)",
                entries,
                lambda e: f"[{e.ts}] {e.role}: {e.content[:80]}{'...' if len(e.content) > 80 else ''}",
            )
        if layer is None or layer == "working":
            entries = memory_manager.get_working()
            _show_memory_layer(
                "WORKING (рабочая — текущая задача)",
                entries,
                lambda e: f"[id={e.id}] {e.key}: {e.value}",
            )
        if layer is None or layer == "long":
            entries = memory_manager.get_long_term()
            _show_memory_layer(
                "LONG-TERM (долговременная — профиль и знания)",
                entries,
                lambda e: f"[id={e.id}] {e.key}: {e.value}"
                + (f"  tags={e.tags}" if e.tags else ""),
            )
        if layer and layer not in ("short", "working", "long"):
            print("Использование: /memory [short|working|long]")
        return current_strategy_num, True

    if command == "/remember":
        if memory_manager is None:
            print("Менеджер памяти не подключён.")
            return current_strategy_num, True
        if len(parts) < 3:
            print("Использование: /remember working|long <текст>")
            return current_strategy_num, True
        layer = parts[1].lower()
        text = " ".join(parts[2:])
        # Автоматический ключ — первые 3 слова или "note"
        words = text.split()
        auto_key = "_".join(words[:3]).lower().rstrip(".,;:!?") if words else "note"
        if layer == "working":
            entry_id = memory_manager.add_to_working(auto_key, text)
            print(f"Сохранено в WORKING [id={entry_id}]: {auto_key}: {text}")
        elif layer == "long":
            entry_id = memory_manager.add_to_long(auto_key, text)
            print(f"Сохранено в LONG-TERM [id={entry_id}]: {auto_key}: {text}")
        else:
            print("Слой должен быть 'working' или 'long'.")
        return current_strategy_num, True

    if command == "/forget":
        if memory_manager is None:
            print("Менеджер памяти не подключён.")
            return current_strategy_num, True
        if len(parts) < 2:
            print("Использование: /forget working|long [id]")
            return current_strategy_num, True
        layer = parts[1].lower()
        entry_id = int(parts[2]) if len(parts) > 2 else None
        if layer == "working":
            count = memory_manager.remove_from_working(entry_id)
            if entry_id:
                print(f"Удалено из WORKING: {count} запись(ей) (id={entry_id}).")
            else:
                print(f"WORKING очищена: удалено {count} запись(ей).")
        elif layer == "long":
            count = memory_manager.remove_from_long(entry_id)
            if entry_id:
                print(f"Удалено из LONG-TERM: {count} запись(ей) (id={entry_id}).")
            else:
                print(f"LONG-TERM очищена: удалено {count} запись(ей).")
        else:
            print("Слой должен быть 'working' или 'long'.")
        return current_strategy_num, True

    if command == "/promote":
        if memory_manager is None:
            print("Менеджер памяти не подключён.")
            return current_strategy_num, True
        if len(parts) < 3:
            print("Использование: /promote working <id>")
            return current_strategy_num, True
        layer = parts[1].lower()
        try:
            entry_id = int(parts[2])
        except ValueError:
            print("id должен быть числом.")
            return current_strategy_num, True
        try:
            new_id = memory_manager.promote(layer, entry_id)
            print(f"Запись {layer}#{entry_id} перемещена в LONG-TERM [id={new_id}].")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    # ---- Задачи (Task Orchestrator) ----
    if command == "/task":
        if task_orchestrator is None:
            print("Task Orchestrator не подключён.")
            return current_strategy_num, True
        _handle_task_command(parts, task_orchestrator)
        return current_strategy_num, True

    # ---- Общие ----
    if command == "/clear":
        agent.clear_history()
        print("История очищена (working и long-term сохранены).\n")
        return current_strategy_num, True

    return current_strategy_num, False


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run_interactive(provider: str, model: str | None) -> None:
    system_prompt = os.environ.get("QWEN_SYSTEM_PROMPT", "").strip() or SYSTEM_PROMPT

    # Строим клиент
    try:
        client = build_client(provider, model=model)
        model_name = model or DEFAULT_MODELS.get(provider, "")
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)

    token_counter = TiktokenCounter()

    # Memory Layers + Profiles (один и тот же DB-файл)
    memory_db = os.path.join(os.path.expanduser("~"), ".llm-agent", "memory.db")
    memory_manager = MemoryManager(memory_db)
    profile_manager = ProfileManager(memory_db)

    # Стратегии (StickyFacts получает тот же клиент для извлечения фактов)
    strategies: dict = {
        1: SlidingWindowStrategy(window_size=WINDOW_SIZE),
        2: StickyFactsStrategy(window_size=FACTS_WINDOW_SIZE, llm_client=client),
        3: BranchingStrategy(),
    }
    current_strategy_num = 1

    agent = StrategyAgent(
        llm_client=client,
        strategy=strategies[current_strategy_num],
        system_prompt=system_prompt,
        token_counter=token_counter,
        provider_name=provider,
        model_name=model_name,
        memory_manager=memory_manager,
        profile_manager=profile_manager,
    )

    # Task Orchestrator (тот же DB-файл)
    task_orchestrator = TaskOrchestrator(
        db_path=memory_db, agent=agent, memory_manager=memory_manager,
    )

    print("=" * 62)
    print("  LLM-агент: стратегии контекста + выбор провайдера")
    print("=" * 62)
    print(f"\n  Провайдер : {provider}  ({PROVIDER_LABELS.get(provider, '')})")
    print(f"  Модель    : {model_name}")
    print(f"  Стратегия : [{current_strategy_num}] {STRATEGY_NAMES[current_strategy_num]}")
    print(f"\n  Введите /help для списка команд.\n")

    while True:
        try:
            branch_info = ""
            if isinstance(agent.strategy, BranchingStrategy):
                branch_info = f" [{agent.strategy.current_branch_id}]"
            active_profile = profile_manager.get_active()
            profile_info = f"|{active_profile.name}" if active_profile else ""
            task_info = ""
            if task_orchestrator.has_active_task:
                t = task_orchestrator.active_task
                task_info = f"|T:{t.status.value}"
            prefix = f"[{agent.provider_name}|{current_strategy_num}{profile_info}{task_info}]{branch_info}"
            raw = input(f"{prefix} Вы: ")
        except (KeyboardInterrupt, EOFError):
            print("\nДо свидания!")
            break

        user_input = (
            raw.encode("utf-8", errors="surrogateescape")
            .decode("utf-8", errors="replace")
            .replace("\ufffd", "")
            .strip()
        )

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("До свидания!")
            break

        if user_input.startswith("/"):
            current_strategy_num, handled = handle_command(
                user_input, agent, strategies, current_strategy_num,
                memory_manager=memory_manager,
                profile_manager=profile_manager,
                task_orchestrator=task_orchestrator,
            )
            if handled:
                continue

        try:
            with spinner():
                if task_orchestrator.has_active_task:
                    reply = task_orchestrator.handle_message(user_input)
                else:
                    reply = agent.ask(user_input)

            print(f"Агент: {reply}")

            usage = agent.last_token_usage
            if usage:
                print(
                    f"  [tokens: prompt={usage.history_tokens}, "
                    f"response={usage.response_tokens}, "
                    f"total={usage.total_tokens}]"
                )
            print()

        except ValueError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Ошибка API: {exc}", file=sys.stderr)

    task_orchestrator.close()
    memory_manager.close()
    profile_manager.close()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat",
        description="LLM-агент с переключением стратегий контекста и провайдеров.",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=None,
        metavar="NAME",
        help=f"LLM-провайдер: {' | '.join(SUPPORTED_PROVIDERS)} "
             f"(по умолчанию: авто из LLM_PROVIDER или первый с ключом).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Название модели (переопределяет переменную окружения).",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="Показать доступные провайдеры и выйти.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_providers:
        print("\nДоступные провайдеры:")
        for info in get_available_providers():
            status = "✓ доступен" if info["available"] else f"✗ нет {info['key_var']}"
            print(f"  {info['provider']:8} — {info['label']:<30} {status}")
            print(f"             модель по умолчанию: {info['default_model']}")
        print()
        return

    provider = args.provider or current_provider_from_env()
    run_interactive(provider=provider, model=args.model)


if __name__ == "__main__":
    main()
