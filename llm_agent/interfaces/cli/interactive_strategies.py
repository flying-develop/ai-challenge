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
from llm_agent.core.invariant_loader import InvariantLoader
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

try:
    from mcp_client.config import MCPConfigParser
    from mcp_client.client import MCPClient
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


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

ИНВАРИАНТЫ (Invariants):
  /invariants              — показать все текущие инварианты
  /invariants reload       — перечитать файлы из config/invariants/
  /invariants check <текст> — проверить текст на соответствие инвариантам

MCP (Model Context Protocol):
  /mcp servers             — список сконфигурированных серверов (config/mcp-servers.md)
  /mcp tools <name>        — подключиться к серверу и вывести список инструментов
  /mcp tools               — инструменты последнего подключённого сервера (кэш)
  /mcp call <server> <tool> [key=value ...]  — вызвать инструмент MCP-сервера
  /mcp status              — статус: какой сервер, сколько инструментов

КУРСЫ ВАЛЮТ (ЦБ РФ):
  /convert 100 USD         — конвертировать 100 долларов в рубли
  /convert 500 EUR         — конвертировать 500 евро в рубли
  /convert 10000 RUB USD   — конвертировать 10000 рублей в доллары
  /convert 50000 RUB CNY   — конвертировать 50000 рублей в юани
  (Требует: сервер cbr_currencies в config/mcp-servers.md)

НОВОСТИ (News Pipeline — Lenta.ru):
  /news                    — полный пайплайн: RSS → LLM → SQLite + Telegram
  /news status             — статус: последний запуск, категории, флаг Telegram
  /news history [N]        — последние N суммаризаций из БД (по умолчанию 10)
  /news fetch              — только шаг 1: получить новости из RSS
  /news summarize          — шаги 1-2: получить + суммаризировать через LLM
  (Требует: сервер news_digest в config/mcp-servers.md, LLM-ключ в .env)
  (Telegram: задайте TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID в .env)

ИССЛЕДОВАНИЕ (Orchestration MCP — двухпроходный агент):
  /research "<запрос>"     — двухпроходное исследование (8 этапов)
    Пример: /research "хочу сходить в кино в эти выходные"
  /research status         — текущий этап и прогресс последнего исследования
  /research log            — полный журнал последнего исследования
  /research last           — финальный ответ последнего исследования
  (Требует: LLM-ключ в .env; SEARCH_MODE=mock работает без ключей Яндекса)
  (Telegram: задайте TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID в .env)

RAG-ПОИСК (документы podkop-wiki):
  /rag vector|hybrid|bm25  — включить RAG с выбранным режимом поиска
  /rag off                 — отключить RAG
  /rag status              — статус: режим, db, реранкинг, rewrite

  Реранкинг (второй этап после поиска):
  /rag rerank none         — без реранкинга (baseline)
  /rag rerank threshold    — фильтр по порогу score (RERANK_THRESHOLD=0.3)
  /rag rerank cohere       — Cohere Rerank v3.5 (нужен COHERE_API_KEY)
  /rag rerank ollama       — локальная модель Ollama (нужен запущенный ollama)

  Query rewrite (перефразирование запроса перед поиском):
  /rag rewrite on|off      — включить/отключить query rewrite

  Цитаты и источники (День 24):
  /rag citations on|off    — включить/отключить обязательные цитаты в ответе
  /rag confidence          — показать confidence последнего запроса
  /rag verify              — верифицировать цитаты последнего ответа

  Оценка:
  /rag eval                — прогнать 10 вопросов по всем доступным конфигурациям
  /rag eval full           — полный прогон: 10 вопросов + 3 антивопроса с цитатами
  /rag compare             — сравнительная таблица последнего eval (см. /rag eval)
  /rag benchmark           — сравнение local vs cloud (10 вопр + антивопр + стабильность)
  /rag benchmark quick     — быстрый бенчмарк (3 вопроса)

  Прямой поиск:
  /rag search "<запрос>"  — найти релевантные чанки из индекса
  /rag search "<запрос>" --strategy fixed_500  — поиск по конкретной стратегии
  /rag search "<запрос>" --top_k 10            — вернуть 10 результатов
  /rag stats               — статистика индекса (все стратегии)
  /rag stats fixed_500     — статистика по стратегии
  (Требует: pre-built индекс; задайте RAG_DB_PATH в .env или ./output/index.db)
  (Эмбеддинги: DASHSCOPE_API_KEY для Qwen, иначе LocalRandomEmbedder)
  (Cohere: COHERE_API_KEY; Ollama: ollama serve && ollama pull qwen2.5:3b)

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
# Обработка MCP-команд
# ---------------------------------------------------------------------------

def _handle_convert_command(args: list[str], mcp_state: dict) -> None:
    """Обработать /convert <сумма> <валюта> | /convert <сумма> RUB <валюта>.

    Примеры:
        /convert 100 USD           — 100 долларов в рубли
        /convert 10000 RUB USD     — 10000 рублей в доллары
        /convert 500 EUR           — 500 евро в рубли
    """
    if not _MCP_AVAILABLE:
        print("❌ MCP: пакет mcp не установлен.")
        print("   Установите: pip install mcp")
        return

    config_parser: MCPConfigParser = mcp_state.get("config_parser")

    if len(args) == 2:
        try:
            amount = float(args[0].replace(",", "."))
        except ValueError:
            print(f"Ошибка: '{args[0]}' — не число. Укажите сумму цифрами.")
            return
        currency = args[1].upper()
        direction = "to_rub"

    elif len(args) == 3 and args[1].upper() == "RUB":
        try:
            amount = float(args[0].replace(",", "."))
        except ValueError:
            print(f"Ошибка: '{args[0]}' — не число. Укажите сумму цифрами.")
            return
        currency = args[2].upper()
        direction = "from_rub"

    else:
        print("Использование:")
        print("  /convert 100 USD        — доллары в рубли")
        print("  /convert 10000 RUB USD  — рубли в доллары")
        return

    if config_parser is None:
        print("❌ MCP: конфигурация не загружена.")
        return

    try:
        servers = config_parser.load()
    except EnvironmentError as exc:
        print(exc)
        return

    found = next((s for s in servers if s.name == "cbr_currencies"), None)
    if found is None:
        print("MCP-сервер 'cbr_currencies' не сконфигурирован.")
        print("Добавьте в config/mcp-servers.md секцию ## cbr_currencies")
        return

    client = MCPClient(found)
    try:
        result = client.call_tool("convert_currency", {
            "amount": amount,
            "currency_code": currency,
            "direction": direction,
        })
        print(f"💱 {result}")
    except (RuntimeError, ValueError) as exc:
        print(str(exc))


def _handle_news_command(parts: list[str], mcp_state: dict) -> None:
    """Обработать /news [status | history [N] | fetch | summarize].

    Примеры:
        /news                    — полный пайплайн: RSS → LLM → SQLite + Telegram
        /news status             — статус последнего запуска из SQLite
        /news history            — последние 10 суммаризаций из БД
        /news history 5          — последние 5 суммаризаций
        /news fetch              — только шаг 1: получить новости из RSS
        /news summarize          — шаги 1-2: получить + суммаризировать
    """
    if not _MCP_AVAILABLE:
        print("❌ MCP: пакет mcp не установлен.")
        print("   Установите: pip install mcp")
        return

    config_parser: MCPConfigParser = mcp_state.get("config_parser")
    if config_parser is None:
        print("❌ MCP: конфигурация не загружена.")
        return

    try:
        servers = config_parser.load()
    except EnvironmentError as exc:
        print(exc)
        return

    found = next((s for s in servers if s.name == "news_digest"), None)
    if found is None:
        print("MCP-сервер 'news_digest' не сконфигурирован.")
        print("Добавьте в config/mcp-servers.md секцию ## news_digest")
        return

    client = MCPClient(found)
    # parts[0] == "/news", parts[1:] — аргументы
    args = parts[1:]

    try:
        if not args:
            # /news → запустить полный пайплайн
            print("🚀 Запускаю пайплайн (fetch → summarize → deliver)...")
            print(client.call_tool("run_news_pipeline", {}))

        elif args[0] == "status":
            # /news status → читаем SQLite напрямую (без MCP overhead)
            print("📊 Статус пайплайна...")
            try:
                from mcp_server.news_api import get_pipeline_status, get_db_path
                status = get_pipeline_status(get_db_path())
                total = status.get("total_summaries", 0)
                last_date = status.get("last_date")
                last_run = status.get("last_run_at", "")
                print(f"  Всего суммаризаций в БД: {total}")
                if last_date:
                    run_time = last_run[:19].replace("T", " ") if last_run else "—"
                    print(f"  Последний запуск: {last_date} (создано {run_time} UTC)")
                else:
                    print("  Пайплайн ещё не запускался")
                last_sums = status.get("last_summaries", [])
                if last_sums:
                    print("\n  Последние суммаризации:")
                    for s in last_sums[:6]:
                        tg = "📲" if s.get("telegram_sent") else "💾"
                        print(f"    {tg} {s['date']} / {s['category']} ({s.get('article_count', 0)} статей)")
            except ImportError:
                print("  (модуль news_api недоступен)")

        elif args[0] == "history":
            # /news history [N] → последние N суммаризаций из SQLite
            limit = 10
            if len(args) > 1:
                try:
                    limit = int(args[1])
                except ValueError:
                    print(f"Ошибка: '{args[1]}' — не число. Используйте: /news history 5")
                    return
            print(f"📖 Последние {limit} суммаризаций из БД...")
            try:
                from mcp_server.news_api import get_summaries, get_db_path, _CATEGORY_NAMES
                rows = get_summaries(limit=limit, db_path=get_db_path())
                if not rows:
                    print("  История пуста. Запустите /news для создания суммаризаций.")
                else:
                    for row in rows:
                        cat = row.get("category", "")
                        cat_name = _CATEGORY_NAMES.get(cat, cat)
                        date_str = row.get("date", "")
                        count = row.get("article_count", 0)
                        tg = "📲" if row.get("telegram_sent") else "💾"
                        print(f"\n  {tg} {date_str} / {cat_name} ({count} статей):")
                        summary = row.get("summary", "")
                        preview = summary[:200] + ("..." if len(summary) > 200 else "")
                        print(f"  {preview}")
            except ImportError:
                print("  (модуль news_api недоступен)")

        elif args[0] == "fetch":
            # /news fetch → только шаг 1: получить новости из RSS
            print("📡 Шаг 1/3: получение новостей...")
            import json as _json
            result = client.call_tool("fetch_news", {})
            try:
                data = _json.loads(result)
                if "error" in data:
                    print(f"❌ {data['error']}")
                else:
                    cats = data.get("categories", {})
                    total = sum(len(v) for v in cats.values())
                    print(f"✅ Получено {total} статей в {len(cats)} категориях:")
                    try:
                        from mcp_server.news_api import _CATEGORY_NAMES
                    except ImportError:
                        _CATEGORY_NAMES = {}
                    for cat, articles in sorted(cats.items()):
                        cat_name = _CATEGORY_NAMES.get(cat, cat)
                        print(f"   {cat_name}: {len(articles)} статей")
            except Exception:
                print(result)

        elif args[0] == "summarize":
            # /news summarize → шаги 1-2: fetch + summarize
            print("📡 Шаг 1/3: получение новостей...")
            news_json = client.call_tool("fetch_news", {})
            print("🧠 Шаг 2/3: суммаризация через LLM...")
            import json as _json
            result = client.call_tool("summarize_news", {"news_json": news_json})
            try:
                data = _json.loads(result)
                if "error" in data:
                    print(f"❌ {data['error']}")
                else:
                    summaries = data.get("summaries", {})
                    try:
                        from mcp_server.news_api import _CATEGORY_NAMES, _CATEGORY_EMOJI
                    except ImportError:
                        _CATEGORY_NAMES, _CATEGORY_EMOJI = {}, {}
                    for cat, summary in sorted(summaries.items()):
                        cat_name = _CATEGORY_NAMES.get(cat, cat)
                        emoji = _CATEGORY_EMOJI.get(cat, "📌")
                        print(f"\n{emoji} {cat_name}:")
                        print(f"  {summary}")
            except Exception:
                print(result)

        else:
            print("Использование:")
            print("  /news                    — полный пайплайн (fetch→summarize→deliver)")
            print("  /news status             — статус последнего запуска")
            print("  /news history [N]        — последние N суммаризаций из БД")
            print("  /news fetch              — только получить новости (шаг 1)")
            print("  /news summarize          — получить + суммаризировать (шаги 1-2)")

    except (RuntimeError, ValueError) as exc:
        print(str(exc))


def _print_journal_fallback(mcp_state: dict, task_id: str) -> None:
    """Показать журнал из общей БД (нет локального контекста CLI)."""
    journal_client = mcp_state.get("research_journal_client")
    if journal_client is None:
        try:
            from mcp_client.client import MCPClient
            from mcp_client.config import MCPServerConfig
            journal_client = MCPClient(MCPServerConfig(
                name="journal_server", transport="stdio",
                description="Journal", command="python",
                args=["-m", "mcp_server.journal_server"],
            ))
            mcp_state["research_journal_client"] = journal_client
        except Exception as exc:
            print(f"Ошибка создания journal-клиента: {exc}")
            return
    try:
        log = journal_client.call_tool("get_log", {"task_id": task_id})
        print(f"\n{log}\n")
    except Exception as exc:
        print(f"Ошибка получения журнала: {exc}")


def _get_rag_indexer_path() -> str:
    import os
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "rag_indexer"))


def _make_agent_llm_fn(agent):
    """Создать llm_fn из LLM-клиента агента (без засорения истории чата).

    Возвращает функцию (system_prompt, user_prompt) -> str,
    совместимую с RAGEvaluator и PipelineEvaluator.
    """
    from llm_agent.domain.models import ChatMessage

    client = agent._llm_client

    def llm_fn(system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))
        response = client.generate(messages)
        return response.text

    return llm_fn


def _init_rag_retrievers(rag_state: dict) -> str:
    """Lazy init of IndexStore, embedder, and retrievers. Returns error msg or empty str."""
    import os
    import sys

    if rag_state.get("store") is not None:
        return ""

    db_path = rag_state.get("db_path", "")
    if not os.path.exists(db_path):
        return f"Файл индекса не найден: {db_path}"

    # Add rag_indexer to sys.path
    rag_root = os.path.normpath(os.path.join(os.path.dirname(db_path), ".."))
    if rag_root not in sys.path:
        sys.path.insert(0, rag_root)

    try:
        from src.storage.index_store import IndexStore
        from src.embedding.provider import EmbeddingProvider
        from src.retrieval.retriever import VectorRetriever, BM25Retriever, HybridRetriever
    except ImportError as e:
        return f"Ошибка импорта RAG модулей: {e}"

    try:
        store = IndexStore(db_path)
        embedder = EmbeddingProvider.create("qwen")
        strategy_filter = rag_state.get("strategy_filter")

        vector_ret = VectorRetriever(store, embedder, strategy_filter)
        bm25_ret = BM25Retriever(store, strategy_filter)
        hybrid_ret = HybridRetriever(vector_ret, bm25_ret)

        rag_state["store"] = store
        rag_state["embedder"] = embedder
        rag_state["retrievers"] = {
            "vector": vector_ret,
            "bm25": bm25_ret,
            "hybrid": hybrid_ret,
        }
        return ""
    except Exception as e:
        return f"Ошибка инициализации RAG: {e}"


def _run_rag_query(query: str, rag_state: dict, agent) -> str:
    """Run a RAG query and return the answer.

    Если включён реранкинг или query rewrite — использует RAGPipeline.
    Иначе — прямой поиск + RAGQueryBuilder (legacy поведение).
    """
    import os
    import sys
    mode = rag_state.get("mode", "off")
    retriever = rag_state["retrievers"].get(mode)
    if not retriever:
        return agent.ask(query)

    rag_root = os.path.normpath(os.path.join(os.path.dirname(rag_state["db_path"]), "..", ".."))
    if rag_root not in sys.path:
        sys.path.insert(0, rag_root)

    rerank_mode = rag_state.get("rerank_mode", "none")
    rewrite_on = rag_state.get("rewrite", False)
    citations_on = rag_state.get("citations", False)

    # Если включён citations, реранкинг или rewrite — используем RAGPipeline
    if rerank_mode != "none" or rewrite_on or citations_on:
        try:
            from src.retrieval.reranker import ThresholdFilter, CohereReranker, OllamaReranker
            from src.retrieval.query_rewrite import QueryRewriter
            from src.retrieval.pipeline import RAGPipeline
        except ImportError as e:
            print(f"[WARN] Не удалось импортировать pipeline: {e}. Используется базовый поиск.")
            rerank_mode = "none"
            rewrite_on = False
            citations_on = False
        else:
            llm_fn = rag_state.get("llm_fn") or _make_agent_llm_fn(agent)

            # Выбор реранкера
            reranker = None
            if rerank_mode == "threshold":
                threshold = float(os.environ.get("RERANK_THRESHOLD", "0.3"))
                reranker = ThresholdFilter(threshold=threshold)
            elif rerank_mode == "cohere":
                reranker = CohereReranker()
            elif rerank_mode == "ollama":
                ollama_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:3b")
                ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                reranker = OllamaReranker(model=ollama_model, base_url=ollama_base)

            # Query rewriter
            query_rewriter = QueryRewriter(llm_fn) if rewrite_on else None

            pipeline = RAGPipeline(
                retriever=retriever,
                llm_fn=llm_fn,
                reranker=reranker,
                query_rewriter=query_rewriter,
                use_structured=citations_on,
            )

            top_k = rag_state.get("top_k", 5)
            initial_k = int(os.environ.get("RERANK_INITIAL_K", "20")) if reranker else top_k

            rag_answer = pipeline.answer(query, top_k=top_k, initial_k=initial_k)

            if rag_answer.rewrite_variants:
                print(f"\nЗапрос переформулирован ({len(rag_answer.rewrite_variants)} вариантов).")

            if reranker:
                print(f"Кандидаты → финал: {rag_answer.initial_results_count} → {rag_answer.final_results_count} ({rerank_mode})")

            # Режим цитат: показываем структурированный ответ
            if citations_on and rag_answer.structured:
                rag_state["last_structured_response"] = rag_answer.structured
                rag_state["last_confidence"] = rag_answer.confidence
                try:
                    from src.retrieval.formatter import format_structured_response, format_refusal
                    sr = rag_answer.structured
                    if sr.is_refusal:
                        print(format_refusal(sr))
                    else:
                        print(format_structured_response(sr))
                except ImportError:
                    pass
                return ""  # уже напечатано

            # Обычный режим: печатаем источники
            print("\nИсточники:")
            seen: set = set()
            for r in rag_answer.sources:
                if r.source not in seen:
                    print(f"  - {r.source} ({r.section or 'нет раздела'})")
                    seen.add(r.source)

            rag_state["last_confidence"] = rag_answer.confidence
            return rag_answer.answer

    # Legacy: прямой поиск без реранкинга
    try:
        from src.retrieval.rag_query import RAGQueryBuilder
    except ImportError:
        return agent.ask(query)

    results = retriever.search(query, top_k=rag_state["top_k"])
    if not results:
        return agent.ask(query)

    ctx = RAGQueryBuilder().build(query, results)
    reply = agent.ask(ctx.user_prompt)

    print("\nИсточники:")
    seen = set()
    for r in results:
        if r.source not in seen:
            print(f"  - {r.source} ({r.section or 'нет раздела'})")
            seen.add(r.source)

    return reply


def _handle_rag_eval_full(parts: list[str], rag_state: dict) -> str:
    """Полный прогон: 10 основных вопросов + 3 антивопроса со structured-промптом."""
    import os
    import sys as _sys

    msg = _init_rag_retrievers(rag_state)
    if msg:
        return msg

    llm_fn = rag_state.get("llm_fn")
    if llm_fn is None:
        return "Для /rag eval full нужен LLM. Введите /rag hybrid сначала."

    hybrid_ret = rag_state["retrievers"].get("hybrid")
    if not hybrid_ret:
        return "Ретривер 'hybrid' не инициализирован. Введите /rag hybrid."

    _rag_dir = _get_rag_indexer_path()
    if _rag_dir not in _sys.path:
        _sys.path.insert(0, _rag_dir)

    try:
        from src.retrieval.pipeline import RAGPipeline
        from src.retrieval.evaluator import EVAL_QUESTIONS, ANTI_QUESTIONS
        from src.retrieval.confidence import ConfidenceScorer
        from src.retrieval.response_parser import ResponseParser
        from src.retrieval.formatter import format_structured_response
    except ImportError as e:
        return f"Ошибка импорта модулей цитат: {e}"

    threshold = float(os.environ.get("RAG_CONFIDENCE_THRESHOLD", "0.4"))
    pipeline = RAGPipeline(hybrid_ret, llm_fn, use_structured=True)
    scorer = ConfidenceScorer()

    # ── Основные вопросы ─────────────────────────────────────────────────────
    print("\n\n══════════ STRUCTURED RAG EVAL (10 вопросов + 3 антивопроса) ══════════\n")
    print("Прогон основных вопросов...")

    main_rows = []
    for q in EVAL_QUESTIONS:
        try:
            rag_answer = pipeline.answer(q["question"], top_k=5, initial_k=20)
        except Exception as exc:
            print(f"  [ОШИБКА] вопрос {q['id']}: {exc}")
            continue

        sr = rag_answer.structured
        if sr:
            num_src = len(sr.sources)
            num_q = len(sr.quotes)
            ver = len(sr.verified_quotes)
            total_q = len(sr.quotes)
            ver_pct = int(sr.verified_ratio * 100) if total_q else 0
            is_ref = sr.is_refusal
        else:
            num_src = num_q = ver = total_q = ver_pct = 0
            is_ref = False

        conf = rag_answer.confidence
        main_rows.append({
            "id": q["id"],
            "question": q["question"],
            "num_src": num_src,
            "num_q": num_q,
            "ver": ver,
            "total_q": total_q,
            "ver_pct": ver_pct,
            "conf": conf,
            "is_ref": is_ref,
        })

    # Таблица основных вопросов
    w = 31
    print(f"\n╔═════╦{'═' * w}╦═══════╦════════╦════════╦══════════╗")
    print(f"║  #  ║ {'Вопрос':<{w - 1}}║  Src  ║ Quotes ║ Verif  ║  Conf    ║")
    print(f"╠═════╬{'═' * w}╬═══════╬════════╬════════╬══════════╣")
    sum_src = sum_q = sum_ver = sum_total_q = sum_conf = 0
    for row in main_rows:
        q_short = row["question"][: w - 2]
        ver_str = f"{row['ver_pct']}%" if row["total_q"] else "—"
        print(
            f"║ {row['id']:<4}║ {q_short:<{w - 1}}║  {row['num_src']:<5}║  {row['num_q']:<6}║ {ver_str:<6} ║ {row['conf']:.2f}     ║"
        )
        sum_src += row["num_src"]
        sum_q += row["num_q"]
        sum_conf += row["conf"]
        if row["total_q"]:
            sum_ver += row["ver"]
            sum_total_q += row["total_q"]
    n = len(main_rows) or 1
    avg_src = sum_src / n
    avg_q = sum_q / n
    avg_conf = sum_conf / n
    avg_ver_pct = int(sum_ver / sum_total_q * 100) if sum_total_q else 0
    print(f"╠═════╬{'═' * w}╬═══════╬════════╬════════╬══════════╣")
    print(
        f"║ AVG ║ {'':< {w - 1}}║ {avg_src:<5.1f} ║ {avg_q:<6.1f} ║ {avg_ver_pct}%    ║ {avg_conf:.2f}     ║"
    )
    print(f"╚═════╩{'═' * w}╩═══════╩════════╩════════╩══════════╝")

    # ── Антивопросы ───────────────────────────────────────────────────────────
    print("\nПрогон антивопросов...")
    anti_rows = []
    for aq in ANTI_QUESTIONS:
        try:
            rag_answer = pipeline.answer(aq["question"], top_k=5, initial_k=20)
        except Exception as exc:
            print(f"  [ОШИБКА] антивопрос {aq['id']}: {exc}")
            continue

        sr = rag_answer.structured
        is_ref = sr.is_refusal if sr else False
        conf = rag_answer.confidence
        anti_rows.append({
            "id": aq["id"],
            "question": aq["question"],
            "is_ref": is_ref,
            "conf": conf,
        })

    print(f"\n╔═════╦{'═' * w}╦══════════╦═══════╗")
    print(f"║  #  ║ {'Вопрос':<{w - 1}}║ Отказал? ║ Conf  ║")
    print(f"╠═════╬{'═' * w}╬══════════╬═══════╣")
    for row in anti_rows:
        q_short = row["question"][: w - 2]
        ref_str = "✅ да" if row["is_ref"] else "❌ нет"
        print(f"║ {row['id']:<4}║ {q_short:<{w - 1}}║ {ref_str:<8} ║ {row['conf']:.2f}  ║")
    print(f"╚═════╩{'═' * w}╩══════════╩═══════╝")

    # ── Итоговые метрики ─────────────────────────────────────────────────────
    with_src = sum(1 for r in main_rows if r["num_src"] > 0)
    with_q = sum(1 for r in main_rows if r["num_q"] > 0)
    refusals_correct = sum(1 for r in anti_rows if r["is_ref"])
    false_refusals = sum(1 for r in main_rows if r["is_ref"])

    print("\n  Итоговые показатели:")
    print(f"    Вопросов с источниками:     {with_src}/{len(main_rows)} ({int(with_src / n * 100)}%)")
    print(f"    Вопросов с цитатами:        {with_q}/{len(main_rows)} ({int(with_q / n * 100)}%)")
    print(f"    Средний verified_ratio:     {avg_ver_pct}%")
    print(f"    Антивопросов с отказом:     {refusals_correct}/{len(anti_rows)}")
    print(f"    Ложных отказов:             {false_refusals}/{len(main_rows)}")

    return "\nОценка завершена."


def _handle_rag_command(parts: list[str], rag_state: dict) -> str:
    """Handle /rag subcommands."""
    import os
    sub = parts[1] if len(parts) > 1 else "status"

    if sub == "off":
        rag_state["mode"] = "off"
        return "RAG отключён."

    elif sub in ("vector", "hybrid", "bm25"):
        msg = _init_rag_retrievers(rag_state)
        if msg:
            return msg
        rag_state["mode"] = sub
        return f"RAG режим: {sub}"

    elif sub == "rerank":
        rerank_mode = parts[2] if len(parts) > 2 else ""
        valid = ("none", "threshold", "cohere", "ollama")
        if rerank_mode not in valid:
            return f"Использование: /rag rerank [{' | '.join(valid)}]"
        rag_state["rerank_mode"] = rerank_mode
        if rerank_mode == "none":
            return "Реранкинг отключён."
        return f"Реранкинг: {rerank_mode}"

    elif sub == "rewrite":
        action = parts[2].lower() if len(parts) > 2 else ""
        if action in ("on", "true", "1"):
            rag_state["rewrite"] = True
            return "Query rewrite: включён."
        elif action in ("off", "false", "0"):
            rag_state["rewrite"] = False
            return "Query rewrite: отключён."
        else:
            current = "включён" if rag_state.get("rewrite") else "отключён"
            return f"Query rewrite сейчас {current}.\nИспользование: /rag rewrite [on|off]"

    elif sub == "citations":
        action = parts[2].lower() if len(parts) > 2 else ""
        if action in ("on", "true", "1"):
            rag_state["citations"] = True
            return "Режим цитат: включён. Каждый ответ будет содержать [ANSWER]/[SOURCES]/[QUOTES]."
        elif action in ("off", "false", "0"):
            rag_state["citations"] = False
            return "Режим цитат: отключён."
        else:
            current = "включён" if rag_state.get("citations") else "отключён"
            return f"Режим цитат сейчас {current}.\nИспользование: /rag citations [on|off]"

    elif sub == "confidence":
        last = rag_state.get("last_structured_response")
        if last is None:
            conf = rag_state.get("last_confidence")
            if conf is None:
                return "Нет данных о confidence. Задайте вопрос с включённым RAG."
            return f"Confidence последнего запроса: {conf:.3f}"
        import sys as _sys
        _rag_dir = _get_rag_indexer_path()
        if _rag_dir not in _sys.path:
            _sys.path.insert(0, _rag_dir)
        try:
            from src.retrieval.confidence import ConfidenceScorer, ConfidenceLevel
            from src.retrieval.formatter import format_confidence_level
        except ImportError:
            return f"Confidence: {last.confidence:.3f}"
        scorer = ConfidenceScorer()
        level = (
            ConfidenceLevel.HIGH if last.confidence >= scorer.HIGH_THRESHOLD
            else ConfidenceLevel.MEDIUM if last.confidence >= scorer.DEFAULT_THRESHOLD
            else ConfidenceLevel.LOW
        )
        return format_confidence_level(level, last.confidence)

    elif sub == "verify":
        last = rag_state.get("last_structured_response")
        if last is None:
            return "Нет данных для верификации. Задайте вопрос с включённым режимом цитат (/rag citations on)."
        import sys as _sys
        _rag_dir = _get_rag_indexer_path()
        if _rag_dir not in _sys.path:
            _sys.path.insert(0, _rag_dir)
        try:
            from src.retrieval.formatter import format_structured_response
        except ImportError:
            pass
        lines = ["\nВерификация цитат последнего ответа:"]
        if not last.quotes:
            lines.append("  Цитат нет.")
        else:
            for q in last.quotes:
                status = "✅ подтверждена" if q.verified else "❌ не найдена в контексте"
                lines.append(f'  [{q.index}] {status}: "{q.text[:80]}..."')
            pct = int(last.verified_ratio * 100)
            lines.append(f"\nИтого: {len(last.verified_quotes)}/{len(last.quotes)} ({pct}%)")
        return "\n".join(lines)

    elif sub == "status":
        mode = rag_state.get("mode", "off")
        db_path = rag_state.get("db_path", "?")
        strategy = rag_state.get("strategy_filter") or "все стратегии"
        top_k = rag_state.get("top_k", 5)
        rerank_mode = rag_state.get("rerank_mode", "none")
        rewrite = "включён" if rag_state.get("rewrite") else "отключён"
        store = rag_state.get("store")
        chunks_info = ""
        if store:
            try:
                stats = store.get_stats()
                chunks_info = f", чанков: {stats.get('chunks', '?')}"
            except Exception:
                pass
        return (
            f"RAG статус:\n"
            f"  режим: {mode}\n"
            f"  реранкинг: {rerank_mode}\n"
            f"  query rewrite: {rewrite}\n"
            f"  db: {db_path}\n"
            f"  стратегия: {strategy}\n"
            f"  top_k: {top_k}{chunks_info}"
        )

    elif sub == "eval":
        eval_mode = parts[2] if len(parts) > 2 else ""
        if eval_mode == "full":
            return _handle_rag_eval_full(parts, rag_state)
        msg = _init_rag_retrievers(rag_state)
        if msg:
            return msg
        import sys
        import json as _json
        import urllib.request as _urllib
        rag_root = os.path.normpath(os.path.join(os.path.dirname(rag_state["db_path"]), "..", ".."))
        if rag_root not in sys.path:
            sys.path.insert(0, rag_root)
        try:
            from src.retrieval.reranker import ThresholdFilter, CohereReranker, OllamaReranker
            from src.retrieval.query_rewrite import QueryRewriter
            from src.retrieval.pipeline import RAGPipeline, PipelineEvaluator
            _pipeline_available = True
        except ImportError:
            _pipeline_available = False

        if not _pipeline_available:
            # fallback: старый evaluator без LLM
            try:
                from src.retrieval.evaluator import RAGEvaluator
            except ImportError as e:
                return f"Ошибка импорта evaluator: {e}"
            retrievers = list(rag_state["retrievers"].values())
            evaluator = RAGEvaluator(retrievers, llm_fn=None, top_k=rag_state["top_k"])
            results = evaluator.run()
            evaluator.print_report(results)
            return "Оценка завершена (без LLM, без реранкинга)."

        # Определяем llm_fn из rag_state или предупреждаем
        llm_fn = rag_state.get("llm_fn")
        if llm_fn is None:
            return (
                "Для /rag eval с пайплайнами нужен LLM.\n"
                "Введите /rag hybrid сначала (инициализация ретривера), "
                "затем /rag eval снова."
            )

        hybrid_ret = rag_state["retrievers"].get("hybrid")
        if not hybrid_ret:
            return "Ретривер 'hybrid' не инициализирован. Введите /rag hybrid."

        top_k = rag_state.get("top_k", 5)
        threshold = float(os.environ.get("RERANK_THRESHOLD", "0.3"))
        ollama_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:3b")
        ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        # Проверка Cohere
        cohere_key = os.environ.get("COHERE_API_KEY", "")
        cohere_ok = False
        if cohere_key:
            try:
                payload = _json.dumps({
                    "model": "rerank-v3.5", "query": "test",
                    "documents": ["a", "b"], "top_n": 2,
                }).encode()
                req = _urllib.Request(
                    "https://api.cohere.com/v2/rerank", data=payload,
                    headers={"Authorization": f"Bearer {cohere_key}",
                             "Content-Type": "application/json"},
                )
                with _urllib.urlopen(req, timeout=10):
                    pass
                cohere_ok = True
            except Exception:
                pass

        # Проверка Ollama
        ollama_ok = False
        try:
            with _urllib.urlopen(f"{ollama_base}/api/tags", timeout=3):
                ollama_ok = True
        except Exception:
            pass

        # Собираем пайплайны
        pipelines = [
            RAGPipeline(hybrid_ret, llm_fn),  # baseline
            RAGPipeline(hybrid_ret, llm_fn, reranker=ThresholdFilter(threshold)),
        ]
        if cohere_ok:
            pipelines.append(RAGPipeline(hybrid_ret, llm_fn, reranker=CohereReranker()))
        else:
            print("  [INFO] COHERE_API_KEY не задан или недоступен — конфигурации cohere/full пропущены.")
        if ollama_ok:
            pipelines.append(RAGPipeline(
                hybrid_ret, llm_fn,
                reranker=OllamaReranker(model=ollama_model, base_url=ollama_base),
            ))
        else:
            print("  [INFO] Ollama недоступна — конфигурация ollama пропущена.")
        if cohere_ok:
            rewriter = QueryRewriter(llm_fn)
            pipelines.append(RAGPipeline(
                hybrid_ret, llm_fn,
                reranker=CohereReranker(),
                query_rewriter=rewriter,
            ))

        n_questions = 5
        print(f"\n  Конфигурации для оценки: {', '.join(p.name for p in pipelines)}")
        print(f"  Вопросов: {n_questions}  |  Вызовов LLM: {len(pipelines) * n_questions}\n")

        import warnings as _warnings
        evaluator = PipelineEvaluator(pipelines, top_k=top_k, initial_k=20)
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("ignore")
            eval_results = evaluator.run()

        evaluator.print_comparison_table(eval_results)
        evaluator.print_timing_summary(eval_results)

        # Сохранить результаты в rag_state для /rag compare
        rag_state["last_eval_results"] = eval_results
        rag_state["last_eval_evaluator"] = evaluator
        return "Оценка завершена. Используйте /rag compare для таблицы."

    elif sub == "search":
        # Legacy search subcommand — delegate to old-style inline search
        import sys as _sys
        db_path_str = rag_state.get("db_path", "")
        _rag_dir = _get_rag_indexer_path()
        if _rag_dir not in _sys.path:
            _sys.path.insert(0, _rag_dir)

        try:
            from src.embedding.provider import EmbeddingProvider
            from src.storage.index_store import IndexStore
        except ImportError:
            return "RAG-модуль не найден."

        args = parts[1:]
        query_parts = []
        strategy_filter = None
        top_k = 5
        i = 1
        while i < len(args):
            if args[i] == "--strategy" and i + 1 < len(args):
                strategy_filter = args[i + 1]
                i += 2
            elif args[i] == "--top_k" and i + 1 < len(args):
                try:
                    top_k = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                query_parts.append(args[i])
                i += 1

        query = " ".join(query_parts).strip().strip('"').strip("'")
        if not query:
            return "Использование: /rag search \"<запрос>\" [--strategy NAME] [--top_k N]"

        if not os.path.exists(db_path_str):
            return f"Индекс не найден: {db_path_str}"

        try:
            provider = EmbeddingProvider.create("qwen")
            query_vector = provider.embed_texts([query])[0]
        except Exception as exc:
            return f"Ошибка генерации эмбеддинга: {exc}"

        try:
            with IndexStore(db_path_str) as store:
                results = store.search(query_vector, strategy=strategy_filter, top_k=top_k)
        except Exception as exc:
            return f"Ошибка поиска: {exc}"

        if not results:
            return "[INFO] Результатов не найдено."

        lines = [f"\nНайдено {len(results)} результатов:\n" + "─" * 60]
        for res in results:
            c = res.chunk
            lines.append(f"[{res.rank}] score={res.score:.4f} | {c.strategy} | {c.source}")
            lines.append(f"    Раздел : {c.section or '(нет)'}")
            lines.append(f"    Токенов: {c.token_count}")
            preview = c.text[:250].replace("\n", " ")
            if len(c.text) > 250:
                preview += "..."
            lines.append(f"    Текст  : {preview}")
            lines.append("─" * 60)
        return "\n".join(lines)

    elif sub == "stats":
        import sys as _sys
        db_path_str = rag_state.get("db_path", "")
        _rag_dir = _get_rag_indexer_path()
        if _rag_dir not in _sys.path:
            _sys.path.insert(0, _rag_dir)

        try:
            from src.storage.index_store import IndexStore
        except ImportError:
            return "RAG-модуль не найден."

        strategy_filter = parts[2] if len(parts) > 2 else None
        if not os.path.exists(db_path_str):
            return f"Индекс не найден: {db_path_str}"
        try:
            with IndexStore(db_path_str) as store:
                if strategy_filter:
                    stats = store.get_stats(strategy_filter)
                    all_stats = {strategy_filter: stats}
                else:
                    strategies_list = store.get_all_strategies()
                    all_stats = {s: store.get_stats(s) for s in strategies_list}
        except Exception as exc:
            return f"Ошибка чтения индекса: {exc}"

        if not all_stats:
            return "[INFO] Индекс пуст."

        lines = []
        for strat, st in all_stats.items():
            lines.append(f"  {strat}: чанков={st['chunks']}, avg_tokens={st['avg_tokens']}, docs={st['documents']}")
        return "\n".join(lines)

    elif sub == "compare":
        # Если есть результаты pipeline eval — показать их таблицу
        last_results = rag_state.get("last_eval_results")
        last_evaluator = rag_state.get("last_eval_evaluator")
        if last_results and last_evaluator:
            last_evaluator.print_comparison_table(last_results)
            last_evaluator.print_timing_summary(last_results)
            return ""
        # Иначе — fallback на старый compare (стратегии индексации)
        import sys as _sys
        db_path_str = rag_state.get("db_path", "")
        _rag_dir = _get_rag_indexer_path()
        if _rag_dir not in _sys.path:
            _sys.path.insert(0, _rag_dir)
        try:
            from src.embedding.provider import EmbeddingProvider
            from src.pipeline import IndexingPipeline
        except ImportError:
            return "RAG-модуль не найден."
        if not os.path.exists(db_path_str):
            return f"Индекс не найден: {db_path_str}"
        try:
            provider = EmbeddingProvider.create("local")
            pipeline = IndexingPipeline(docs_path=".", db_path=db_path_str, embedding_provider=provider)
            pipeline.compare_strategies()
            return ""
        except Exception as exc:
            return f"Ошибка: {exc}"

    elif sub == "benchmark":
        quick = (len(parts) > 2 and parts[2] == "quick")
        return _handle_rag_benchmark(rag_state, quick=quick)

    else:
        return (
            "Использование: /rag [off|vector|bm25|hybrid|status|eval|search|stats|compare|benchmark]\n"
            "  Реранкинг: /rag rerank [none|threshold|cohere|ollama]\n"
            "  Rewrite:   /rag rewrite [on|off]\n"
            "  Цитаты:    /rag citations [on|off]\n"
            "  Данные:    /rag confidence | /rag verify\n"
            "  Оценка:    /rag eval | /rag eval full\n"
            "  Бенчмарк:  /rag benchmark [quick]"
        )


def _handle_rag_benchmark(rag_state: dict, quick: bool = False) -> str:
    """Запустить сравнительный бенчмарк local vs cloud стеков.

    Строит два отдельных RAGPipeline (из index_local.db и index_cloud.db),
    прогоняет набор вопросов и выводит пять сравнительных таблиц.

    Args:
        rag_state: Словарь состояния RAG из основного цикла.
        quick:     True — только 3 вопроса (быстрая проверка).

    Returns:
        Пустая строка (всё выводится через print).
    """
    import sys as _sys
    import os

    db_path = rag_state.get("db_path", "")
    if not db_path:
        return "RAG не инициализирован. Задайте RAG_DB_PATH в .env."

    # rag_indexer/ — один уровень выше output/
    db_dir  = os.path.dirname(os.path.abspath(db_path))
    rag_dir = os.path.normpath(os.path.join(db_dir, ".."))
    if rag_dir not in _sys.path:
        _sys.path.insert(0, rag_dir)

    # Пути к двум индексам
    local_db = os.path.join(db_dir, "index_local.db")
    cloud_db = os.path.join(db_dir, "index_cloud.db")

    try:
        from src.retrieval.benchmark import RAGBenchmark
        from src.retrieval.pipeline import RAGPipeline
        from src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever
        from src.storage.index_store import IndexStore
    except ImportError as exc:
        return f"Ошибка импорта benchmark: {exc}"

    mode_label = "quick (3 вопроса)" if quick else "full (10 вопросов + антивопросы + стабильность)"
    print(f"\n  /rag benchmark — режим: {mode_label}")
    print(f"  local db : {local_db}")
    print(f"  cloud db : {cloud_db}\n")

    # ---- Local pipeline -----------------------------------------------
    local_pipeline = None
    if os.path.exists(local_db):
        print("  [>] Инициализация local pipeline...", end=" ", flush=True)
        try:
            from src.embedding.ollama_embedder import OllamaEmbedder
            from src.retrieval.reranker import OllamaReranker

            base_url     = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            embed_model  = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            rerank_model = os.environ.get(
                "OLLAMA_RERANK_MODEL",
                os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b"),
            )
            llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")

            embedder  = OllamaEmbedder(model=embed_model, base_url=base_url)
            reranker  = OllamaReranker(model=rerank_model, base_url=base_url)
            store     = IndexStore(local_db)
            retriever = HybridRetriever(
                vector_retriever=VectorRetriever(store=store, embedder=embedder),
                bm25_retriever=BM25Retriever(store=store),
            )

            from llm_agent.infrastructure.ollama_client import OllamaHttpClient
            _llm_client = OllamaHttpClient(model=llm_model, base_url=base_url, timeout=300.0)

            def _local_llm_fn(system: str, user: str) -> str:
                from llm_agent.domain.models import ChatMessage
                msgs = []
                if system:
                    msgs.append(ChatMessage(role="system", content=system))
                msgs.append(ChatMessage(role="user", content=user))
                return _llm_client.generate(msgs).text

            local_pipeline = RAGPipeline(
                retriever=retriever,
                llm_fn=_local_llm_fn,
                reranker=reranker,
                use_structured=True,
            )
            print("OK")
        except Exception as exc:
            print(f"❌ {exc}")
    else:
        print(f"  ⚠️  index_local.db не найден: {local_db}")
        print(
            "  Создайте: cd rag_indexer && "
            "python main.py index --docs ../podkop-wiki/content "
            "--db output/index_local.db --embedder ollama"
        )

    # ---- Cloud pipeline -----------------------------------------------
    cloud_pipeline = None
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if os.path.exists(cloud_db) and qwen_key:
        print("  [>] Инициализация cloud pipeline...", end=" ", flush=True)
        try:
            from src.embedding.provider import EmbeddingProvider
            from src.retrieval.reranker import CohereReranker, ThresholdFilter

            qwen_api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            embedder  = EmbeddingProvider.create("qwen", api_key=qwen_api_key)
            store     = IndexStore(cloud_db)
            retriever = HybridRetriever(
                vector_retriever=VectorRetriever(store=store, embedder=embedder),
                bm25_retriever=BM25Retriever(store=store),
            )

            cohere_key = os.environ.get("COHERE_API_KEY", "")
            reranker   = CohereReranker() if cohere_key else ThresholdFilter(threshold=0.0)

            # Используем llm_fn агента (уже облачный провайдер)
            cloud_llm_fn = rag_state.get("llm_fn")
            if cloud_llm_fn is None:
                return (
                    "Для cloud pipeline нужен LLM. "
                    "Введите /rag hybrid сначала (инициализирует llm_fn)."
                )

            cloud_pipeline = RAGPipeline(
                retriever=retriever,
                llm_fn=cloud_llm_fn,
                reranker=reranker,
                use_structured=True,
            )
            print("OK")
        except Exception as exc:
            print(f"❌ {exc}")
    elif not os.path.exists(cloud_db):
        print(f"  ⚠️  index_cloud.db не найден: {cloud_db}")
        print(
            "  Создайте: cd rag_indexer && "
            "python main.py index --docs ../podkop-wiki/content "
            "--db output/index_cloud.db --embedder qwen"
        )
    elif not qwen_key:
        print("  ⚠️  QWEN_API_KEY / DASHSCOPE_API_KEY не задан — cloud стек пропущен.")

    if local_pipeline is None and cloud_pipeline is None:
        return "\nНи один пайплайн не создан. Проверьте конфигурацию."

    # ---- Запуск бенчмарка ---------------------------------------------
    benchmark = RAGBenchmark(
        local_pipeline=local_pipeline,
        cloud_pipeline=cloud_pipeline,
    )
    results, anti_results, stability_results = benchmark.run(quick=quick)
    benchmark.print_all_tables(results, anti_results, stability_results)
    benchmark.print_conclusions(results, anti_results, stability_results)
    return ""


def _handle_research_command(parts: list[str], mcp_state: dict) -> None:
    """Обработать /research [<запрос> | status | log | last].

    Примеры:
        /research "хочу сходить в кино в эти выходные"
        /research status
        /research log
        /research last
    """
    if not _MCP_AVAILABLE:
        print("❌ MCP: пакет mcp не установлен.")
        print("   Установите: pip install mcp")
        return

    # parts[0] == "/research"
    args = parts[1:]

    # Без аргументов — показываем помощь
    if not args:
        print("Использование:")
        print('  /research "<запрос>"  — запустить исследование')
        print('  /research status      — текущий этап и прогресс')
        print('  /research log         — журнал последнего исследования')
        print('  /research last        — финальный ответ последнего исследования')
        return

    sub = args[0].lower()

    # /research status
    if sub == "status":
        state = mcp_state.get("research_state")
        ctx = mcp_state.get("research_context")
        if state is not None and ctx is not None:
            print(f"  Задача:  {ctx.task[:80]}")
            print(f"  Этап:    {state}")
            print(f"  task_id: {ctx.task_id}")
            print(f"  Ссылок собрано: {ctx.total_links}")
            print(f"  Документов: {ctx.total_docs}")
            elapsed = ctx.elapsed
            print(f"  Время:   {elapsed:.1f}с")
        else:
            # Нет локального контекста — читаем из общей БД (Telegram bot или прошлая сессия)
            print("  (локальный контекст CLI отсутствует — читаю из общего журнала БД)\n")
            _print_journal_fallback(mcp_state, task_id="")
        return

    # /research log
    if sub == "log":
        ctx = mcp_state.get("research_context")
        journal_client = mcp_state.get("research_journal_client")
        if journal_client is None:
            try:
                from mcp_client.client import MCPClient
                from mcp_client.config import MCPServerConfig
                journal_client = MCPClient(MCPServerConfig(
                    name="journal_server", transport="stdio",
                    description="Journal", command="python",
                    args=["-m", "mcp_server.journal_server"],
                ))
                mcp_state["research_journal_client"] = journal_client
            except Exception as exc:
                print(f"Ошибка создания journal-клиента: {exc}")
                return
        task_id = ctx.task_id if ctx is not None else ""
        try:
            log = journal_client.call_tool("get_log", {"task_id": task_id})
            print(f"\n{log}\n")
        except Exception as exc:
            print(f"Ошибка получения журнала: {exc}")
        return

    # /research last
    if sub == "last":
        ctx = mcp_state.get("research_context")
        if ctx is None or not ctx.final_result:
            print("Нет сохранённого результата. Запустите: /research \"ваш запрос\"")
            return
        print(f"\n{'─' * 60}")
        print(f"Задача: {ctx.task}")
        print(f"{'─' * 60}")
        print(ctx.final_result)
        print(f"{'─' * 60}\n")
        return

    # /research "<запрос>" — запустить исследование
    # Собираем запрос из оставшихся частей
    task = " ".join(args).strip().strip('"').strip("'")
    if not task:
        print('Укажите запрос: /research "что искать"')
        return

    # Создаём клиенты для 4 серверов
    try:
        from mcp_client.client import MCPClient
        from mcp_client.config import MCPServerConfig

        def _mk(module: str, name: str) -> MCPClient:
            return MCPClient(MCPServerConfig(
                name=name, transport="stdio", description=name,
                command="python", args=["-m", module],
            ))

        servers = {
            "search":   _mk("mcp_server.search_server",   "search_server"),
            "scraper":  _mk("mcp_server.scraper_server",  "scraper_server"),
            "telegram": _mk("mcp_server.telegram_server", "telegram_server"),
            "journal":  _mk("mcp_server.journal_server",  "journal_server"),
        }
    except Exception as exc:
        print(f"❌ Ошибка создания MCP-клиентов: {exc}")
        return

    # Создаём LLM-функцию
    try:
        from mcp_server.llm_client import create_llm_fn
        llm_fn = create_llm_fn(timeout=60.0)
    except ValueError as exc:
        print(f"❌ LLM недоступен: {exc}")
        print("   Задайте QWEN_API_KEY, OPENAI_API_KEY или ANTHROPIC_API_KEY в .env")
        return
    except Exception as exc:
        print(f"❌ Ошибка инициализации LLM: {exc}")
        return

    # Запускаем оркестратор
    try:
        from orchestrator.research_orchestrator import ResearchOrchestrator
    except ImportError as exc:
        print(f"❌ orchestrator не найден: {exc}")
        print("   Убедитесь, что папка orchestrator/ существует в корне проекта.")
        return

    import os as _os
    chat_id = _os.environ.get("TELEGRAM_CHAT_ID", "")

    print(f'\n🎬 Запускаю исследование: "{task}"')
    print("   Это займёт 30-60 секунд...\n")

    orchestrator = ResearchOrchestrator(
        mcp_clients=servers,
        llm_fn=llm_fn,
        verbose=True,
    )

    try:
        result = orchestrator.run(task, chat_id=chat_id)
        # Сохраняем контекст для /research status/log/last
        mcp_state["research_state"] = str(orchestrator.state.value)
        mcp_state["research_context"] = orchestrator.context
        mcp_state["research_journal_client"] = servers["journal"]

        print(f"\n{'─' * 60}")
        print("Финальный ответ:")
        print("─" * 60)
        print(result[:3000] + ("..." if len(result) > 3000 else ""))
        print("─" * 60)
        print("\nИспользуйте /research log для полного журнала этапов.\n")
    except Exception as exc:
        print(f"\n❌ Ошибка исследования: {exc}\n")


def _handle_mcp_command(parts: list[str], mcp_state: dict) -> None:
    """Обработать /mcp <подкоманда> ...

    mcp_state — мутируемый словарь с ключами:
        config_parser: MCPConfigParser | None
        last_client: MCPClient | None
    """
    if not _MCP_AVAILABLE:
        print("❌ MCP: пакет mcp не установлен.")
        print("   Установите: pip install mcp")
        return

    config_parser: MCPConfigParser = mcp_state.get("config_parser")
    last_client: MCPClient | None = mcp_state.get("last_client")

    sub = parts[1].lower() if len(parts) > 1 else "status"

    # /mcp servers
    if sub == "servers":
        if config_parser is None:
            print("❌ MCP: конфигурация не загружена.")
            return
        try:
            servers = config_parser.load()
        except EnvironmentError as exc:
            print(exc)
            return
        if not servers:
            print("  Серверы не найдены в config/mcp-servers.md")
            return
        print(f"\n  MCP-серверы ({len(servers)} шт.):")
        for s in servers:
            print(f"    [{s.name}] transport={s.transport}")
            if s.description:
                print(f"      {s.description}")
        print()
        return

    # /mcp tools [name]
    if sub == "tools":
        server_name = parts[2] if len(parts) > 2 else None

        if server_name is None:
            # Показать инструменты из кэша
            if last_client is None or not last_client.tools:
                print("  Кэш пуст. Используйте: /mcp tools <name>")
                return
            print(last_client.get_tools_summary())
            return

        # Подключиться и получить инструменты
        if config_parser is None:
            print("❌ MCP: конфигурация не загружена.")
            return
        try:
            servers = config_parser.load()
        except EnvironmentError as exc:
            print(exc)
            return

        found = next((s for s in servers if s.name == server_name), None)
        if found is None:
            available = ", ".join(s.name for s in servers) or "(нет)"
            print(f"❌ MCP: сервер '{server_name}' не найден.")
            print(f"   Доступные серверы: {available}")
            return

        client = MCPClient(found)
        print(f"  Подключаюсь к '{found.name}'...")
        try:
            client.connect_and_list_tools()
        except RuntimeError as exc:
            print(exc)
            return

        mcp_state["last_client"] = client
        print(client.get_tools_summary())
        return

    # /mcp status
    if sub == "status":
        if last_client is None:
            print("  MCP: нет активного подключения.")
            print("  Используйте: /mcp tools <name>")
        else:
            count = len(last_client.tools)
            print(f"  MCP: последний сервер — '{last_client.config.name}', {count} инструментов.")
        return

    # /mcp call <server> <tool> [key=value ...]
    if sub == "call":
        if len(parts) < 4:
            print("Использование: /mcp call <server> <tool> [key=value ...]")
            print("Пример: /mcp call cbr_currencies get_exchange_rates")
            print("Пример: /mcp call cbr_currencies get_currency_rate currency_code=USD")
            return

        server_name = parts[2]
        tool_name = parts[3]
        raw_kwargs = parts[4:]

        # Парсинг key=value аргументов с автоматическим приведением типов
        arguments: dict = {}
        for kv in raw_kwargs:
            if "=" not in kv:
                print(f"Ошибка: аргумент '{kv}' должен быть в формате key=value")
                return
            key, _, value = kv.partition("=")
            try:
                arguments[key] = int(value)
            except ValueError:
                try:
                    arguments[key] = float(value)
                except ValueError:
                    arguments[key] = value

        if config_parser is None:
            print("❌ MCP: конфигурация не загружена.")
            return
        try:
            servers = config_parser.load()
        except EnvironmentError as exc:
            print(exc)
            return

        found = next((s for s in servers if s.name == server_name), None)
        if found is None:
            available = ", ".join(s.name for s in servers) or "(нет)"
            print(f"❌ MCP: сервер '{server_name}' не найден.")
            print(f"   Доступные серверы: {available}")
            return

        client = MCPClient(found)
        args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
        print(f"📡 Вызов: {tool_name}({args_str})")
        try:
            result = client.call_tool(tool_name, arguments)
            print(result)
        except (RuntimeError, ValueError) as exc:
            print(str(exc))
        return

    print(f"Неизвестная подкоманда MCP: '{sub}'")
    print("Доступные: /mcp servers | /mcp tools [name] | /mcp call <server> <tool> [key=value ...] | /mcp status")


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
    invariant_loader: InvariantLoader | None = None,
    mcp_state: dict | None = None,
    rag_state: dict | None = None,
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
            status = "✓" if info["available"] else "✗"
            marker = " ◄ текущий" if info["provider"] == agent.provider_name else ""
            print(
                f"    [{info['provider']:8}] {info['label']:<28} "
                f"{status}{marker}"
            )
            if not info["available"] and info.get("hint"):
                print(f"              {info['hint']}")
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

    # ---- Инварианты ----
    if command == "/invariants":
        if invariant_loader is None:
            print("Загрузчик инвариантов не подключён.")
            return current_strategy_num, True

        sub = parts[1].lower() if len(parts) > 1 else None

        if sub is None:
            print(invariant_loader.format_for_display())
            return current_strategy_num, True

        if sub == "reload":
            cats = invariant_loader.reload()
            total_req = sum(len(c.required) for c in cats)
            total_rec = sum(len(c.recommended) for c in cats)
            print(
                f"Инварианты перезагружены: {len(cats)} категорий, "
                f"{total_req} обязательных, {total_rec} рекомендуемых."
            )
            return current_strategy_num, True

        if sub == "check":
            if len(parts) < 3:
                print("Использование: /invariants check <текст>")
                return current_strategy_num, True
            text_to_check = " ".join(parts[2:])
            inv_block = invariant_loader.build_prompt_block()
            check_prompt = (
                f"Проверь следующий текст или запрос на соответствие инвариантам проекта.\n\n"
                f"{inv_block}\n\n"
                f"Текст для проверки:\n{text_to_check}\n\n"
                f"Если есть нарушения обязательных инвариантов, укажи конкретный инвариант "
                f"и объясни конфликт. Используй формат:\n"
                f"⛔ Конфликт с инвариантом: [категория] → [правило]\n"
                f"Причина ограничения: [объяснение]\n"
                f"Альтернативное решение: [предложение]\n\n"
                f"Если нарушений нет, напиши: '✅ Запрос не нарушает инварианты.'"
            )
            try:
                from llm_agent.domain.models import ChatMessage
                messages = [ChatMessage(role="user", content=check_prompt)]
                with spinner("Проверяю"):
                    response = agent._llm_client.generate(messages)
                print(f"\n{response.text.strip()}\n")
            except Exception as exc:
                print(f"Ошибка при проверке: {exc}")
            return current_strategy_num, True

        print(f"Неизвестная подкоманда: '{sub}'")
        print("Доступные: /invariants, /invariants reload, /invariants check <текст>")
        return current_strategy_num, True

    # ---- Задачи (Task Orchestrator) ----
    if command == "/task":
        if task_orchestrator is None:
            print("Task Orchestrator не подключён.")
            return current_strategy_num, True
        _handle_task_command(parts, task_orchestrator)
        return current_strategy_num, True

    # ---- MCP ----
    if command == "/mcp":
        if mcp_state is None:
            print("MCP не настроен.")
        else:
            _handle_mcp_command(parts, mcp_state)
        return current_strategy_num, True

    # ---- Конвертация валют ----
    if command == "/convert":
        if mcp_state is None:
            print("MCP не настроен.")
            return current_strategy_num, True
        _handle_convert_command(parts[1:], mcp_state)
        return current_strategy_num, True

    # ---- Новости ----
    if command == "/news":
        if mcp_state is None:
            print("MCP не настроен.")
            return current_strategy_num, True
        _handle_news_command(parts, mcp_state)
        return current_strategy_num, True

    # ---- Исследование (Orchestration MCP) ----
    if command == "/research":
        if mcp_state is None:
            print("MCP не настроен.")
            return current_strategy_num, True
        _handle_research_command(parts, mcp_state)
        return current_strategy_num, True

    # ---- RAG-поиск ----
    if command == "/rag":
        if rag_state is None:
            print("RAG не настроен.")
        else:
            # Инжектируем llm_fn при первом обращении (нужен для pipeline eval и rewrite)
            if rag_state.get("llm_fn") is None:
                rag_state["llm_fn"] = _make_agent_llm_fn(agent)
            result = _handle_rag_command(parts, rag_state)
            if result:
                print(result)
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

    # Invariant Loader инициализируется до агента, чтобы передать при создании
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    invariant_loader = InvariantLoader(os.path.join(_project_root, "config"))

    agent = StrategyAgent(
        llm_client=client,
        strategy=strategies[current_strategy_num],
        system_prompt=system_prompt,
        token_counter=token_counter,
        provider_name=provider,
        model_name=model_name,
        memory_manager=memory_manager,
        profile_manager=profile_manager,
        invariant_loader=invariant_loader,
    )

    # Task Orchestrator (тот же DB-файл)
    task_orchestrator = TaskOrchestrator(
        db_path=memory_db,
        agent=agent,
        memory_manager=memory_manager,
        config_dir=os.path.join(_project_root, "config"),
    )

    # MCP-состояние: config_parser + последний клиент (кэш)
    mcp_state: dict = {"config_parser": None, "last_client": None}
    if _MCP_AVAILABLE:
        mcp_state["config_parser"] = MCPConfigParser(
            config_path=os.path.join(_project_root, "config", "mcp-servers.md")
        )

    # RAG-состояние: режим поиска + ленивая инициализация ретривера
    _rag_db_path = os.environ.get(
        "RAG_DB_PATH",
        os.path.join(_project_root, "rag_indexer", "output", "index.db"),
    )
    rag_state: dict = {
        "mode": os.environ.get("RAG_MODE", "off"),
        "db_path": _rag_db_path,
        "strategy_filter": os.environ.get("RAG_STRATEGY_FILTER") or None,
        "top_k": int(os.environ.get("RAG_TOP_K", "5")),
        "store": None,
        "embedder": None,
        "retrievers": {},
        # Реранкинг
        "rerank_mode": os.environ.get("RERANK_MODE", "none"),
        "rewrite": os.environ.get("QUERY_REWRITE", "false").lower() in ("true", "1"),
        # llm_fn инжектируется при первом /rag eval
        "llm_fn": None,
        # Результаты последнего eval для /rag compare
        "last_eval_results": None,
        "last_eval_evaluator": None,
        # Режим цитат (День 24)
        "citations": os.environ.get("RAG_REQUIRE_CITATIONS", "false").lower() in ("true", "1"),
        "last_structured_response": None,
        "last_confidence": None,
    }

    _inv_cats = invariant_loader.categories
    _inv_req = sum(len(c.required) for c in _inv_cats)
    _inv_rec = sum(len(c.recommended) for c in _inv_cats)

    print("=" * 62)
    print("  LLM-агент: стратегии контекста + выбор провайдера")
    print("=" * 62)
    print(f"\n  Провайдер : {provider}  ({PROVIDER_LABELS.get(provider, '')})")
    print(f"  Модель    : {model_name}")
    print(f"  Стратегия : [{current_strategy_num}] {STRATEGY_NAMES[current_strategy_num]}")
    print(f"  Инварианты: {_inv_req} обязательных, {_inv_rec} рекомендуемых (/invariants)")
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
            rag_info = f"|RAG:{rag_state['mode']}" if rag_state.get("mode", "off") != "off" else ""
            prefix = f"[{agent.provider_name}|{current_strategy_num}{profile_info}{task_info}{rag_info}]{branch_info}"
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
                invariant_loader=invariant_loader,
                mcp_state=mcp_state,
                rag_state=rag_state,
            )
            if handled:
                continue

        try:
            with spinner():
                if task_orchestrator.has_active_task:
                    reply = task_orchestrator.handle_message(user_input)
                elif rag_state.get("mode", "off") != "off":
                    reply = _run_rag_query(user_input, rag_state, agent)
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
