"""
Файловый ассистент — ReAct-агент для работы с проектом.

Запуск:
  python tools/file_assistant.py
  python tools/file_assistant.py --task "найди все использования компонента Editor"
  python tools/file_assistant.py --task "проверь src/ на соответствие инвариантам" --dry-run
  python tools/file_assistant.py --root ../documaker
  python tools/file_assistant.py --task "..." --provider qwen --verbose

Флаги:
  --task      задача (если не задана — интерактивный режим)
  --root      корневая директория проекта (default: ../documaker)
  --dry-run   только анализ, запретить write_file и apply_diff
  --provider  llm провайдер: claude, qwen, openai, ollama (default: auto)
  --verbose   показывать каждый шаг агента
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from llm_agent.agent_loop import AgentLoop, AgentResult
from llm_agent.file_tools import FileSystemToolkit
from llm_agent.infrastructure.llm_factory import build_client, current_provider_from_env

_CONFIG_PATH = _ROOT / "config" / "file-assistant.md"

# ANSI
_R      = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"


# ─── Конфиг ──────────────────────────────────────────────────────────────────

def _load_allowed_dirs(config_path: Path, root_override: str | None = None) -> list[str]:
    """
    Загрузить разрешённые директории из config/file-assistant.md.
    Если задан root_override — он используется как единственная директория.
    """
    if root_override:
        return [str(Path(root_override).resolve())]

    if not config_path.exists():
        return [str(_ROOT)]

    content = config_path.read_text(encoding="utf-8")
    allowed: list[str] = []
    in_section = False

    for line in content.splitlines():
        if "Разрешённые директории" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("##"):
                break
            if line.startswith("- "):
                # "- ../path/to/dir        — описание"
                raw = line[2:].split("—")[0].split("  ")[0].strip()
                if raw:
                    abs_path = str((_ROOT / raw).resolve())
                    allowed.append(abs_path)

    return allowed or [str(_ROOT)]


# ─── Вывод ───────────────────────────────────────────────────────────────────

def _print_header(root: str, dry_run: bool, provider: str) -> None:
    mode = f"{_YELLOW}только чтение{_R}" if dry_run else f"{_GREEN}полный{_R}"
    print(f"\n{_BOLD}Ассистент файлов проекта{_R}")
    print(f"Корень: {_CYAN}{root}{_R} | Режим: {mode} | Провайдер: {_CYAN}{provider}{_R}")
    print("─" * 60)


def _print_result(result: AgentResult) -> None:
    bar = "━" * 60
    print(f"\n{bar}")
    print(f"{_BOLD}Результат{_R}")
    print(bar)
    print(f"\n{result.summary}")

    if result.files_read:
        print(f"\nПрочитано файлов: {_CYAN}{len(result.files_read)}{_R}")
        for f in result.files_read[:10]:
            print(f"  • {f}")
        if len(result.files_read) > 10:
            print(f"  {_DIM}... и ещё {len(result.files_read) - 10}{_R}")
    else:
        print(f"{_DIM}Файлы не читались.{_R}")

    if result.files_modified:
        print(f"\nИзменено файлов: {_YELLOW}{len(result.files_modified)}{_R}")
        for f in result.files_modified:
            print(f"  • {f}")
    else:
        print(f"{_DIM}Изменений не вносилось.{_R}")

    print(f"\nШагов: {result.steps} | Время: {result.elapsed_sec:.1f} сек")

    if result.warning:
        print(f"\n{_YELLOW}⚠️  {result.warning}{_R}")


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Файловый ассистент — ReAct-агент для работы с проектом",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", help="Задача для агента")
    parser.add_argument(
        "--root",
        default=str((_ROOT / ".." / "documaker").resolve()),
        help="Корневая директория проекта (default: ../documaker)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только анализ, без записи файлов",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM провайдер: claude, qwen, openai, ollama",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Показывать каждый шаг агента",
    )
    args = parser.parse_args()

    provider = args.provider or current_provider_from_env()

    try:
        llm_client = build_client(provider)
    except ValueError as e:
        print(f"{_RED}Ошибка инициализации LLM ({provider}): {e}{_R}", file=sys.stderr)
        sys.exit(1)

    allowed_dirs = _load_allowed_dirs(_CONFIG_PATH, root_override=args.root)

    # Убедимся что root сам входит в allowed (на случай --root без конфига)
    root_real = str(Path(args.root).resolve())
    if root_real not in allowed_dirs:
        allowed_dirs.insert(0, root_real)

    toolkit = FileSystemToolkit(
        allowed_dirs=allowed_dirs,
        base_dir=args.root,
        dry_run=args.dry_run,
    )
    loop = AgentLoop(verbose=args.verbose)

    _print_header(args.root, args.dry_run, provider)

    if args.task:
        result = loop.run(task=args.task, toolkit=toolkit, llm_client=llm_client)
        _print_result(result)
        return

    # ── Интерактивный режим ──────────────────────────────────────────────────
    print(f"\nРежим: интерактивный. Введите задачу или {_CYAN}quit{_R} для выхода.\n")
    while True:
        try:
            task = input("Задача: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "/quit", "q"):
            print("Выход.")
            break

        result = loop.run(task=task, toolkit=toolkit, llm_client=llm_client)
        _print_result(result)
        print()


if __name__ == "__main__":
    main()
