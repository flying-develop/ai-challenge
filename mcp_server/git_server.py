"""MCP-сервер для работы с git-репозиторием проекта.

Транспорт: stdio (FastMCP).
Инструменты:
    - get_current_branch
    - get_recent_commits
    - list_changed_files
    - get_file_diff
"""

import subprocess
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment]


class _DummyMCP:
    """Fallback, если пакет mcp не установлен."""

    def tool(self):
        def decorator(func):
            return func
        return decorator

    def run(self) -> None:
        raise SystemExit("Пакет 'mcp' не установлен. Выполните: pip install mcp")


mcp = FastMCP("Git Server") if FastMCP is not None else _DummyMCP()


def _run_git(repo_path: str, args: list[str]) -> tuple[bool, str]:
    """Выполнить git-команду и вернуть (ok, stdout_or_stderr)."""
    repo_dir = Path(repo_path).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        stderr = getattr(exc, "stderr", "") or str(exc)
        return False, stderr.strip()


@mcp.tool()
def get_current_branch(repo_path: str = ".") -> str:
    """
    Получить текущую ветку git.

    Использует: subprocess + git rev-parse --abbrev-ref HEAD
    При ошибке (не git репо) — вернуть "unknown (not a git repo)"
    """
    ok, output = _run_git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    if not ok or not output:
        return "unknown (not a git repo)"
    return output


@mcp.tool()
def get_recent_commits(repo_path: str = ".", limit: int = 10) -> list[dict]:
    """
    Последние N коммитов.

    Формат: git log --format="%H|%s|%an|%ad" --date=short -n {limit}
    Вернуть список: [{hash, message, author, date}]
    """
    safe_limit = max(1, min(int(limit), 50))
    ok, output = _run_git(
        repo_path,
        ["log", '--format=%H|%s|%an|%ad', "--date=short", "-n", str(safe_limit)],
    )
    if not ok or not output:
        return []

    commits: list[dict] = []
    for line in output.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        commit_hash, message, author, date = parts
        commits.append(
            {
                "hash": commit_hash,
                "message": message,
                "author": author,
                "date": date,
            }
        )
    return commits


@mcp.tool()
def list_changed_files(repo_path: str = ".") -> dict:
    """
    Изменённые файлы в рабочей директории.

    git status --porcelain
    Вернуть: {staged: [...], unstaged: [...], untracked: [...]}
    """
    ok, output = _run_git(repo_path, ["status", "--porcelain"])
    result = {"staged": [], "unstaged": [], "untracked": []}
    if not ok or not output:
        return result

    for line in output.splitlines():
        if len(line) < 4:
            continue
        staged_status = line[0]
        unstaged_status = line[1]
        path = line[3:].strip()

        if staged_status == "?" and unstaged_status == "?":
            result["untracked"].append(path)
            continue
        if staged_status != " ":
            result["staged"].append(path)
        if unstaged_status != " ":
            result["unstaged"].append(path)

    return result


@mcp.tool()
def get_file_diff(repo_path: str = ".", file_path: str = "") -> str:
    """
    Diff конкретного файла.

    git diff HEAD -- {file_path}
    Ограничить вывод: первые 100 строк diff.
    При пустом file_path — вернуть git diff HEAD (все изменения, 200 строк).
    """
    args = ["diff", "HEAD"]
    line_limit = 200
    if file_path.strip():
        args.extend(["--", file_path.strip()])
        line_limit = 100

    ok, output = _run_git(repo_path, args)
    if not ok:
        return output or "git diff failed"
    if not output:
        return "No changes"
    return "\n".join(output.splitlines()[:line_limit])


if __name__ == "__main__":
    mcp.run()
