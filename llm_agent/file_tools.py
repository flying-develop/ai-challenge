"""
Набор инструментов для работы с файлами.
Все операции ограничены разрешёнными директориями из конфига.
"""
from __future__ import annotations

import difflib
import fnmatch
import glob as glob_module
import os
import shutil
from datetime import datetime
from pathlib import Path

# Директории, которые всегда пропускаются
_SKIP_DIRS: frozenset[str] = frozenset(
    {"node_modules", ".git", "dist", "build", "__pycache__", ".next", ".venv", "venv", ".tox"}
)

TOOLS_SCHEMA: list[dict] = [
    {
        "name": "list_files",
        "description": "Получить список файлов по glob-паттерну. "
                       "Используй для исследования структуры проекта.",
        "parameters": {
            "pattern": "glob-паттерн, например 'src/**/*.tsx' или 'docs/*.md'",
            "base_dir": "корневая директория (опционально)",
        },
    },
    {
        "name": "read_file",
        "description": "Прочитать содержимое файла. "
                       "Используй когда нужно проанализировать конкретный файл.",
        "parameters": {
            "path": "путь к файлу",
            "start_line": "начальная строка (опционально, default 1)",
            "end_line": "конечная строка (опционально, default — весь файл)",
        },
    },
    {
        "name": "search_in_files",
        "description": "Поиск строки или паттерна во всех файлах директории. "
                       "Используй для поиска использований компонента, функции, импорта.",
        "parameters": {
            "query": "строка поиска",
            "directory": "директория для поиска",
            "file_pattern": "фильтр файлов, например '*.tsx' (опционально)",
            "case_sensitive": "учитывать регистр (default: false)",
        },
    },
    {
        "name": "write_file",
        "description": "Создать или перезаписать файл. "
                       "Используй для генерации новых файлов (README, CHANGELOG, ADR).",
        "parameters": {
            "path": "путь к файлу",
            "content": "содержимое файла",
            "create_dirs": "создать директории если не существуют (default: true)",
        },
    },
    {
        "name": "apply_diff",
        "description": "Применить изменения к существующему файлу. "
                       "Используй вместо write_file когда нужно частично обновить файл.",
        "parameters": {
            "path": "путь к файлу",
            "old_text": "фрагмент который нужно заменить (точное совпадение)",
            "new_text": "новый текст",
        },
    },
    {
        "name": "show_diff",
        "description": "Показать что изменится если применить правки, без реального изменения файла. "
                       "Используй перед apply_diff для проверки.",
        "parameters": {
            "path": "путь к файлу",
            "old_text": "заменяемый фрагмент",
            "new_text": "новый текст",
        },
    },
    {
        "name": "finish",
        "description": "Завершить выполнение задачи и вернуть итоговый отчёт пользователю.",
        "parameters": {
            "summary": "краткое резюме того что было сделано",
            "files_read": "список прочитанных файлов",
            "files_modified": "список изменённых файлов (пустой если только анализ)",
        },
    },
]


class FileSystemToolkit:
    """
    Реализация файловых операций.
    Все пути проходят валидацию против allowed_dirs.
    """

    MAX_FILE_SIZE_BYTES: int = 100_000   # 100 KB
    MAX_SEARCH_RESULTS: int = 50
    _BINARY_CHECK_BYTES: int = 1024

    def __init__(
        self,
        allowed_dirs: list[str],
        base_dir: str = ".",
        dry_run: bool = False,
    ) -> None:
        """
        allowed_dirs: список разрешённых корневых директорий.
        base_dir: корень для разрешения относительных путей.
        dry_run: если True — write_file и apply_diff бросают PermissionError.
        """
        self._base_dir = os.path.realpath(base_dir)
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs]
        if not self._allowed_dirs:
            self._allowed_dirs = [self._base_dir]
        self._dry_run = dry_run

    # ─── Публичный интерфейс ──────────────────────────────────────────────────

    def list_files(self, pattern: str, base_dir: str = ".") -> list[dict]:
        """
        glob по паттерну.
        Возвращает: [{path, size_bytes, modified_at, extension}]
        Пропускает: node_modules, .git, dist, build, __pycache__
        """
        base = self._resolve(base_dir)
        self._validate_path(base)

        full_pattern = os.path.join(base, pattern)
        results: list[dict] = []

        for match in sorted(glob_module.glob(full_pattern, recursive=True)):
            p = Path(match)
            if not p.is_file():
                continue
            # Skip forbidden dirs anywhere in the path
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            try:
                st = p.stat()
                results.append(
                    {
                        "path": os.path.relpath(match, self._base_dir),
                        "size_bytes": st.st_size,
                        "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(
                            timespec="seconds"
                        ),
                        "extension": p.suffix,
                    }
                )
            except OSError:
                continue

        return results

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict:
        """
        Читать файл.
        Если файл > MAX_FILE_SIZE_BYTES → первые 500 строк + предупреждение.
        Если бинарный → ошибка.
        """
        abs_path = self._resolve(path)
        self._validate_path(abs_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Файл не найден: {path!r}")

        # Binary check
        with open(abs_path, "rb") as fb:
            chunk = fb.read(self._BINARY_CHECK_BYTES)
        if b"\x00" in chunk:
            raise ValueError(f"Файл является бинарным, чтение невозможно: {path!r}")

        size = os.path.getsize(abs_path)
        warning: str | None = None

        with open(abs_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)

        if size > self.MAX_FILE_SIZE_BYTES and end_line is None:
            end_line = 500
            warning = f"Файл большой ({size / 1024:.1f} KB > 100 KB). Показаны строки 1–500."

        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line is not None else total_lines

        selected = all_lines[start_idx:end_idx]
        rel_path = os.path.relpath(abs_path, self._base_dir)

        result: dict = {
            "path": rel_path,
            "content": "".join(selected),
            "total_lines": total_lines,
            "shown_lines": f"{start_idx + 1}–{start_idx + len(selected)}",
            "encoding": "utf-8",
        }
        if warning:
            result["warning"] = warning
        return result

    def search_in_files(
        self,
        query: str,
        directory: str,
        file_pattern: str = "*",
        case_sensitive: bool = False,
    ) -> list[dict]:
        """
        Текстовый поиск по файлам (str.find / str.lower).
        Возвращает: [{file_path, line_number, line_content, context_before, context_after}]
        Максимум MAX_SEARCH_RESULTS результатов.
        """
        abs_dir = self._resolve(directory)
        self._validate_path(abs_dir)

        if not os.path.isdir(abs_dir):
            raise NotADirectoryError(f"Директория не найдена: {directory!r}")

        needle = query if case_sensitive else query.lower()
        results: list[dict] = []

        for root, dirs, files in os.walk(abs_dir):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]

            for filename in sorted(files):
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue

                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "rb") as fb:
                        if b"\x00" in fb.read(512):
                            continue
                    with open(filepath, encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                except OSError:
                    continue

                for i, line in enumerate(lines):
                    haystack = line if case_sensitive else line.lower()
                    if needle in haystack:
                        rel = os.path.relpath(filepath, self._base_dir)
                        results.append(
                            {
                                "file_path": rel,
                                "line_number": i + 1,
                                "line_content": line.rstrip("\n"),
                                "context_before": lines[i - 1].rstrip("\n") if i > 0 else "",
                                "context_after": lines[i + 1].rstrip("\n") if i < len(lines) - 1 else "",
                            }
                        )
                        if len(results) >= self.MAX_SEARCH_RESULTS:
                            return results

        return results

    def write_file(
        self,
        path: str,
        content: str,
        create_dirs: bool = True,
    ) -> dict:
        """
        Создать или перезаписать файл.
        Если файл существует — создать backup: path + '.bak'
        """
        if self._dry_run:
            raise PermissionError(
                "Запись файлов отключена в режиме --dry-run. "
                "Используйте show_diff для просмотра изменений."
            )

        abs_path = self._resolve(path)
        self._validate_path(abs_path)

        if create_dirs:
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

        backup_created = False
        if os.path.exists(abs_path):
            shutil.copy2(abs_path, abs_path + ".bak")
            backup_created = True

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        rel_path = os.path.relpath(abs_path, self._base_dir)
        return {
            "path": rel_path,
            "bytes_written": len(content.encode("utf-8")),
            "backup_created": backup_created,
        }

    def apply_diff(self, path: str, old_text: str, new_text: str) -> dict:
        """
        Найти old_text в файле и заменить на new_text.
        Если не найден → ошибка. Если несколько → заменить первое + предупреждение.
        """
        if self._dry_run:
            raise PermissionError(
                "Изменение файлов отключено в режиме --dry-run. "
                "Используйте show_diff для просмотра изменений."
            )

        abs_path = self._resolve(path)
        self._validate_path(abs_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Файл не найден: {path!r}")

        with open(abs_path, encoding="utf-8", errors="replace") as f:
            original = f.read()

        if old_text not in original:
            raise ValueError(
                f"Фрагмент для замены не найден в файле {path!r}.\n"
                "Убедитесь, что текст совпадает точно (включая пробелы и переносы строк)."
            )

        count = original.count(old_text)
        warning: str | None = None
        if count > 1:
            warning = f"Фрагмент встречается {count} раз в файле. Заменено первое вхождение."

        shutil.copy2(abs_path, abs_path + ".bak")
        new_content = original.replace(old_text, new_text, 1)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        rel_path = os.path.relpath(abs_path, self._base_dir)
        result: dict = {
            "path": rel_path,
            "replacements_made": 1,
            "backup_created": True,
        }
        if warning:
            result["warning"] = warning
        return result

    def show_diff(self, path: str, old_text: str, new_text: str) -> str:
        """
        Показать unified diff без изменения файла.
        """
        abs_path = self._resolve(path)
        self._validate_path(abs_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Файл не найден: {path!r}")

        with open(abs_path, encoding="utf-8", errors="replace") as f:
            original = f.read()

        if old_text not in original:
            return f"Фрагмент не найден в файле {path!r}. Diff невозможен."

        modified = original.replace(old_text, new_text, 1)
        rel_path = os.path.relpath(abs_path, self._base_dir)

        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"{rel_path} (до)",
                tofile=f"{rel_path} (после)",
                lineterm="",
            )
        )

        if not diff_lines:
            return "Изменений нет (old_text совпадает с new_text)."

        return "".join(diff_lines)

    # ─── Внутренние методы ────────────────────────────────────────────────────

    def _resolve(self, path: str) -> str:
        """Разрешить путь: если относительный — относительно base_dir."""
        if not os.path.isabs(path):
            path = os.path.join(self._base_dir, path)
        return os.path.realpath(path)

    def _validate_path(self, path: str) -> str:
        """
        Проверить что path находится внутри allowed_dirs.
        Использует os.path.realpath для защиты от ../../../
        """
        real = os.path.realpath(path)
        for allowed in self._allowed_dirs:
            if real == allowed or real.startswith(allowed + os.sep):
                return real
        friendly_dirs = [os.path.relpath(d) for d in self._allowed_dirs]
        raise ValueError(
            f"Путь {path!r} находится за пределами разрешённых директорий.\n"
            f"Разрешено: {friendly_dirs}"
        )
