"""Источник документации: GitHub репозиторий.

Клонирует репозиторий во временную директорию,
рекурсивно находит .md файлы, возвращает RawDocument-ы.

После загрузки временная директория удаляется.

Конфигурация (.env):
    INDEX_GITHUB_BRANCH — ветка для клонирования (default: main)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .source import DocumentSource, RawDocument
from .local_source import LocalMarkdownSource


class GitHubRepoSource(DocumentSource):
    """Источник документации: клонирование GitHub репозитория.

    Клонирует репозиторий с заданной веткой во временную директорию,
    находит все .md файлы и возвращает их как RawDocument-ы через
    LocalMarkdownSource.

    Args:
        repo_url: URL репозитория (https://github.com/user/repo).
        branch:   Ветка для клонирования (из env INDEX_GITHUB_BRANCH).
    """

    def __init__(self, repo_url: str, branch: str | None = None) -> None:
        self.repo_url = repo_url
        self.branch = branch or os.environ.get("INDEX_GITHUB_BRANCH", "main")

    def fetch(self) -> list[RawDocument]:
        """Клонировать репозиторий и загрузить .md файлы.

        Шаги:
            1. git clone --depth=1 --branch {branch} {url} во временную директорию
            2. Найти все .md файлы через LocalMarkdownSource
            3. Удалить временную директорию

        Returns:
            Список RawDocument-ов с контентом из .md файлов.

        Raises:
            RuntimeError: Если git не установлен или клонирование не удалось.
        """
        print(f"[GitHubRepoSource] Клонирование: {self.repo_url} (ветка: {self.branch})")

        tmp_dir = tempfile.mkdtemp(prefix="support_bot_repo_")
        try:
            self._clone(tmp_dir)
            print(f"  Клонирование завершено в {tmp_dir}")

            # Ищем директорию с .md файлами
            docs_dir = self._find_docs_dir(tmp_dir)
            print(f"  Директория документации: {docs_dir}")

            source = LocalMarkdownSource(docs_dir)
            raw_docs = source.fetch()

            # Перезаписываем source_path — указываем URL вместо temp пути
            reponame = self.repo_url.rstrip("/").split("/")[-1]
            docs_relative = Path(docs_dir).relative_to(tmp_dir)
            result = []
            for doc in raw_docs:
                result.append(RawDocument(
                    content=doc.content,
                    source_path=f"{self.repo_url}/blob/{self.branch}/{docs_relative}/{doc.source_path}",
                    title=doc.title,
                ))

            print(f"[GitHubRepoSource] Загружено {len(result)} документов")
            return result

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  Временная директория удалена")

    def _clone(self, target_dir: str) -> None:
        """Выполнить git clone во временную директорию."""
        cmd = [
            "git", "clone",
            "--depth=1",
            "--branch", self.branch,
            "--single-branch",
            self.repo_url,
            target_dir,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"git clone завершился с ошибкой:\n{result.stderr}"
                )
        except FileNotFoundError:
            raise RuntimeError("git не найден. Установите git для использования GitHubRepoSource.")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Тайм-аут при клонировании {self.repo_url}")

    def _find_docs_dir(self, repo_dir: str) -> str:
        """Найти директорию с документацией в репозитории.

        Проверяет стандартные места: docs/, content/, wiki/, root.

        Returns:
            Путь к директории с .md файлами.
        """
        repo_path = Path(repo_dir)
        candidates = ["docs", "content", "wiki", "documentation", "doc", "."]

        for candidate in candidates:
            candidate_path = repo_path / candidate
            if not candidate_path.is_dir():
                continue
            # Проверяем что там есть .md файлы
            md_files = list(candidate_path.rglob("*.md"))
            if md_files:
                return str(candidate_path)

        # Fallback: корень репозитория
        return repo_dir

    @property
    def source_description(self) -> str:
        return f"github:{self.repo_url}@{self.branch}"
