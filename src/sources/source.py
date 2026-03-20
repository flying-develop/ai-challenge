"""Абстрактный интерфейс источника документации.

Три реализации:
    LocalMarkdownSource  — рекурсивное чтение .md файлов из директории
    URLDocSource         — загрузка документации по URL (обход ссылок)
    GitHubRepoSource     — клонирование GitHub репозитория

После fetch() документы проходят стандартный пайплайн:
    RawDocument-ы → IndexingPipeline → ChunkingStrategy → IndexStore
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RawDocument:
    """Документ в «сыром» виде до разбивки на чанки.

    Attributes:
        content:     Текстовое содержимое (Markdown).
        source_path: Путь к файлу или URL страницы.
        title:       Заголовок (из frontmatter, <title> или имени файла).
    """

    content: str
    source_path: str
    title: str


class DocumentSource(ABC):
    """Абстрактный источник документации.

    Каждый источник реализует метод fetch(), который возвращает список
    RawDocument-ов для последующей индексации.
    """

    @abstractmethod
    def fetch(self) -> list[RawDocument]:
        """Загрузить все документы из источника.

        Returns:
            Список RawDocument-ов для индексации.
        """
        ...

    @property
    @abstractmethod
    def source_description(self) -> str:
        """Человекочитаемое описание источника (для /index status)."""
        ...
