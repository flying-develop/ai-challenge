"""InvariantLoader: загрузчик инвариантов из MD-файлов.

Инварианты — ограничения, которые ассистент не имеет права нарушать при генерации
ответов. Хранятся в .md-файлах в директории config/invariants/.

Формат MD-файла:
  # Заголовок категории

  ## Обязательные
  - Правило 1
  - Правило 2

  ## Рекомендуемые
  - Правило 3

Две категории:
  - Обязательные: жёсткие инварианты, нарушение = отказ выполнить запрос.
  - Рекомендуемые: мягкие, нарушение = предупреждение, но выполнение.

Поддерживает hot-reload: перечитывает файлы без перезапуска агента.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InvariantCategory:
    """Категория инвариантов, загруженная из одного MD-файла.

    Attributes:
        name: Имя категории (имя файла без расширения).
        title: Заголовок из # в MD-файле.
        required: Список обязательных правил (нарушение = отказ).
        recommended: Список рекомендуемых правил (нарушение = предупреждение).
    """

    name: str
    title: str
    required: list[str] = field(default_factory=list)
    recommended: list[str] = field(default_factory=list)


class InvariantLoader:
    """Загрузчик инвариантов из MD-файлов.

    Читает все .md-файлы из config_dir/invariants/, парсит их в
    структурированный формат и формирует блок для system prompt.

    Parameters:
        config_dir: Путь к директории config/ проекта.
    """

    # Раздел обязательных инвариантов (различные варианты заголовка)
    _REQUIRED_PATTERN = re.compile(r"^##\s+Обязательные", re.IGNORECASE)
    # Раздел рекомендуемых инвариантов
    _RECOMMENDED_PATTERN = re.compile(r"^##\s+Рекомендуемые", re.IGNORECASE)
    # Пункт списка: "- текст" или "* текст"
    _ITEM_PATTERN = re.compile(r"^\s*[-*]\s+(.+)$")
    # Заголовок первого уровня
    _TITLE_PATTERN = re.compile(r"^#\s+(.+)$")

    def __init__(self, config_dir: str | Path) -> None:
        self._config_dir = Path(config_dir)
        self._invariants_dir = self._config_dir / "invariants"
        self._categories: list[InvariantCategory] = []
        self.load()

    # ------------------------------------------------------------------
    # Загрузка и парсинг
    # ------------------------------------------------------------------

    def load(self) -> list[InvariantCategory]:
        """Прочитать все .md-файлы из config/invariants/ и распарсить их.

        Returns:
            Список категорий инвариантов.
        """
        self._categories = []

        if not self._invariants_dir.exists():
            return self._categories

        for md_file in sorted(self._invariants_dir.glob("*.md")):
            try:
                category = self._parse_file(md_file)
                self._categories.append(category)
            except Exception:
                # Не даём одному битому файлу сломать загрузку остальных
                pass

        return self._categories

    def reload(self) -> list[InvariantCategory]:
        """Hot-reload: перечитать файлы без перезапуска агента.

        Returns:
            Обновлённый список категорий инвариантов.
        """
        return self.load()

    def _parse_file(self, path: Path) -> InvariantCategory:
        """Распарсить один MD-файл в InvariantCategory.

        Парсим вручную (split/regex), без markdown-библиотек — это инвариант проекта.

        Args:
            path: Путь к .md-файлу.

        Returns:
            Распарсенная категория.
        """
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        name = path.stem  # имя файла без расширения
        title = name      # по умолчанию — имя файла

        required: list[str] = []
        recommended: list[str] = []

        # Текущий раздел: None, "required" или "recommended"
        current_section: str | None = None

        for line in lines:
            # Заголовок первого уровня → заголовок категории
            title_match = self._TITLE_PATTERN.match(line)
            if title_match:
                title = title_match.group(1).strip()
                current_section = None
                continue

            # Заголовок раздела ## Обязательные
            if self._REQUIRED_PATTERN.match(line):
                current_section = "required"
                continue

            # Заголовок раздела ## Рекомендуемые
            if self._RECOMMENDED_PATTERN.match(line):
                current_section = "recommended"
                continue

            # Любой другой заголовок ## завершает текущий раздел
            if line.startswith("##"):
                current_section = None
                continue

            # Пункт списка
            item_match = self._ITEM_PATTERN.match(line)
            if item_match and current_section:
                rule = item_match.group(1).strip()
                if current_section == "required":
                    required.append(rule)
                elif current_section == "recommended":
                    recommended.append(rule)

        return InvariantCategory(
            name=name,
            title=title,
            required=required,
            recommended=recommended,
        )

    # ------------------------------------------------------------------
    # Доступ к данным
    # ------------------------------------------------------------------

    @property
    def categories(self) -> list[InvariantCategory]:
        """Список загруженных категорий."""
        return list(self._categories)

    def get_all_required(self) -> list[tuple[str, str]]:
        """Все обязательные инварианты.

        Returns:
            Список пар (заголовок_категории, правило).
        """
        result = []
        for cat in self._categories:
            for rule in cat.required:
                result.append((cat.title, rule))
        return result

    def get_all_recommended(self) -> list[tuple[str, str]]:
        """Все рекомендуемые инварианты.

        Returns:
            Список пар (заголовок_категории, правило).
        """
        result = []
        for cat in self._categories:
            for rule in cat.recommended:
                result.append((cat.title, rule))
        return result

    def is_empty(self) -> bool:
        """Вернуть True если инварианты не загружены или все списки пусты."""
        return all(
            not cat.required and not cat.recommended
            for cat in self._categories
        )

    # ------------------------------------------------------------------
    # Формирование блока для system prompt
    # ------------------------------------------------------------------

    def build_prompt_block(self) -> str:
        """Сформировать блок <INVARIANTS> для включения в system prompt.

        Блок содержит все обязательные и рекомендуемые инварианты.
        Встраивается в system prompt перед остальным содержимым, чтобы
        модель учитывала ограничения при каждом ответе.

        Returns:
            Строка с блоком инвариантов или пустая строка если нет инвариантов.
        """
        if self.is_empty():
            return ""

        required_lines: list[str] = []
        recommended_lines: list[str] = []

        for cat in self._categories:
            for rule in cat.required:
                required_lines.append(f"  [{cat.title}] {rule}")
            for rule in cat.recommended:
                recommended_lines.append(f"  [{cat.title}] {rule}")

        parts = [
            "<INVARIANTS>",
            "Ты ОБЯЗАН соблюдать следующие инварианты. "
            "Нарушение обязательных инвариантов ЗАПРЕЩЕНО. "
            "При каждом ответе, содержащем код или архитектурное предложение, "
            "проверяй соответствие инвариантам.",
        ]

        if required_lines:
            parts.append("\n## Обязательные (нарушение = отказ с объяснением)")
            parts.extend(required_lines)

        if recommended_lines:
            parts.append("\n## Рекомендуемые (предупреждение при отклонении)")
            parts.extend(recommended_lines)

        parts.append("</INVARIANTS>")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Отображение для CLI
    # ------------------------------------------------------------------

    def format_for_display(self) -> str:
        """Отформатировать инварианты для вывода в CLI.

        Returns:
            Читаемая строка со всеми инвариантами по категориям.
        """
        if not self._categories:
            return "Инварианты не загружены. Проверьте директорию config/invariants/"

        lines = [f"\n  Инварианты ({self._invariants_dir}):"]

        for cat in self._categories:
            lines.append(f"\n  [{cat.name}] {cat.title}")

            if cat.required:
                lines.append("    Обязательные:")
                for rule in cat.required:
                    lines.append(f"      ⛔ {rule}")

            if cat.recommended:
                lines.append("    Рекомендуемые:")
                for rule in cat.recommended:
                    lines.append(f"      ⚠  {rule}")

        total_req = sum(len(c.required) for c in self._categories)
        total_rec = sum(len(c.recommended) for c in self._categories)
        lines.append(
            f"\n  Итого: {total_req} обязательных, {total_rec} рекомендуемых "
            f"в {len(self._categories)} категориях."
        )

        return "\n".join(lines)
