"""tests/test_invariants.py — тесты для InvariantLoader.

Проверяем:
  - Загрузку MD-файлов из директории
  - Парсинг заголовков, обязательных и рекомендуемых правил
  - Формирование блока для system prompt
  - Hot-reload (перечитывание без перезапуска)
  - Граничные случаи: пустой файл, нет директории, частичный MD
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_agent.core.invariant_loader import InvariantCategory, InvariantLoader


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Временная директория config/ с поддиректорией invariants/."""
    inv_dir = tmp_path / "config" / "invariants"
    inv_dir.mkdir(parents=True)
    return tmp_path / "config"


def write_md(config_dir: Path, filename: str, content: str) -> Path:
    """Вспомогательная функция: записать MD-файл в config/invariants/."""
    path = config_dir / "invariants" / filename
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Тесты парсинга одного файла
# ---------------------------------------------------------------------------


class TestParsing:
    def test_parses_title(self, tmp_config: Path) -> None:
        write_md(tmp_config, "test.md", "# Мой заголовок\n\n## Обязательные\n- Правило 1\n")
        loader = InvariantLoader(tmp_config)
        assert len(loader.categories) == 1
        assert loader.categories[0].title == "Мой заголовок"

    def test_parses_required_rules(self, tmp_config: Path) -> None:
        write_md(
            tmp_config,
            "test.md",
            "# Заголовок\n\n## Обязательные\n- Правило A\n- Правило B\n",
        )
        loader = InvariantLoader(tmp_config)
        cat = loader.categories[0]
        assert cat.required == ["Правило A", "Правило B"]
        assert cat.recommended == []

    def test_parses_recommended_rules(self, tmp_config: Path) -> None:
        write_md(
            tmp_config,
            "test.md",
            "# Заголовок\n\n## Рекомендуемые\n- Рек. A\n* Рек. B\n",
        )
        loader = InvariantLoader(tmp_config)
        cat = loader.categories[0]
        assert cat.required == []
        assert cat.recommended == ["Рек. A", "Рек. B"]

    def test_parses_both_sections(self, tmp_config: Path) -> None:
        content = (
            "# Стек\n\n"
            "## Обязательные\n"
            "- Python 3.11+\n"
            "- SQLite only\n\n"
            "## Рекомендуемые\n"
            "- Предпочитать stdlib\n"
        )
        write_md(tmp_config, "stack.md", content)
        loader = InvariantLoader(tmp_config)
        cat = loader.categories[0]
        assert cat.name == "stack"
        assert len(cat.required) == 2
        assert len(cat.recommended) == 1

    def test_category_name_is_stem(self, tmp_config: Path) -> None:
        write_md(tmp_config, "business-rules.md", "# Title\n\n## Обязательные\n- R\n")
        loader = InvariantLoader(tmp_config)
        assert loader.categories[0].name == "business-rules"

    def test_title_defaults_to_stem_when_no_h1(self, tmp_config: Path) -> None:
        """Если в MD нет # заголовка, title = имя файла."""
        write_md(tmp_config, "myfile.md", "## Обязательные\n- Правило\n")
        loader = InvariantLoader(tmp_config)
        assert loader.categories[0].title == "myfile"

    def test_asterisk_list_items_parsed(self, tmp_config: Path) -> None:
        """Пункты списка со * тоже парсятся."""
        write_md(
            tmp_config,
            "test.md",
            "# T\n\n## Обязательные\n* Правило 1\n* Правило 2\n",
        )
        loader = InvariantLoader(tmp_config)
        assert len(loader.categories[0].required) == 2

    def test_ignores_lines_outside_sections(self, tmp_config: Path) -> None:
        """Строки вне разделов ## Обязательные / ## Рекомендуемые игнорируются."""
        content = (
            "# Title\n"
            "Это вводный текст, его нельзя путать с правилами.\n\n"
            "## Обязательные\n"
            "- Правило\n\n"
            "Ещё текст после раздела\n"
        )
        write_md(tmp_config, "test.md", content)
        loader = InvariantLoader(tmp_config)
        cat = loader.categories[0]
        assert cat.required == ["Правило"]


# ---------------------------------------------------------------------------
# Тесты загрузки нескольких файлов
# ---------------------------------------------------------------------------


class TestMultipleFiles:
    def test_loads_multiple_md_files(self, tmp_config: Path) -> None:
        write_md(tmp_config, "arch.md", "# Архитектура\n\n## Обязательные\n- R1\n")
        write_md(tmp_config, "stack.md", "# Стек\n\n## Обязательные\n- R2\n- R3\n")
        loader = InvariantLoader(tmp_config)
        assert len(loader.categories) == 2

    def test_files_sorted_alphabetically(self, tmp_config: Path) -> None:
        write_md(tmp_config, "zzz.md", "# ZZZ\n\n## Обязательные\n- R\n")
        write_md(tmp_config, "aaa.md", "# AAA\n\n## Обязательные\n- R\n")
        loader = InvariantLoader(tmp_config)
        assert loader.categories[0].name == "aaa"
        assert loader.categories[1].name == "zzz"

    def test_get_all_required_from_multiple(self, tmp_config: Path) -> None:
        write_md(tmp_config, "a.md", "# A\n\n## Обязательные\n- A1\n- A2\n")
        write_md(tmp_config, "b.md", "# B\n\n## Обязательные\n- B1\n")
        loader = InvariantLoader(tmp_config)
        required = loader.get_all_required()
        # Должны быть все правила из всех файлов
        assert len(required) == 3
        # Формат: (title, rule)
        rules_text = [r for _, r in required]
        assert "A1" in rules_text
        assert "A2" in rules_text
        assert "B1" in rules_text

    def test_get_all_recommended_from_multiple(self, tmp_config: Path) -> None:
        write_md(tmp_config, "a.md", "# A\n\n## Рекомендуемые\n- Rec1\n")
        write_md(tmp_config, "b.md", "# B\n\n## Рекомендуемые\n- Rec2\n- Rec3\n")
        loader = InvariantLoader(tmp_config)
        recommended = loader.get_all_recommended()
        assert len(recommended) == 3


# ---------------------------------------------------------------------------
# Тесты граничных случаев
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_invariants_dir(self, tmp_config: Path) -> None:
        """Пустая директория → пустой список категорий."""
        loader = InvariantLoader(tmp_config)
        assert loader.categories == []
        assert loader.is_empty()

    def test_nonexistent_config_dir(self, tmp_path: Path) -> None:
        """Несуществующая директория → пустой список, без исключения."""
        loader = InvariantLoader(tmp_path / "nonexistent")
        assert loader.categories == []

    def test_empty_md_file(self, tmp_config: Path) -> None:
        """Пустой MD-файл → категория с пустыми списками."""
        write_md(tmp_config, "empty.md", "")
        loader = InvariantLoader(tmp_config)
        assert len(loader.categories) == 1
        cat = loader.categories[0]
        assert cat.required == []
        assert cat.recommended == []

    def test_md_only_title_no_sections(self, tmp_config: Path) -> None:
        """Файл с заголовком но без разделов → пустые списки правил."""
        write_md(tmp_config, "title_only.md", "# Только заголовок\n")
        loader = InvariantLoader(tmp_config)
        cat = loader.categories[0]
        assert cat.title == "Только заголовок"
        assert cat.required == []
        assert cat.recommended == []

    def test_is_empty_with_rules(self, tmp_config: Path) -> None:
        write_md(tmp_config, "test.md", "# T\n\n## Обязательные\n- Правило\n")
        loader = InvariantLoader(tmp_config)
        assert not loader.is_empty()

    def test_is_empty_without_rules(self, tmp_config: Path) -> None:
        write_md(tmp_config, "test.md", "# T\n")
        loader = InvariantLoader(tmp_config)
        assert loader.is_empty()

    def test_non_md_files_ignored(self, tmp_config: Path) -> None:
        """Файлы не с расширением .md игнорируются."""
        (tmp_config / "invariants" / "readme.txt").write_text("- Не правило")
        (tmp_config / "invariants" / "notes.json").write_text('{"key": "val"}')
        loader = InvariantLoader(tmp_config)
        assert loader.categories == []


# ---------------------------------------------------------------------------
# Тесты формирования prompt-блока
# ---------------------------------------------------------------------------


class TestPromptBlock:
    def test_empty_loader_returns_empty_string(self, tmp_config: Path) -> None:
        loader = InvariantLoader(tmp_config)
        assert loader.build_prompt_block() == ""

    def test_block_contains_invariants_tag(self, tmp_config: Path) -> None:
        write_md(tmp_config, "test.md", "# T\n\n## Обязательные\n- Правило\n")
        loader = InvariantLoader(tmp_config)
        block = loader.build_prompt_block()
        assert "<INVARIANTS>" in block
        assert "</INVARIANTS>" in block

    def test_block_contains_required_rules(self, tmp_config: Path) -> None:
        write_md(
            tmp_config,
            "stack.md",
            "# Стек\n\n## Обязательные\n- SQLite only\n- Python 3.11+\n",
        )
        loader = InvariantLoader(tmp_config)
        block = loader.build_prompt_block()
        assert "SQLite only" in block
        assert "Python 3.11+" in block

    def test_block_contains_recommended_rules(self, tmp_config: Path) -> None:
        write_md(tmp_config, "test.md", "# T\n\n## Рекомендуемые\n- Предпочитать stdlib\n")
        loader = InvariantLoader(tmp_config)
        block = loader.build_prompt_block()
        assert "Предпочитать stdlib" in block

    def test_block_section_headers(self, tmp_config: Path) -> None:
        write_md(
            tmp_config,
            "test.md",
            "# T\n\n## Обязательные\n- R1\n\n## Рекомендуемые\n- R2\n",
        )
        loader = InvariantLoader(tmp_config)
        block = loader.build_prompt_block()
        assert "Обязательные" in block
        assert "Рекомендуемые" in block

    def test_block_only_required_no_recommended_section(self, tmp_config: Path) -> None:
        """Если нет рекомендуемых, секция рекомендуемых не добавляется."""
        write_md(tmp_config, "test.md", "# T\n\n## Обязательные\n- R\n")
        loader = InvariantLoader(tmp_config)
        block = loader.build_prompt_block()
        assert "Рекомендуемые" not in block

    def test_block_includes_category_title(self, tmp_config: Path) -> None:
        write_md(tmp_config, "stack.md", "# Ограничения по стеку\n\n## Обязательные\n- R\n")
        loader = InvariantLoader(tmp_config)
        block = loader.build_prompt_block()
        assert "Ограничения по стеку" in block


# ---------------------------------------------------------------------------
# Тест hot-reload
# ---------------------------------------------------------------------------


class TestHotReload:
    def test_reload_picks_up_new_file(self, tmp_config: Path) -> None:
        """Reload подхватывает новые файлы без создания нового экземпляра."""
        loader = InvariantLoader(tmp_config)
        assert loader.categories == []

        # Добавляем файл после создания загрузчика
        write_md(tmp_config, "new.md", "# Новый\n\n## Обязательные\n- Новое правило\n")

        reloaded = loader.reload()
        assert len(reloaded) == 1
        assert reloaded[0].title == "Новый"
        assert "Новое правило" in reloaded[0].required

    def test_reload_picks_up_changed_file(self, tmp_config: Path) -> None:
        """Reload подхватывает изменения в существующем файле."""
        md_path = write_md(
            tmp_config, "test.md", "# T\n\n## Обязательные\n- Старое правило\n"
        )
        loader = InvariantLoader(tmp_config)
        assert loader.categories[0].required == ["Старое правило"]

        # Изменяем файл
        md_path.write_text(
            "# T\n\n## Обязательные\n- Старое правило\n- Новое правило\n",
            encoding="utf-8",
        )

        loader.reload()
        assert len(loader.categories[0].required) == 2
        assert "Новое правило" in loader.categories[0].required

    def test_reload_removes_deleted_file(self, tmp_config: Path) -> None:
        """Reload убирает категорию если файл удалён."""
        md_path = write_md(tmp_config, "test.md", "# T\n\n## Обязательные\n- R\n")
        loader = InvariantLoader(tmp_config)
        assert len(loader.categories) == 1

        md_path.unlink()
        loader.reload()
        assert loader.categories == []

    def test_reload_returns_categories(self, tmp_config: Path) -> None:
        """reload() возвращает список категорий равный categories."""
        write_md(tmp_config, "test.md", "# T\n\n## Обязательные\n- R\n")
        loader = InvariantLoader(tmp_config)
        reloaded = loader.reload()
        assert reloaded == loader.categories


# ---------------------------------------------------------------------------
# Тест format_for_display
# ---------------------------------------------------------------------------


class TestFormatForDisplay:
    def test_empty_loader_message(self, tmp_config: Path) -> None:
        loader = InvariantLoader(tmp_config)
        display = loader.format_for_display()
        assert "не загружены" in display.lower() or "config/invariants" in display

    def test_display_shows_all_categories(self, tmp_config: Path) -> None:
        write_md(tmp_config, "a.md", "# Cat A\n\n## Обязательные\n- R\n")
        write_md(tmp_config, "b.md", "# Cat B\n\n## Рекомендуемые\n- R\n")
        loader = InvariantLoader(tmp_config)
        display = loader.format_for_display()
        assert "Cat A" in display
        assert "Cat B" in display

    def test_display_shows_counts(self, tmp_config: Path) -> None:
        write_md(
            tmp_config,
            "test.md",
            "# T\n\n## Обязательные\n- R1\n- R2\n\n## Рекомендуемые\n- R3\n",
        )
        loader = InvariantLoader(tmp_config)
        display = loader.format_for_display()
        assert "2" in display  # 2 обязательных
        assert "1" in display  # 1 рекомендуемый
