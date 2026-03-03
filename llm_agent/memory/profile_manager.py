"""ProfileManager: управление профилями пользователя.

Профиль — именованный набор настроек, который:
  - хранится в SQLite (таблица profiles, тот же memory.db)
  - подключается к каждому LLM-запросу через system prompt
  - не влияет на память (память общая для всех профилей)
  - активен только один одновременно
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from llm_agent.memory.profile_models import Profile


class ProfileManager:
    """Менеджер профилей пользователя на базе SQLite.

    Открывает отдельное соединение к тому же memory.db, что и MemoryManager.
    Хранит все профили в таблице `profiles`.

    Parameters:
        db_path: Путь к файлу SQLite (создаётся автоматически).
    """

    DEFAULT_PROFILE = "default"

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._setup_schema()
        self._ensure_default_profile()

    # ------------------------------------------------------------------
    # Схема БД
    # ------------------------------------------------------------------

    def _setup_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                name          TEXT PRIMARY KEY,
                display_name  TEXT    NOT NULL,
                system_prompt TEXT    NOT NULL DEFAULT '',
                created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                is_active     INTEGER NOT NULL DEFAULT 0
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Инициализация системного профиля
    # ------------------------------------------------------------------

    def _ensure_default_profile(self) -> None:
        """Создать профиль 'default' если не существует; активировать если нет активного."""
        if self.get(self.DEFAULT_PROFILE) is None:
            self._conn.execute(
                "INSERT INTO profiles (name, display_name, system_prompt) VALUES (?, ?, ?)",
                (self.DEFAULT_PROFILE, "По умолчанию", ""),
            )
            self._conn.commit()
        if self.get_active() is None:
            self._conn.execute(
                "UPDATE profiles SET is_active = 1 WHERE name = ?",
                (self.DEFAULT_PROFILE,),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Внутренние утилиты
    # ------------------------------------------------------------------

    def _row_to_profile(self, row: sqlite3.Row) -> Profile:
        return Profile(
            name=row["name"],
            display_name=row["display_name"],
            system_prompt=row["system_prompt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, name: str, display_name: str, system_prompt: str) -> Profile:
        """Создать новый профиль.

        Raises:
            ValueError: Если профиль с таким именем уже существует.
        """
        try:
            self._conn.execute(
                "INSERT INTO profiles (name, display_name, system_prompt) "
                "VALUES (?, ?, ?)",
                (name, display_name, system_prompt),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Профиль '{name}' уже существует.")
        row = self._conn.execute(
            "SELECT * FROM profiles WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_profile(row)

    def get(self, name: str) -> Profile | None:
        """Получить профиль по имени. Возвращает None если не найден."""
        row = self._conn.execute(
            "SELECT * FROM profiles WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_profile(row) if row else None

    def get_active(self) -> Profile | None:
        """Получить активный профиль. Возвращает None если нет активного."""
        row = self._conn.execute(
            "SELECT * FROM profiles WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        return self._row_to_profile(row) if row else None

    def list_all(self) -> list[Profile]:
        """Список всех профилей (по алфавиту)."""
        rows = self._conn.execute(
            "SELECT * FROM profiles ORDER BY name"
        ).fetchall()
        return [self._row_to_profile(r) for r in rows]

    def update(
        self,
        name: str,
        *,
        system_prompt: str | None = None,
        display_name: str | None = None,
    ) -> Profile:
        """Обновить поля профиля.

        Raises:
            ValueError: Если профиль не найден или является системным.
        """
        if name == self.DEFAULT_PROFILE:
            raise ValueError(
                f"Профиль '{self.DEFAULT_PROFILE}' является системным и не может быть изменён."
            )
        profile = self.get(name)
        if profile is None:
            raise ValueError(f"Профиль '{name}' не найден.")

        new_sp = system_prompt if system_prompt is not None else profile.system_prompt
        new_dn = display_name if display_name is not None else profile.display_name

        self._conn.execute(
            "UPDATE profiles SET system_prompt=?, display_name=?, "
            "updated_at=datetime('now') WHERE name=?",
            (new_sp, new_dn, name),
        )
        self._conn.commit()
        return self.get(name)  # type: ignore[return-value]

    def delete(self, name: str) -> None:
        """Удалить профиль.

        Raises:
            ValueError: Если профиль не найден, является системным или активным.
        """
        if name == self.DEFAULT_PROFILE:
            raise ValueError(
                f"Профиль '{self.DEFAULT_PROFILE}' является системным и не может быть удалён."
            )
        profile = self.get(name)
        if profile is None:
            raise ValueError(f"Профиль '{name}' не найден.")
        if profile.is_active:
            raise ValueError(
                f"Нельзя удалить активный профиль '{name}'. "
                "Сначала переключитесь на другой: /profile use <name>"
            )
        self._conn.execute("DELETE FROM profiles WHERE name = ?", (name,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Активация
    # ------------------------------------------------------------------

    def deactivate_all(self) -> None:
        """Снять активность со всех профилей."""
        self._conn.execute("UPDATE profiles SET is_active = 0")
        self._conn.commit()

    def set_active(self, name: str) -> Profile:
        """Сделать профиль активным (сбрасывает остальные).

        Raises:
            ValueError: Если профиль не найден.
        """
        if self.get(name) is None:
            raise ValueError(f"Профиль '{name}' не найден.")
        self._conn.execute("UPDATE profiles SET is_active = 0")
        self._conn.execute(
            "UPDATE profiles SET is_active = 1 WHERE name = ?", (name,)
        )
        self._conn.commit()
        return self.get(name)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Экспорт / Импорт
    # ------------------------------------------------------------------

    def export_json(self, name: str) -> str:
        """Сериализовать профиль в JSON-строку.

        Raises:
            ValueError: Если профиль не найден.
        """
        profile = self.get(name)
        if profile is None:
            raise ValueError(f"Профиль '{name}' не найден.")
        data = {
            "name": profile.name,
            "display_name": profile.display_name,
            "system_prompt": profile.system_prompt,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def import_json(self, json_str: str) -> Profile:
        """Создать профиль из JSON-строки.

        Ожидаемые поля: name, display_name, system_prompt.

        Raises:
            ValueError: Если JSON некорректен или профиль уже существует.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Некорректный JSON: {exc}") from exc

        required = {"name", "display_name", "system_prompt"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"В JSON отсутствуют поля: {', '.join(sorted(missing))}")

        return self.create(
            name=str(data["name"]),
            display_name=str(data["display_name"]),
            system_prompt=str(data["system_prompt"]),
        )

    # ------------------------------------------------------------------
    # Сборка system prompt
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        base_prompt: str = "",
        long_term_text: str = "",
        working_text: str = "",
    ) -> str:
        """Собрать итоговый system message для LLM-запроса.

        Порядок блоков (пустые пропускаются, разделитель — два переноса строки):
          1. base_prompt     — базовый системный промпт агента
          2. long_term_text  — блок из long-term памяти
          3. working_text    — блок из working памяти
          4. User profile    — system_prompt активного профиля (наибольший приоритет)

        Returns:
            Итоговый system message.
        """
        parts: list[str] = []

        if base_prompt.strip():
            parts.append(base_prompt.strip())

        if long_term_text.strip():
            parts.append(long_term_text.strip())

        if working_text.strip():
            parts.append(working_text.strip())

        active = self.get_active()
        if active and active.system_prompt.strip():
            parts.append(f"## User profile\n{active.system_prompt.strip()}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Управление ресурсом
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Закрыть соединение с БД."""
        self._conn.close()

    def __enter__(self) -> ProfileManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
