"""Модели данных для профилей пользователя."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Profile:
    """Профиль пользователя — именованный набор настроек ассистента.

    Attributes:
        name: Уникальный slug (PK). Например: "expert", "student".
        display_name: Человекочитаемое название.
        system_prompt: Текст, дословно вставляемый в system message каждого запроса.
        created_at: Дата создания (ISO-строка из SQLite).
        updated_at: Дата последнего изменения (ISO-строка из SQLite).
        is_active: Только один профиль активен одновременно.
    """

    name: str
    display_name: str
    system_prompt: str
    created_at: str
    updated_at: str
    is_active: bool
