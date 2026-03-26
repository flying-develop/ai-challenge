"""Конфигурации параметров генерации Ollama для оптимизации RAG support-бота.

Предоставляет:
    OllamaConfig  — dataclass с параметрами генерации
    CONFIGS       — словарь из 4 предопределённых конфигураций
    from_env()    — создать конфиг из переменных окружения

Использование::

    from src.providers.ollama_config import CONFIGS, OllamaConfig

    cfg = CONFIGS["precise"]
    options = cfg.to_options()   # dict для /api/chat options
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OllamaConfig:
    """Параметры генерации для одного прогона Ollama.

    Все параметры соответствуют полю ``options`` в теле запроса /api/chat.

    Args:
        name:        Идентификатор конфига (baseline, precise, fast, creative).
        description: Человекочитаемое описание.
        temperature: Температура сэмплинга (0.0 — детерминированно, 1.0+ — творчески).
        num_ctx:     Размер контекстного окна в токенах.
        num_predict: Максимальное число токенов в ответе.
        top_p:       Nucleus sampling — порог совокупной вероятности токенов.
        seed:        Зерно для воспроизводимости (0 = случайный).
    """

    name: str
    description: str
    temperature: float = 0.7
    num_ctx: int = 2048
    num_predict: int = 256
    top_p: float = 0.9
    seed: int = 0

    def to_options(self) -> dict:
        """Преобразовать в словарь для поля ``options`` Ollama API.

        Returns:
            dict с ключами temperature, num_ctx, num_predict, top_p,
            и seed (только если seed != 0).
        """
        opts: dict = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "top_p": self.top_p,
        }
        if self.seed != 0:
            opts["seed"] = self.seed
        return opts

    def __str__(self) -> str:
        seed_str = f", seed={self.seed}" if self.seed != 0 else ""
        return (
            f"{self.name}: temp={self.temperature}, ctx={self.num_ctx}, "
            f"predict={self.num_predict}, top_p={self.top_p}{seed_str}"
        )


# ---------------------------------------------------------------------------
# Предопределённые конфигурации
# ---------------------------------------------------------------------------

CONFIGS: dict[str, OllamaConfig] = {
    "baseline": OllamaConfig(
        name="baseline",
        description="Стандартные параметры Ollama (контроль)",
        temperature=0.7,
        num_ctx=2048,
        num_predict=256,
        top_p=0.9,
        seed=0,
    ),
    "precise": OllamaConfig(
        name="precise",
        description="Точный режим: низкая температура, широкий контекст",
        temperature=0.1,
        num_ctx=4096,
        num_predict=512,
        top_p=0.95,
        seed=42,
    ),
    "fast": OllamaConfig(
        name="fast",
        description="Быстрый режим: короткий контекст, меньше токенов",
        temperature=0.3,
        num_ctx=1024,
        num_predict=128,
        top_p=0.8,
        seed=0,
    ),
    "rag_tuned": OllamaConfig(
        name="rag_tuned",
        description="Оптимизировано для RAG: детерминированность + полный ответ",
        temperature=0.2,
        num_ctx=3072,
        num_predict=400,
        top_p=0.9,
        seed=42,
    ),
}


def from_env() -> OllamaConfig:
    """Создать конфиг из переменных окружения.

    Переменные::

        OLLAMA_TEMPERATURE   float (default 0.7)
        OLLAMA_NUM_CTX       int   (default 2048)
        OLLAMA_NUM_PREDICT   int   (default 256)
        OLLAMA_TOP_P         float (default 0.9)
        OLLAMA_SEED          int   (default 0)

    Returns:
        OllamaConfig с именем "env".
    """
    return OllamaConfig(
        name="env",
        description="Параметры из переменных окружения",
        temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0.7")),
        num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "2048")),
        num_predict=int(os.environ.get("OLLAMA_NUM_PREDICT", "256")),
        top_p=float(os.environ.get("OLLAMA_TOP_P", "0.9")),
        seed=int(os.environ.get("OLLAMA_SEED", "0")),
    )
