"""Подсчёт токенов для OpenAI-совместимых моделей.

Основная реализация использует библиотеку tiktoken (точный счётчик).
Если tiktoken не установлен — автоматически используется встроенный
regex-аппроксиматор, который не требует внешних зависимостей.

Точность аппроксиматора: ±10–15% от реального числа токенов.
Для учебных целей этого достаточно.
"""

from __future__ import annotations

import re

from llm_agent.domain.models import ChatMessage


# ---------------------------------------------------------------------------
# Попытка импорта tiktoken (точный счётчик)
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback: regex-аппроксиматор (без внешних зависимостей)
# ---------------------------------------------------------------------------

# Паттерн, приближающий разбивку cl100k_base (GPT-3.5 / GPT-4):
#   - числа — каждое отдельно
#   - кирилличные/латинские буквы — кусками до 4 символов
#   - знаки препинания и пробелы — поодиночке
_TOKEN_PATTERN = re.compile(
    r"\d+"                      # числа целиком
    r"|[a-zA-Z]{1,4}"           # латиница: куски по 1–4 символа (≈ BPE merges)
    r"|[а-яёА-ЯЁ]{1,2}"        # кирилица: куски по 1–2 символа (UTF-8 bytes)
    r"|[^\w\s]"                 # знаки препинания
    r"|\s+",                    # пробелы/переносы
    re.UNICODE,
)


def _approx_count_tokens(text: str) -> int:
    """Аппроксимировать число токенов без tiktoken.

    Алгоритм имитирует BPE cl100k_base:
    - ASCII слова: ~1 токен на 4 символа
    - Кирилица: ~1 токен на 1–2 символа (2 байта UTF-8 → чаще сплит)
    - Числа, пунктуация, пробелы: ~1 токен каждый
    """
    if not text:
        return 0
    return len(_TOKEN_PATTERN.findall(text))


# ---------------------------------------------------------------------------
# Публичный класс
# ---------------------------------------------------------------------------

class TiktokenCounter:
    """Считает токены для OpenAI-совместимых моделей.

    Предпочитает tiktoken, если он доступен. Иначе использует
    встроенный regex-аппроксиматор (_approx_count_tokens).

    Формула накладных расходов для chat-формата OpenAI:
        каждое сообщение = 4 служебных токена + роль + контент
        конец списка = 2 токена (прайминг ответа)

    Источник: https://platform.openai.com/docs/guides/chat/managing-tokens
    """

    _TOKENS_PER_MESSAGE = 4  # накладные расходы на каждое сообщение
    _TOKENS_PER_REPLY = 2    # прайминг ответа ассистента в конце списка

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self._model = model
        self._encoding = None

        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoding = _tiktoken.encoding_for_model(model)
            except KeyError:
                self._encoding = _tiktoken.get_encoding("cl100k_base")

    @property
    def uses_tiktoken(self) -> bool:
        """True, если используется точный счётчик tiktoken."""
        return self._encoding is not None

    def count_tokens(self, text: str) -> int:
        """Подсчитать количество токенов в произвольном тексте."""
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return _approx_count_tokens(text)

    def count_messages_tokens(self, messages: list[ChatMessage]) -> int:
        """Подсчитать токены для всего списка сообщений (формат OpenAI chat).

        Returns:
            Суммарное число токенов, которые будут отправлены в API.
        """
        total = self._TOKENS_PER_REPLY
        for msg in messages:
            total += self._TOKENS_PER_MESSAGE
            total += self.count_tokens(msg.role)
            total += self.count_tokens(msg.content)
        return total
