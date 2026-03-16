"""MCP-сервер: загрузка веб-страниц и конвертация в Markdown.

Образовательная концепция: разделение ответственностей — search_server
ищет ссылки, scraper_server забирает контент. Разные таймауты, разные
источники ошибок, независимое масштабирование.

Пайплайн для каждого URL:
  1. urllib.request.urlopen(url, timeout=10)
  2. Поиск основного контента (<article>, <main>, <div class="content">)
  3. Удаление навигации (<nav>, <footer>, <header>, <aside>, <script>, <style>)
  4. HTML → Markdown (заголовки, параграфы, списки, ссылки)
  5. Обрезка до 3000 символов с пометкой [truncated]

Инварианты:
  - Не более 10 URL за один вызов
  - Ошибка одного URL не убивает остальные
  - Таймаут: SCRAPER_TIMEOUT секунд (по умолчанию 10)

Запуск:
    python -m mcp_server.scraper_server

Инструменты:
    fetch_urls — загрузить страницы и вернуть Markdown
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from mcp.server.fastmcp import FastMCP

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

mcp = FastMCP("Scraper Server")

_MAX_URLS = 10
_DEFAULT_TIMEOUT = 10
_DEFAULT_MAX_CONTENT = 3000

# Теги, содержащие основной контент
_CONTENT_TAGS = {"article", "main"}
_CONTENT_CLASSES = {"content", "article", "post", "entry", "text", "body"}

# Теги для удаления (мусор)
_NOISE_TAGS = {"nav", "footer", "header", "aside", "script", "style", "noscript",
               "iframe", "form", "button", "figure"}


# ---------------------------------------------------------------------------
# HTML → Markdown
# ---------------------------------------------------------------------------

def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _remove_noise_tags(html_text: str) -> str:
    """Удалить теги-мусор вместе с содержимым."""
    for tag in _NOISE_TAGS:
        pattern = rf"<{tag}[^>]*>.*?</{tag}>"
        html_text = re.sub(pattern, "", html_text, flags=re.DOTALL | re.IGNORECASE)
    return html_text


def _extract_main_content(html_text: str) -> str:
    """Попытаться выделить основной контент страницы."""
    # 1. Ищем <article> или <main>
    for tag in ("article", "main"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html_text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)

    # 2. Ищем <div class="content"> или похожие
    for cls in _CONTENT_CLASSES:
        patterns = [
            rf'<div[^>]+class="[^"]*\b{cls}\b[^"]*"[^>]*>(.*?)</div>',
            rf"<div[^>]+class='[^']*\b{cls}\b[^']*'[^>]*>(.*?)</div>",
        ]
        for pat in patterns:
            m = re.search(pat, html_text, re.DOTALL | re.IGNORECASE)
            if m and len(m.group(1)) > 200:
                return m.group(1)

    # 3. Fallback — весь body
    m = re.search(r"<body[^>]*>(.*?)</body>", html_text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else html_text


def _html_to_markdown(html_text: str) -> str:
    """Простая конвертация HTML → Markdown без внешних зависимостей."""
    text = html_text

    # Удаляем мусорные теги вместе с содержимым
    text = _remove_noise_tags(text)

    # Заголовки
    for i in range(6, 0, -1):
        text = re.sub(
            rf"<h{i}[^>]*>(.*?)</h{i}>",
            lambda m, n=i: "\n" + "#" * n + " " + m.group(1).strip() + "\n",
            text, flags=re.DOTALL | re.IGNORECASE
        )

    # Параграфы
    text = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", text, flags=re.DOTALL | re.IGNORECASE)

    # Списки
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[ou]l[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</[ou]l>", "\n", text, flags=re.IGNORECASE)

    # Ссылки: <a href="url">text</a> → [text](url)
    text = re.sub(
        r'<a[^>]+href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        r"[\2](\1)",
        text, flags=re.DOTALL | re.IGNORECASE
    )

    # Жирный/курсив
    text = re.sub(r"<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>", r"**\1**",
                  text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(?:i|em)[^>]*>(.*?)</(?:i|em)>", r"*\1*",
                  text, flags=re.DOTALL | re.IGNORECASE)

    # Перенос строк
    text = re.sub(r"<br[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<hr[^>]*>", "\n---\n", text, flags=re.IGNORECASE)

    # Убрать все оставшиеся теги
    text = re.sub(r"<[^>]+>", "", text)

    # HTML entities
    text = html.unescape(text)

    # Нормализация пробелов
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_title(html_text: str) -> str:
    """Извлечь заголовок страницы из <title>."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.DOTALL | re.IGNORECASE)
    if m:
        return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return ""


def _fetch_url(url: str, timeout: int, max_content: int) -> dict:
    """Загрузить одну страницу и сконвертировать в Markdown."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ResearchBot/1.0; "
                    "+https://github.com/ai-challenge)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Определяем кодировку
            content_type = resp.headers.get("Content-Type", "")
            charset_match = re.search(r"charset=([^\s;]+)", content_type)
            charset = charset_match.group(1).lower() if charset_match else "utf-8"
            # Ограничиваем размер скачиваемого контента
            raw = resp.read(500_000)

        try:
            html_text = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html_text = raw.decode("utf-8", errors="replace")

        title = _extract_title(html_text)
        content_html = _extract_main_content(html_text)
        content_md = _html_to_markdown(content_html)

        # Обрезаем до max_content символов
        if len(content_md) > max_content:
            content_md = content_md[:max_content] + "\n\n[truncated]"

        return {"url": url, "title": title, "content": content_md, "error": None}

    except urllib.error.HTTPError as exc:
        return {"url": url, "title": "", "content": "", "error": f"HTTP {exc.code}: {exc.reason}"}
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            return {"url": url, "title": "", "content": "", "error": "timeout"}
        return {"url": url, "title": "", "content": "", "error": f"network error: {reason}"}
    except TimeoutError:
        return {"url": url, "title": "", "content": "", "error": "timeout"}
    except Exception as exc:
        return {"url": url, "title": "", "content": "", "error": f"parse error: {exc}"}


# ---------------------------------------------------------------------------
# Инструмент: fetch_urls
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_urls(urls: list[str]) -> str:
    """
    Скачать веб-страницы и конвертировать в чистый Markdown.

    Инварианты:
      - Не более 10 URL за вызов
      - Ошибка одного URL не убивает остальные
      - Контент обрезается до SCRAPER_MAX_CONTENT_LENGTH символов

    Args:
        urls: Список URL для обработки (максимум 10)

    Returns:
        JSON: [{"url": "...", "title": "...", "content": "...", "error": null}, ...]
        При ошибке конкретного URL: {"url": ..., "error": "timeout|not found|..."}
    """
    timeout = _get_env_int("SCRAPER_TIMEOUT", _DEFAULT_TIMEOUT)
    max_content = _get_env_int("SCRAPER_MAX_CONTENT_LENGTH", _DEFAULT_MAX_CONTENT)

    # Инвариант: не более 10 URL
    urls = list(urls)[:_MAX_URLS]

    results = []
    for url in urls:
        result = _fetch_url(url, timeout, max_content)
        results.append(result)

    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
