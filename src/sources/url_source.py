"""Источник документации: загрузка по URL с обходом внутренних ссылок.

Алгоритм:
    1. Загрузить стартовую страницу через urllib
    2. Извлечь внутренние ссылки с тем же доменом и path-префиксом
    3. Загрузить найденные страницы (до INDEX_MAX_PAGES из .env)
    4. Каждую страницу → RawDocument (markdown + URL как source)

Глубина обхода контролируется INDEX_MAX_DEPTH из .env.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

from .source import DocumentSource, RawDocument


_DEFAULT_MAX_PAGES = 50
_DEFAULT_MAX_DEPTH = 2
_DEFAULT_TIMEOUT = 15
_NOISE_TAGS = {"nav", "footer", "header", "aside", "script", "style", "noscript",
               "iframe", "form", "button"}


class URLDocSource(DocumentSource):
    """Источник документации: загрузка страниц по URL.

    Начинает со стартовой страницы, извлекает внутренние ссылки
    (тот же домен, тот же path-префикс) и загружает их рекурсивно.

    Args:
        start_url:  Стартовый URL документации.
        max_pages:  Максимальное кол-во страниц (из env INDEX_MAX_PAGES).
        max_depth:  Глубина обхода ссылок (из env INDEX_MAX_DEPTH).
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int | None = None,
        max_depth: int | None = None,
    ) -> None:
        self.start_url = start_url
        self.max_pages = max_pages or _get_env_int("INDEX_MAX_PAGES", _DEFAULT_MAX_PAGES)
        self.max_depth = max_depth or _get_env_int("INDEX_MAX_DEPTH", _DEFAULT_MAX_DEPTH)

        parsed = urllib.parse.urlparse(start_url)
        self._base_domain = parsed.netloc
        self._path_prefix = parsed.path.rstrip("/")

    def fetch(self) -> list[RawDocument]:
        """Загрузить страницы начиная со стартового URL.

        Обходит внутренние ссылки в BFS порядке до max_pages страниц
        и глубины max_depth. Возвращает RawDocument-ы.

        Returns:
            Список RawDocument-ов с контентом в Markdown.
        """
        print(f"[URLDocSource] Старт: {self.start_url}")
        print(f"  max_pages={self.max_pages}, max_depth={self.max_depth}")

        # BFS: (url, depth)
        queue: deque[tuple[str, int]] = deque()
        queue.append((self.start_url, 0))
        visited: set[str] = set()
        raw_docs: list[RawDocument] = []

        while queue and len(raw_docs) < self.max_pages:
            url, depth = queue.popleft()

            # Нормализуем URL
            url = _normalize_url(url)
            if url in visited:
                continue
            visited.add(url)

            print(f"  [{len(raw_docs)+1}/{self.max_pages}] {url} (глубина {depth})")
            result = _fetch_page(url)

            if result["error"]:
                print(f"    Ошибка: {result['error']}")
                continue

            content = result["content"]
            title = result["title"] or _url_to_title(url)

            if len(content.strip()) < 50:
                print(f"    Пропуск (слишком мало контента)")
                continue

            raw_docs.append(RawDocument(
                content=content,
                source_path=url,
                title=title,
            ))

            # Извлекаем ссылки для следующего уровня
            if depth < self.max_depth:
                links = _extract_internal_links(
                    result["html"], url,
                    self._base_domain, self._path_prefix,
                )
                for link in links:
                    link = _normalize_url(link)
                    if link not in visited:
                        queue.append((link, depth + 1))

        print(f"[URLDocSource] Загружено {len(raw_docs)} страниц")
        return raw_docs

    @property
    def source_description(self) -> str:
        return f"url:{self.start_url}"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _normalize_url(url: str) -> str:
    """Убрать fragment (#...) и trailing slash из URL."""
    parsed = urllib.parse.urlparse(url)
    normalized = parsed._replace(fragment="", query="")
    result = urllib.parse.urlunparse(normalized)
    return result.rstrip("/")


def _url_to_title(url: str) -> str:
    """Сгенерировать заголовок из последнего сегмента URL."""
    path = urllib.parse.urlparse(url).path
    parts = [p for p in path.split("/") if p]
    if parts:
        return parts[-1].replace("-", " ").replace("_", " ").title()
    return url


def _extract_internal_links(
    html_text: str,
    base_url: str,
    domain: str,
    path_prefix: str,
) -> list[str]:
    """Извлечь внутренние ссылки с тем же доменом и path-префиксом.

    Args:
        html_text:   HTML страницы.
        base_url:    URL текущей страницы (для разрешения относительных ссылок).
        domain:      Разрешённый домен (например, laravel.com).
        path_prefix: Разрешённый префикс пути (например, /docs/13.x).

    Returns:
        Список абсолютных URL.
    """
    links = []
    # Ищем все href="..."
    for m in re.finditer(r'href=["\']([^"\'#?][^"\']*)["\']', html_text):
        href = m.group(1)

        # Разрешаем относительные ссылки
        absolute = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(absolute)

        # Проверяем домен и префикс пути
        if parsed.netloc != domain:
            continue
        if path_prefix and not parsed.path.startswith(path_prefix):
            continue
        # Пропускаем нетекстовые ресурсы
        if re.search(r'\.(png|jpg|jpeg|gif|svg|ico|pdf|zip|css|js)$',
                     parsed.path, re.IGNORECASE):
            continue

        links.append(absolute)

    return list(dict.fromkeys(links))  # дедупликация с сохранением порядка


def _remove_noise(html_text: str) -> str:
    """Удалить теги-мусор вместе с содержимым."""
    for tag in _NOISE_TAGS:
        html_text = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", "", html_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return html_text


def _html_to_markdown(html_text: str) -> str:
    """Конвертировать HTML в Markdown без внешних зависимостей."""
    text = _remove_noise(html_text)

    # Заголовки
    for i in range(6, 0, -1):
        text = re.sub(
            rf"<h{i}[^>]*>(.*?)</h{i}>",
            lambda m, n=i: "\n" + "#" * n + " " + re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n",
            text, flags=re.DOTALL | re.IGNORECASE,
        )

    # Параграфы и списки
    text = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[ou]l[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</[ou]l>", "\n", text, flags=re.IGNORECASE)

    # Ссылки и форматирование
    text = re.sub(r'<a[^>]+href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
                  r"\2", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>", r"**\1**",
                  text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(?:i|em)[^>]*>(.*?)</(?:i|em)>", r"*\1*",
                  text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`",
                  text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    # Нормализация
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_title(html_text: str) -> str:
    """Извлечь заголовок из тега <title>."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.DOTALL | re.IGNORECASE)
    if m:
        return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return ""


def _extract_main_content(html_text: str) -> str:
    """Выделить основной контент страницы (article, main или body)."""
    for tag in ("article", "main"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html_text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)

    for cls in ("content", "article", "post", "entry", "documentation", "docs-content"):
        for pat in [
            rf'<div[^>]+class="[^"]*\b{cls}\b[^"]*"[^>]*>(.*?)</div>',
            rf"<div[^>]+class='[^']*\b{cls}\b[^']*'[^>]*>(.*?)</div>",
        ]:
            m = re.search(pat, html_text, re.DOTALL | re.IGNORECASE)
            if m and len(m.group(1)) > 200:
                return m.group(1)

    m = re.search(r"<body[^>]*>(.*?)</body>", html_text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else html_text


def _fetch_page(url: str) -> dict:
    """Загрузить страницу и вернуть {html, title, content, error}."""
    timeout = _get_env_int("SCRAPER_TIMEOUT", _DEFAULT_TIMEOUT)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SupportBot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            charset_match = re.search(r"charset=([^\s;]+)", content_type)
            charset = charset_match.group(1).lower() if charset_match else "utf-8"
            raw = resp.read(500_000)

        try:
            html_text = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html_text = raw.decode("utf-8", errors="replace")

        title = _extract_title(html_text)
        content_html = _extract_main_content(html_text)
        content_md = _html_to_markdown(content_html)

        return {"html": html_text, "title": title, "content": content_md, "error": None}

    except urllib.error.HTTPError as exc:
        return {"html": "", "title": "", "content": "", "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"html": "", "title": "", "content": "", "error": str(exc)}
