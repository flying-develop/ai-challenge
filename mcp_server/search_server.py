"""MCP-сервер: поиск ссылок по запросу.

Образовательная концепция: агент формирует поисковый запрос (мозг),
search_server выполняет реальный поиск в Яндексе (руки).
LLM решает ЧТО искать, сервер выполняет КАК.

Два режима (SEARCH_MODE в .env):
  yandex_cloud — реальный поиск через Yandex Cloud Search API
  mock          — воспроизводимая заглушка из search_mock_data.json

Запуск:
    python -m mcp_server.search_server

Инструменты:
    search_web — поиск по запросу, возвращает JSON-список ссылок
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from mcp.server.fastmcp import FastMCP

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

mcp = FastMCP("Search Server")

# Жёсткий лимит: не более 10 результатов
_MAX_RESULTS = 10

# Кэш IAM-токена (живёт 12 часов)
_iam_token_cache: dict = {"token": None, "expires_at": 0.0}


# ---------------------------------------------------------------------------
# IAM-токен (Yandex Cloud)
# ---------------------------------------------------------------------------

def _get_iam_token() -> str:
    """Получить IAM-токен из OAuth-токена. Кэшируем на 11 часов."""
    now = time.time()
    if _iam_token_cache["token"] and now < _iam_token_cache["expires_at"]:
        return _iam_token_cache["token"]

    oauth_token = os.environ.get("YANDEX_CLOUD_OAUTH_TOKEN", "").strip()
    if not oauth_token:
        raise ValueError(
            "YANDEX_CLOUD_OAUTH_TOKEN не задан в .env. "
            "Получите токен: https://yandex.cloud/en-ru/docs/iam/concepts/authorization/oauth-token"
        )

    payload = json.dumps({"yandexPassportOauthToken": oauth_token}).encode("utf-8")
    req = urllib.request.Request(
        "https://iam.api.cloud.yandex.net/iam/v1/tokens",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ошибка получения IAM-токена {exc.code}: {body}") from exc

    token = data.get("iamToken", "")
    if not token:
        raise RuntimeError(f"IAM API не вернул токен: {data}")

    _iam_token_cache["token"] = token
    _iam_token_cache["expires_at"] = now + 11 * 3600  # 11 часов
    return token


# ---------------------------------------------------------------------------
# Yandex Cloud Search
# ---------------------------------------------------------------------------

def _yandex_search(query: str, limit: int) -> list[dict]:
    """Поиск через Yandex Cloud Search API (асинхронный, двухшаговый)."""
    folder_id = os.environ.get("YANDEX_CLOUD_FOLDER_ID", "").strip()
    if not folder_id:
        raise ValueError(
            "YANDEX_CLOUD_FOLDER_ID не задан в .env. "
            "Получите: https://yandex.cloud/en-ru/docs/resource-manager/operations/folder/get-id"
        )

    iam_token = _get_iam_token()
    groups = min(limit, _MAX_RESULTS)

    # Шаг 1: Отправить запрос, получить operation_id
    body = {
        "query": {
            "search_type": "SEARCH_TYPE_RU",
            "query_text": query,
        },
        "sort_spec": {"sort_mode": "SORT_MODE_BY_RELEVANCE"},
        "group_spec": {"groups_on_page": groups},
        "folder_id": folder_id,
    }
    req1 = urllib.request.Request(
        "https://searchapi.api.cloud.yandex.net/v2/web/searchAsync",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {iam_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req1, timeout=15) as resp:
            op = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ошибка Yandex Search API {exc.code}: {body_err}") from exc

    operation_id = op.get("id", "")
    if not operation_id:
        raise RuntimeError(f"Yandex Search API не вернул operation_id: {op}")

    # Шаг 2: Поллинг пока done=true (обычно 1-3 секунды)
    poll_url = f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
    poll_req = urllib.request.Request(
        poll_url,
        headers={"Authorization": f"Bearer {iam_token}"},
        method="GET",
    )

    for attempt in range(10):
        time.sleep(1.5 if attempt == 0 else 1.0)
        try:
            with urllib.request.urlopen(poll_req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_err = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ошибка поллинга результата {exc.code}: {body_err}") from exc

        if result.get("done"):
            break
    else:
        raise RuntimeError("Yandex Search: таймаут ожидания результатов (10 попыток)")

    # Парсинг XML из response.rawData
    raw_data = result.get("response", {}).get("rawData", "")
    if not raw_data:
        return []

    import base64
    try:
        xml_bytes = base64.b64decode(raw_data)
        xml_text = xml_bytes.decode("utf-8", errors="replace")
    except Exception:
        xml_text = raw_data  # попробуем как есть

    return _parse_yandex_xml(xml_text)


def _parse_yandex_xml(xml_text: str) -> list[dict]:
    """Парсим XML-ответ Yandex Search API."""
    results: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    # <group><doc><url>...</url><title>...</title><headline>...</headline></doc></group>
    for group in root.iter("group"):
        for doc in group.iter("doc"):
            url_el = doc.find("url")
            title_el = doc.find("title")
            headline_el = doc.find("headline")

            url = url_el.text.strip() if url_el is not None and url_el.text else ""
            title = _strip_xml_tags(
                ET.tostring(title_el, encoding="unicode") if title_el is not None else ""
            )
            snippet = _strip_xml_tags(
                ET.tostring(headline_el, encoding="unicode") if headline_el is not None else ""
            )
            if url:
                results.append({"title": title, "url": url, "snippet": snippet})

    return results


def _strip_xml_tags(xml_str: str) -> str:
    """Убрать XML-теги, оставить только текст."""
    try:
        root = ET.fromstring(f"<x>{xml_str}</x>")
        return "".join(root.itertext()).strip()
    except ET.ParseError:
        import re
        return re.sub(r"<[^>]+>", "", xml_str).strip()


# ---------------------------------------------------------------------------
# Mock-режим
# ---------------------------------------------------------------------------

def _mock_search(query: str, limit: int) -> list[dict]:
    """Поиск-заглушка: возвращает данные из search_mock_data.json."""
    mock_path = Path(__file__).parent / "search_mock_data.json"
    if not mock_path.exists():
        # Fallback: генерируем синтетические данные
        return [
            {
                "title": f"Результат {i+1} по запросу «{query}»",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"Синтетический сниппет {i+1} для демонстрации работы поиска.",
            }
            for i in range(min(limit, _MAX_RESULTS))
        ]

    try:
        data = json.loads(mock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    results = data.get("results", data if isinstance(data, list) else [])
    return results[:min(limit, _MAX_RESULTS)]


# ---------------------------------------------------------------------------
# Инструмент: search_web
# ---------------------------------------------------------------------------

@mcp.tool()
def search_web(query: str, limit: int = 10) -> str:
    """
    Поиск ссылок в Яндексе по запросу.

    Режим определяется переменной SEARCH_MODE в .env:
      yandex_cloud — реальный поиск через Yandex Cloud Search API
      mock          — воспроизводимая заглушка (по умолчанию)

    Args:
        query: Поисковый запрос на естественном языке
        limit: Максимум результатов (1-10, по умолчанию 10)

    Returns:
        JSON: [{"title": "...", "url": "...", "snippet": "..."}, ...]
    """
    # Инвариант: не более 10 результатов
    limit = max(1, min(limit, _MAX_RESULTS))

    mode = os.environ.get("SEARCH_MODE", "mock").strip().lower()

    try:
        if mode == "yandex_cloud":
            results = _yandex_search(query, limit)
        else:
            results = _mock_search(query, limit)
    except (ValueError, RuntimeError, ConnectionError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return json.dumps(results[:limit], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
