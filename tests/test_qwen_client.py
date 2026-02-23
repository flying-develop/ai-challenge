"""Tests for the infrastructure layer (QwenHttpClient)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_agent.domain.models import LLMResponse
from llm_agent.infrastructure.qwen_client import QwenHttpClient


BASE_URL = "https://api.example.com/v1"
API_KEY = "test-key-12345"
MODEL = "qwen-plus"

SUCCESS_PAYLOAD = {
    "id": "cmpl-001",
    "object": "chat.completion",
    "model": "qwen-plus",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello, world!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_mock_response(status_code: int, payload: dict) -> MagicMock:
    """Build a mock httpx.Response-like object."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = payload
    mock_resp.text = json.dumps(payload)

    if status_code >= 400:
        mock_request = MagicMock(spec=httpx.Request)
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=mock_request,
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status.return_value = None

    return mock_resp


def test_qwen_client_handles_success_response() -> None:
    mock_resp = _make_mock_response(200, SUCCESS_PAYLOAD)

    with patch("httpx.Client.post", return_value=mock_resp):
        client = QwenHttpClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=5.0)
        result = client.generate("Hello")

    assert isinstance(result, LLMResponse)
    assert result.text == "Hello, world!"
    assert result.model == "qwen-plus"
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_qwen_client_raises_on_http_error() -> None:
    mock_resp = _make_mock_response(401, {"error": "Unauthorized"})

    with patch("httpx.Client.post", return_value=mock_resp):
        client = QwenHttpClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=5.0)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.generate("Hello")

    assert "401" in str(exc_info.value)


def test_qwen_client_raises_on_server_error() -> None:
    mock_resp = _make_mock_response(500, {"error": "Internal Server Error"})

    with patch("httpx.Client.post", return_value=mock_resp):
        client = QwenHttpClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=5.0)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.generate("Hello")

    assert "500" in str(exc_info.value)


def test_qwen_client_sends_correct_model_in_body() -> None:
    mock_resp = _make_mock_response(200, SUCCESS_PAYLOAD)
    captured_kwargs: dict = {}

    def capture_post(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_resp

    with patch("httpx.Client.post", side_effect=capture_post):
        client = QwenHttpClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=5.0)
        client.generate("test prompt")

    body = captured_kwargs.get("json", {})
    assert body["model"] == MODEL
    assert body["messages"] == [{"role": "user", "content": "test prompt"}]
