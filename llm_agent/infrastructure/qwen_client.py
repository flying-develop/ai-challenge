"""HTTP adapter for the Qwen (Alibaba Cloud) LLM API.

The endpoint URL can be changed via QWEN_BASE_URL without touching any
business logic — this is the only file that knows about the HTTP wire format.
"""

from __future__ import annotations

import httpx

from llm_agent.domain.models import LLMResponse


class QwenHttpClient:
    """Implements LLMClientProtocol against the Qwen HTTP API.

    Uses httpx.Client (synchronous) with a single persistent connection pool.
    Supports context-manager usage for deterministic resource cleanup.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        # Ensure trailing slash so httpx merges relative paths correctly.
        # httpx RFC 3986 resolution: a leading slash in the path means
        # "absolute path from origin", dropping any base path prefix.
        # Using base_url with trailing slash + relative path avoids this.
        normalized_base = base_url.rstrip("/") + "/"
        self._client = httpx.Client(
            base_url=normalized_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def generate(self, prompt: str) -> LLMResponse:
        """Send prompt to the Qwen API and return a structured response.

        Args:
            prompt: The user's input text.

        Returns:
            LLMResponse with the model's reply text, model name, and token usage.

        Raises:
            httpx.HTTPStatusError: On non-2xx HTTP responses (includes status + body).
            httpx.TimeoutException: When the request exceeds the configured timeout.
            httpx.RequestError: On network connectivity failures.
        """
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = self._client.post("chat/completions", json=payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"Qwen API error {exc.response.status_code}: {exc.response.text}",
                request=exc.request,
                response=exc.response,
            ) from exc

        data = response.json()
        # Note: choices may be empty on content-moderation rejection (200 but no choices).
        choice = data["choices"][0]
        return LLMResponse(
            text=choice["message"]["content"],
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "QwenHttpClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
