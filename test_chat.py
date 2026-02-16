import io
import sys
import time
import types
import unittest
from unittest.mock import patch, MagicMock

# --- Мокаем openai модуль до импорта chat ---

mock_openai = types.ModuleType("openai")

mock_openai.OpenAI = MagicMock
mock_openai.APIError = type("APIError", (Exception,), {
    "__init__": lambda self, message="", response=None, body=None: (
        setattr(self, "message", message) or
        setattr(self, "response", response) or
        setattr(self, "body", body) or
        Exception.__init__(self, message)
    ),
})
mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {
    "__init__": lambda self, message="", request=None: (
        setattr(self, "message", message) or
        Exception.__init__(self, message)
    ),
})
mock_openai.RateLimitError = type("RateLimitError", (mock_openai.APIError,), {})

sys.modules["openai"] = mock_openai

import chat  # noqa: E402 — после мока

APIError = mock_openai.APIError
APIConnectionError = mock_openai.APIConnectionError
RateLimitError = mock_openai.RateLimitError


# --- Helpers ---

def make_chunk(token: str) -> MagicMock:
    """Создаёт мок одного chunk-а стриминга."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = token
    return chunk


def make_mock_client(tokens: list) -> MagicMock:
    """Мок OpenAI клиента, возвращающий стрим из токенов."""
    client = MagicMock()
    chunks = [make_chunk(t) for t in tokens]
    client.chat.completions.create.return_value = iter(chunks)
    return client


# --- Tests: send_message ---

class TestSendMessage(unittest.TestCase):

    def test_returns_full_reply(self):
        client = make_mock_client(["Hello", " world", "!"])

        result = chat.send_message(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(result, "Hello world!")

    def test_streams_tokens_to_stdout(self):
        client = make_mock_client(["one", "two"])

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            chat.send_message(client, [{"role": "user", "content": "hi"}])

        output = captured.getvalue()
        self.assertIn("one", output)
        self.assertIn("two", output)

    def test_empty_response(self):
        client = make_mock_client([])

        result = chat.send_message(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(result, "")

    def test_raises_on_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = APIError(
            message="Server Error",
        )

        with self.assertRaises(APIError):
            chat.send_message(client, [{"role": "user", "content": "hi"}])

    @patch("time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep):
        client = MagicMock()

        rate_error = RateLimitError(message="Rate limit")

        chunks = [make_chunk("ok")]
        client.chat.completions.create.side_effect = [
            rate_error,
            rate_error,
            iter(chunks),
        ]

        result = chat.send_message(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(result, "ok")
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        client = MagicMock()

        rate_error = RateLimitError(message="Rate limit")

        client.chat.completions.create.side_effect = rate_error

        with self.assertRaises(RateLimitError):
            chat.send_message(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(
            client.chat.completions.create.call_count,
            chat.MAX_RETRIES + 1,
        )


# --- Tests: Spinner ---

class TestSpinner(unittest.TestCase):

    def test_start_stop(self):
        spinner = chat.Spinner()
        spinner.start()
        time.sleep(0.3)
        spinner.stop()
        self.assertFalse(spinner._thread.is_alive())

    def test_stop_without_start(self):
        spinner = chat.Spinner()
        spinner.stop()  # не должен упасть


# --- Tests: main (integration) ---

class TestMain(unittest.TestCase):

    @patch("chat.API_TOKEN", None)
    def test_exits_without_token(self):
        with self.assertRaises(SystemExit) as ctx:
            chat.main()
        self.assertEqual(ctx.exception.code, 1)

    @patch("builtins.input", side_effect=["hi", "quit"])
    @patch("chat.API_TOKEN", "test-token")
    def test_chat_loop(self, mock_input):
        mock_client = make_mock_client(["Hello!"])

        with patch("chat.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                chat.main()

        output = captured.getvalue()
        self.assertIn("AI: Hello!", output)
        self.assertIn("Пока!", output)

    @patch("builtins.input", side_effect=["", "quit"])
    @patch("chat.API_TOKEN", "test-token")
    def test_skips_empty_input(self, mock_input):
        mock_client = MagicMock()

        with patch("chat.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                chat.main()

        output = captured.getvalue()
        self.assertIn("Пока!", output)

    @patch("builtins.input", side_effect=["hi", "quit"])
    @patch("chat.API_TOKEN", "test-token")
    def test_handles_api_error(self, mock_input):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIError(
            message="Server Error",
        )

        with patch("chat.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                chat.main()

        output = captured.getvalue()
        self.assertIn("Ошибка API", output)

    @patch("builtins.input", side_effect=["\udcd1\udcbb\udcd0\udcb8\udcb2\udcd0\udcb5\udcd1\udc82", "quit"])
    @patch("chat.API_TOKEN", "test-token")
    def test_handles_surrogate_input(self, mock_input):
        mock_client = make_mock_client(["ok"])

        with patch("chat.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                chat.main()

        # Не должен упасть с UnicodeEncodeError
        output = captured.getvalue()
        self.assertIn("AI: ok", output)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("chat.API_TOKEN", "test-token")
    def test_handles_ctrl_c(self, mock_input):
        mock_client = MagicMock()

        with patch("chat.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                chat.main()

        output = captured.getvalue()
        self.assertIn("Пока!", output)


class TestConfig(unittest.TestCase):

    @patch.dict("os.environ", {"API_URL": "https://api.openai.com/v1/chat/completions"})
    def test_strips_chat_completions_suffix(self):
        import importlib
        importlib.reload(chat)
        self.assertEqual(chat.API_URL, "https://api.openai.com/v1")

    @patch.dict("os.environ", {"API_URL": "https://api.openai.com/v1"})
    def test_keeps_clean_url(self):
        import importlib
        importlib.reload(chat)
        self.assertEqual(chat.API_URL, "https://api.openai.com/v1")

    @patch.dict("os.environ", {"API_URL": "https://api.openai.com/v1/chat/completions/"})
    def test_strips_chat_completions_with_trailing_slash(self):
        import importlib
        importlib.reload(chat)
        self.assertEqual(chat.API_URL, "https://api.openai.com/v1")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fallback-key", "API_TOKEN": ""})
    def test_fallback_to_openai_api_key_env(self):
        import importlib
        importlib.reload(chat)
        self.assertEqual(chat.API_TOKEN, "fallback-key")


if __name__ == "__main__":
    unittest.main()
