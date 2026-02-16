import json
import io
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

import requests

import chat


# --- Helpers ---

def make_sse_stream(tokens: list, done: bool = True) -> list:
    """Создаёт список SSE-строк, имитируя стриминг OpenAI (decode_unicode=True)."""
    lines = []
    for token in tokens:
        chunk = {
            "choices": [{"delta": {"content": token}}]
        }
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    if done:
        lines.append("data: [DONE]")
    return lines


def make_mock_response(tokens: list, status: int = 200) -> MagicMock:
    """Мок requests.Response со стримингом."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} Error"
        )
    resp.iter_lines = MagicMock(
        return_value=iter(make_sse_stream(tokens))
    )
    return resp


# --- Tests: send_message ---

class TestSendMessage(unittest.TestCase):

    @patch("chat.request_with_retry")
    def test_returns_full_reply(self, mock_req):
        mock_req.return_value = make_mock_response(["Hello", " world", "!"])

        result = chat.send_message([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "Hello world!")

    @patch("chat.request_with_retry")
    def test_streams_tokens_to_stdout(self, mock_req):
        mock_req.return_value = make_mock_response(["one", "two"])

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            chat.send_message([{"role": "user", "content": "hi"}])

        output = captured.getvalue()
        self.assertIn("one", output)
        self.assertIn("two", output)

    @patch("chat.request_with_retry")
    def test_empty_response(self, mock_req):
        mock_req.return_value = make_mock_response([])

        result = chat.send_message([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "")

    @patch("chat.request_with_retry")
    def test_raises_on_api_error(self, mock_req):
        mock_req.side_effect = requests.exceptions.HTTPError("500 Server Error")

        with self.assertRaises(requests.exceptions.HTTPError):
            chat.send_message([{"role": "user", "content": "hi"}])


# --- Tests: request_with_retry ---

class TestRequestWithRetry(unittest.TestCase):

    @patch("requests.post")
    def test_success_on_first_try(self, mock_post):
        resp = make_mock_response(["ok"])
        mock_post.return_value = resp

        result = chat.request_with_retry("http://test", {}, {})

        self.assertEqual(result, resp)
        self.assertEqual(mock_post.call_count, 1)

    @patch("time.sleep")
    @patch("requests.post")
    def test_retries_on_429(self, mock_post, mock_sleep):
        resp_429 = MagicMock(spec=requests.Response)
        resp_429.status_code = 429

        resp_ok = make_mock_response(["ok"])

        mock_post.side_effect = [resp_429, resp_429, resp_ok]

        result = chat.request_with_retry("http://test", {}, {})

        self.assertEqual(result, resp_ok)
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("time.sleep")
    @patch("requests.post")
    def test_raises_after_max_retries(self, mock_post, mock_sleep):
        resp_429 = MagicMock(spec=requests.Response)
        resp_429.status_code = 429
        resp_429.raise_for_status = MagicMock(
            side_effect=requests.exceptions.HTTPError("429 Too Many Requests")
        )

        mock_post.return_value = resp_429

        with self.assertRaises(requests.exceptions.HTTPError):
            chat.request_with_retry("http://test", {}, {})

        self.assertEqual(mock_post.call_count, chat.MAX_RETRIES + 1)


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
        with patch("chat.request_with_retry") as mock_req:
            mock_req.return_value = make_mock_response(["Hello!"])

            captured = io.StringIO()
            with patch("sys.stdout", captured):
                chat.main()

        output = captured.getvalue()
        self.assertIn("AI: Hello!", output)
        self.assertIn("Пока!", output)

    @patch("builtins.input", side_effect=["", "quit"])
    @patch("chat.API_TOKEN", "test-token")
    def test_skips_empty_input(self, mock_input):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            chat.main()

        output = captured.getvalue()
        self.assertIn("Пока!", output)

    @patch("chat.send_message", side_effect=requests.exceptions.HTTPError("500"))
    @patch("builtins.input", side_effect=["hi", "quit"])
    @patch("chat.API_TOKEN", "test-token")
    def test_handles_api_error(self, mock_input, mock_send):  # noqa: ARG002
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            chat.main()

        output = captured.getvalue()
        self.assertIn("Ошибка API", output)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("chat.API_TOKEN", "test-token")
    def test_handles_ctrl_c(self, mock_input):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            chat.main()

        output = captured.getvalue()
        self.assertIn("Пока!", output)


if __name__ == "__main__":
    unittest.main()
