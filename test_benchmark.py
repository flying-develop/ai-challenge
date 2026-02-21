import io
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# --- Мокаем openai модуль до импорта benchmark ---

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

import benchmark  # noqa: E402

APIError = mock_openai.APIError
APIConnectionError = mock_openai.APIConnectionError


# --- Helpers ---

def make_mock_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    """Создаёт мок не-стримингового ответа с usage."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    return response


def make_mock_client(content: str = "Test reply", prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    """Мок клиента, возвращающего заданный ответ."""
    client = MagicMock()
    client.chat.completions.create.return_value = make_mock_response(
        content, prompt_tokens, completion_tokens
    )
    return client


# --- Tests: run_benchmark ---

class TestRunBenchmark(unittest.TestCase):

    def test_returns_correct_reply(self):
        client = make_mock_client("Hello from model")
        model_info = {
            "name": "gpt-test",
            "label": "Тест",
            "price_input": 0.001,
            "price_output": 0.002,
        }

        result = benchmark.run_benchmark(client, model_info, "Вопрос")

        self.assertEqual(result["reply"], "Hello from model")
        self.assertEqual(result["model"], "gpt-test")
        self.assertEqual(result["label"], "Тест")

    def test_token_counts(self):
        client = make_mock_client("Reply", prompt_tokens=5, completion_tokens=15)
        model_info = {
            "name": "gpt-test",
            "label": "Тест",
            "price_input": 0.0,
            "price_output": 0.0,
        }

        result = benchmark.run_benchmark(client, model_info, "Q")

        self.assertEqual(result["input_tokens"], 5)
        self.assertEqual(result["output_tokens"], 15)
        self.assertEqual(result["total_tokens"], 20)

    def test_cost_calculation(self):
        # 1000 входных токенов × $0.001 + 1000 выходных × $0.002 = $3.0
        client = make_mock_client("R", prompt_tokens=1000, completion_tokens=1000)
        model_info = {
            "name": "gpt-test",
            "label": "Тест",
            "price_input": 1.0,
            "price_output": 2.0,
        }

        result = benchmark.run_benchmark(client, model_info, "Q")

        self.assertAlmostEqual(result["cost_usd"], 3.0, places=6)

    def test_elapsed_is_positive(self):
        client = make_mock_client("Reply")
        model_info = {
            "name": "gpt-test",
            "label": "Тест",
            "price_input": 0.001,
            "price_output": 0.002,
        }

        result = benchmark.run_benchmark(client, model_info, "Q")

        self.assertGreaterEqual(result["elapsed"], 0.0)

    def test_empty_reply(self):
        client = make_mock_client("")
        model_info = {
            "name": "gpt-test",
            "label": "Тест",
            "price_input": 0.0,
            "price_output": 0.0,
        }

        result = benchmark.run_benchmark(client, model_info, "Q")

        self.assertEqual(result["reply"], "")

    def test_passes_correct_model_to_api(self):
        client = make_mock_client()
        model_info = {
            "name": "gpt-specific",
            "label": "Тест",
            "price_input": 0.0,
            "price_output": 0.0,
        }

        benchmark.run_benchmark(client, model_info, "Q")

        call_kwargs = client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("model") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["model"], "gpt-specific")


# --- Tests: format_cost ---

class TestFormatCost(unittest.TestCase):

    def test_zero_returns_nd(self):
        self.assertEqual(benchmark.format_cost(0.0), "н/д")

    def test_positive_returns_dollar(self):
        result = benchmark.format_cost(0.001234)
        self.assertTrue(result.startswith("$"))
        self.assertIn("0.001234", result)

    def test_very_small_cost(self):
        result = benchmark.format_cost(0.000001)
        self.assertTrue(result.startswith("$"))


# --- Tests: print_result ---

class TestPrintResult(unittest.TestCase):

    def _make_result(self, reply="Hello", elapsed=1.5, input_t=10, output_t=20, cost=0.001):
        return {
            "model": "gpt-test",
            "label": "Тест",
            "hf_link": "https://example.com",
            "reply": reply,
            "elapsed": elapsed,
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": input_t + output_t,
            "cost_usd": cost,
        }

    def test_contains_reply(self):
        result = self._make_result(reply="My answer")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_result(result, 1)
        self.assertIn("My answer", captured.getvalue())

    def test_contains_elapsed(self):
        result = self._make_result(elapsed=2.75)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_result(result, 1)
        self.assertIn("2.75", captured.getvalue())

    def test_contains_token_counts(self):
        result = self._make_result(input_t=5, output_t=15)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_result(result, 1)
        output = captured.getvalue()
        self.assertIn("5", output)
        self.assertIn("15", output)

    def test_contains_model_name(self):
        result = self._make_result()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_result(result, 1)
        self.assertIn("gpt-test", captured.getvalue())


# --- Tests: print_comparison ---

class TestPrintComparison(unittest.TestCase):

    def _make_results(self):
        return [
            {
                "model": "gpt-weak",
                "label": "Слабая",
                "hf_link": "",
                "reply": "Short",
                "elapsed": 0.5,
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "cost_usd": 0.0001,
            },
            {
                "model": "gpt-medium",
                "label": "Средняя",
                "hf_link": "",
                "reply": "Medium answer",
                "elapsed": 1.5,
                "input_tokens": 10,
                "output_tokens": 40,
                "total_tokens": 50,
                "cost_usd": 0.001,
            },
            {
                "model": "gpt-strong",
                "label": "Сильная",
                "hf_link": "",
                "reply": "Very detailed answer",
                "elapsed": 3.0,
                "input_tokens": 10,
                "output_tokens": 100,
                "total_tokens": 110,
                "cost_usd": 0.01,
            },
        ]

    def test_shows_all_model_names(self):
        results = self._make_results()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_comparison(results)
        output = captured.getvalue()
        for r in results:
            self.assertIn(r["model"], output)

    def test_shows_speed_comparison(self):
        results = self._make_results()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_comparison(results)
        self.assertIn("Скорость", captured.getvalue())

    def test_shows_cost_comparison(self):
        results = self._make_results()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_comparison(results)
        self.assertIn("Стоимость", captured.getvalue())

    def test_shows_links(self):
        results = self._make_results()
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_comparison(results)
        self.assertIn("https://", captured.getvalue())

    def test_empty_results_does_nothing(self):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_comparison([])
        self.assertEqual(captured.getvalue(), "")

    def test_single_result(self):
        results = self._make_results()[:1]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            benchmark.print_comparison(results)
        self.assertIn("gpt-weak", captured.getvalue())


# --- Tests: main ---

class TestMain(unittest.TestCase):

    @patch("benchmark.API_TOKEN", None)
    def test_exits_without_token(self):
        with self.assertRaises(SystemExit) as ctx:
            benchmark.main()
        self.assertEqual(ctx.exception.code, 1)

    @patch("benchmark.API_TOKEN", "test-token")
    def test_runs_all_models(self):
        mock_client = make_mock_client("Answer", prompt_tokens=10, completion_tokens=30)

        with patch("benchmark.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                benchmark.main()

        # create.call_count должен совпадать с кол-вом моделей
        self.assertEqual(
            mock_client.chat.completions.create.call_count,
            len(benchmark.MODELS),
        )

    @patch("benchmark.API_TOKEN", "test-token")
    def test_custom_prompt_is_used(self):
        mock_client = make_mock_client("Answer")

        with patch("benchmark.create_client", return_value=mock_client):
            with patch("sys.stdout", io.StringIO()):
                benchmark.main(prompt="Custom question?")

        call_args = mock_client.chat.completions.create.call_args_list[0]
        messages = call_args.kwargs.get("messages") or call_args.args[1]
        self.assertEqual(messages[0]["content"], "Custom question?")

    @patch("benchmark.API_TOKEN", "test-token")
    def test_handles_api_error_gracefully(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIError(message="Bad model")

        with patch("benchmark.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                benchmark.main()  # не должен упасть

        self.assertIn("Ошибка API", captured.getvalue())

    @patch("benchmark.API_TOKEN", "test-token")
    def test_handles_connection_error_gracefully(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(message="No connection")

        with patch("benchmark.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                benchmark.main()  # не должен упасть

        self.assertIn("подключиться", captured.getvalue())

    @patch("benchmark.API_TOKEN", "test-token")
    def test_output_contains_comparison(self):
        mock_client = make_mock_client("Reply")

        with patch("benchmark.create_client", return_value=mock_client):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                benchmark.main()

        self.assertIn("СРАВНЕНИЕ", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
