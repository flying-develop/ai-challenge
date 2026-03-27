"""День 30: Локальная LLM как приватный HTTP-сервис.

Проверяет удалённый Ollama-сервер на VPS:
  1. Health check — доступность API по сети
  2. Stability test — 5 последовательных запросов, замер latency
  3. Parallel test — 3 параллельных запроса через threading
  4. Max context test — запрос с контекстом ~3072 токена
  5. Rate limit test — 15 быстрых запросов, ожидаем 429 после бёрста

Использование:
    # Подключение к VPS через nginx + basic auth
    VPS_OLLAMA_URL=http://<IP>/v1 VPS_OLLAMA_USER=llmuser VPS_OLLAMA_PASSWORD=xxx \
        python demo_vps_service.py

    # Локальный тест (Ollama на localhost без auth)
    python demo_vps_service.py --local

    # Только health check
    python demo_vps_service.py --health-only

    # Вывести команды деплоя на VPS
    python demo_vps_service.py --deploy-guide
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


# ── Конфигурация ──────────────────────────────────────────────────────────────

DEPLOY_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║          День 30: Команды деплоя Ollama на VPS (CPU-only)                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Рекомендуемый VPS: AMD EPYC 7502 (2 cores, 4GB RAM) — быстрее на CPU     ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ 1. Базовая установка ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    sudo apt update && sudo apt upgrade -y
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable ollama && sudo systemctl start ollama

━━━ 2. Открыть API на всех интерфейсах ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee /etc/systemd/system/ollama.service.d/override.conf <<EOF
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0:11434"
    Environment="OLLAMA_NUM_PARALLEL=2"
    Environment="OLLAMA_MAX_LOADED_MODELS=1"
    EOF
    sudo systemctl daemon-reload && sudo systemctl restart ollama

━━━ 3. Загрузить модель ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ollama pull qwen3.5:2b               # ~2.6GB, Q4_K_M
    ollama pull nomic-embed-text      # 274MB embeddings
    curl http://localhost:11434/api/tags   # проверка

━━━ 4. nginx reverse proxy + basic auth + rate limit ━━━━━━━━━━━━━━━━━━━━━━

    sudo apt install nginx apache2-utils -y
    sudo htpasswd -c /etc/nginx/.htpasswd llmuser

    sudo tee /etc/nginx/sites-available/llm-api <<'NGINX'
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;

    server {
        listen 80;
        server_name _;

        location /v1/ {
            auth_basic "LLM API";
            auth_basic_user_file /etc/nginx/.htpasswd;
            limit_req zone=api burst=5 nodelay;
            proxy_pass http://127.0.0.1:11434/;
            proxy_set_header Host $host;
            proxy_read_timeout 300s;
        }

        location /health {
            proxy_pass http://127.0.0.1:11434/api/tags;
            proxy_read_timeout 10s;
        }
    }
    NGINX

    sudo ln -s /etc/nginx/sites-available/llm-api /etc/nginx/sites-enabled/
    sudo nginx -t && sudo nginx -s reload

    # Закрыть прямой доступ к Ollama
    sudo ufw allow 22 && sudo ufw allow 80 && sudo ufw deny 11434 && sudo ufw enable

━━━ 5. Проверка с локальной машины ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    curl http://<VPS_IP>/health
    curl -u llmuser:PASSWORD http://<VPS_IP>/v1/api/generate \\
      -d '{\"model\":\"qwen3.5:2b\",\"prompt\":\"Hello\",\"stream\":false}'

━━━ 6. Запуск этого демо ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    VPS_OLLAMA_URL=http://<VPS_IP>/v1 \\
    VPS_OLLAMA_USER=llmuser \\
    VPS_OLLAMA_PASSWORD=<пароль> \\
    python demo_vps_service.py
"""


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    latencies: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    @property
    def p95(self) -> float | None:
        if len(self.latencies) < 2:
            return self.latencies[0] if self.latencies else None
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


# ── HTTP-утилиты ──────────────────────────────────────────────────────────────

def _make_headers(auth_header: str | None) -> dict:
    h = {"Content-Type": "application/json"}
    if auth_header:
        h["Authorization"] = auth_header
    return h


def _make_auth_header(user: str | None, password: str | None) -> str | None:
    if not user or not password:
        return None
    import base64
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def _get(url: str, auth_header: str | None, timeout: float = 10.0) -> dict:
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post(url: str, payload: dict, auth_header: str | None, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers=_make_headers(auth_header), method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_raw_status(url: str, payload: dict, auth_header: str | None, timeout: float = 5.0) -> int:
    """Возвращает HTTP статус-код без исключений."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers=_make_headers(auth_header), method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


# ── Тесты ─────────────────────────────────────────────────────────────────────

def test_health(base_url: str, auth_header: str | None) -> TestResult:
    """Проверить доступность API по сети."""
    health_url = base_url.rstrip("/").replace("/v1", "") + "/health"
    # Попробовать /health, затем /api/tags напрямую
    urls_to_try = [health_url, base_url.rstrip("/") + "/api/tags"]

    for url in urls_to_try:
        try:
            start = time.perf_counter()
            data = _get(url, auth_header, timeout=10.0)
            elapsed = time.perf_counter() - start
            models = [m["name"] for m in data.get("models", [])]
            models_str = ", ".join(models[:5]) or "—"
            return TestResult(
                name="Health Check",
                passed=True,
                details=f"OK в {elapsed:.2f}с | Модели: {models_str}",
                latencies=[elapsed],
            )
        except Exception:
            continue

    return TestResult(name="Health Check", passed=False, details="Сервер недоступен")


def test_stability(
    base_url: str, auth_header: str | None, model: str, n: int = 5
) -> TestResult:
    """N последовательных запросов — проверить стабильность и latency."""
    url = base_url.rstrip("/") + "/api/generate"
    latencies: list[float] = []
    errors: list[str] = []

    for i in range(n):
        payload = {
            "model": model,
            "prompt": f"Ответь одним словом: сколько будет {i+1}+{i+1}?",
            "stream": False,
            "options": {"num_predict": 20, "temperature": 0.0},
        }
        try:
            start = time.perf_counter()
            _post(url, payload, auth_header, timeout=60.0)
            latencies.append(time.perf_counter() - start)
        except Exception as e:
            errors.append(f"req{i+1}: {e}")

    ok = n - len(errors)
    passed = ok == n
    details = f"{ok}/{n} успешно"
    if latencies:
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        details += f" | p50={p50:.1f}с p95={p95:.1f}с"
    if errors:
        details += f" | Ошибки: {'; '.join(errors[:2])}"
    return TestResult(name="Stability (5 req)", passed=passed, details=details, latencies=latencies)


def test_parallel(
    base_url: str, auth_header: str | None, model: str, n: int = 3
) -> TestResult:
    """N параллельных запросов через threading."""
    url = base_url.rstrip("/") + "/api/generate"
    results: list[tuple[bool, float]] = [None] * n  # type: ignore

    def worker(idx: int) -> None:
        payload = {
            "model": model,
            "prompt": f"Скажи 'OK{idx}'",
            "stream": False,
            "options": {"num_predict": 10, "temperature": 0.0},
        }
        try:
            start = time.perf_counter()
            _post(url, payload, auth_header, timeout=120.0)
            results[idx] = (True, time.perf_counter() - start)
        except Exception as e:
            results[idx] = (False, 0.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    start_all = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - start_all

    ok = sum(1 for r in results if r and r[0])
    latencies = [r[1] for r in results if r and r[0]]
    passed = ok == n
    details = f"{ok}/{n} успешно | суммарное время: {total:.1f}с"
    if latencies:
        details += f" | avg: {statistics.mean(latencies):.1f}с"
    return TestResult(name=f"Parallel ({n} req)", passed=passed, details=details, latencies=latencies)


def test_max_context(
    base_url: str, auth_header: str | None, model: str, target_tokens: int = 3000
) -> TestResult:
    """Запрос с большим контекстом — проверить лимит токенов."""
    url = base_url.rstrip("/") + "/api/generate"

    # Генерируем ~3000 токенов контекста (~4 символа/токен)
    filler = ("Это тестовое предложение для проверки максимального контекста модели. " * 60)[:target_tokens * 4]
    prompt = f"{filler}\n\nВ конце этого текста ответь одним словом: 'КОНЕЦ'"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 10, "num_ctx": 4096, "temperature": 0.0},
    }

    try:
        start = time.perf_counter()
        data = _post(url, payload, auth_header, timeout=180.0)
        elapsed = time.perf_counter() - start
        prompt_tokens = data.get("prompt_eval_count", "?")
        completion_tokens = data.get("eval_count", "?")
        return TestResult(
            name=f"Max Context (~{target_tokens} tok)",
            passed=True,
            details=f"OK в {elapsed:.1f}с | промпт={prompt_tokens}tok, ответ={completion_tokens}tok",
            latencies=[elapsed],
        )
    except Exception as e:
        return TestResult(
            name=f"Max Context (~{target_tokens} tok)",
            passed=False,
            details=f"Ошибка: {e}",
        )


def test_rate_limit(
    base_url: str, auth_header: str | None, model: str, n_requests: int = 15
) -> TestResult:
    """Быстрые запросы подряд — ожидаем 429 при наличии rate limit в nginx."""
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": "1+1=?",
        "stream": False,
        "options": {"num_predict": 5, "temperature": 0.0},
    }

    statuses: list[int] = []
    first_429_at: int | None = None

    for i in range(n_requests):
        status = _post_raw_status(url, payload, auth_header, timeout=5.0)
        statuses.append(status)
        if status == 429 and first_429_at is None:
            first_429_at = i + 1
        # Небольшая пауза чтобы не положить сервер
        time.sleep(0.1)

    ok_count = statuses.count(200)
    rate_limited = statuses.count(429)

    if first_429_at is not None:
        passed = True
        details = f"Rate limit сработал на запросе #{first_429_at} | 200:{ok_count} 429:{rate_limited}"
    elif all(s == 200 for s in statuses):
        # Нет rate limit настроен — это нормально для прямого Ollama
        passed = True
        details = f"Rate limit не настроен (все {ok_count}/{n_requests} — 200 OK) — добавьте nginx для продакшна"
    else:
        passed = False
        details = f"Неожиданные статусы: {set(statuses)}"

    return TestResult(name=f"Rate Limit ({n_requests} req)", passed=passed, details=details)


# ── Вывод результатов ─────────────────────────────────────────────────────────

def print_results(results: list[TestResult]) -> None:
    print()
    print("═" * 72)
    print(f"  {'Тест':<30} {'Статус':<8} {'Детали'}")
    print("═" * 72)
    for r in results:
        icon = "✅" if r.passed else "❌"
        print(f"  {r.name:<30} {icon}      {r.details}")
    print("═" * 72)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n  Итог: {passed}/{total} тестов прошли\n")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="День 30: Тест локальной LLM как приватного HTTP-сервиса"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Подключиться к localhost:11434 (без auth)",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Только health check",
    )
    parser.add_argument(
        "--deploy-guide",
        action="store_true",
        help="Вывести команды деплоя Ollama на VPS",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Имя модели (по умолчанию: qwen3.5:2b или из env OLLAMA_LLM_MODEL)",
    )
    args = parser.parse_args()

    if args.deploy_guide:
        print(DEPLOY_GUIDE)
        return

    # Определяем параметры подключения
    if args.local:
        base_url = "http://localhost:11434"
        auth_header = None
        mode = "локальный (localhost)"
    else:
        base_url = os.environ.get("VPS_OLLAMA_URL", "")
        user = os.environ.get("VPS_OLLAMA_USER", "")
        password = os.environ.get("VPS_OLLAMA_PASSWORD", "")
        auth_header = _make_auth_header(user or None, password or None)

        if not base_url:
            print("\n❌ Не задан VPS_OLLAMA_URL.")
            print("   Варианты запуска:")
            print("     python demo_vps_service.py --local              # localhost")
            print("     VPS_OLLAMA_URL=http://<IP>/v1 python demo_vps_service.py")
            print("     python demo_vps_service.py --deploy-guide       # команды деплоя")
            return

        auth_str = "с basic auth" if auth_header else "без auth"
        mode = f"VPS ({base_url}, {auth_str})"

    model = args.model or os.environ.get("OLLAMA_LLM_MODEL", "qwen3.5:2b")

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         День 30: Локальная LLM как приватный HTTP-сервис        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Адрес : {base_url}")
    print(f"  Модель: {model}")
    print(f"  Режим : {mode}")

    # Health check всегда первый
    print("\n[1/5] Health check...")
    health = test_health(base_url, auth_header)
    if not health.passed:
        print(f"\n❌ {health.details}")
        print("\nСервер недоступен. Проверьте:")
        print("  • Ollama запущена: sudo systemctl status ollama")
        print("  • nginx запущен: sudo nginx -t && sudo systemctl status nginx")
        print("  • Firewall: sudo ufw status")
        print("  • URL и порт корректны")
        print(f"\nДля команд деплоя: python demo_vps_service.py --deploy-guide")
        return

    print(f"  ✅ {health.details}")

    if args.health_only:
        print("\nHealth check пройден.\n")
        return

    # Остальные тесты
    results = [health]

    print("\n[2/5] Stability test (5 последовательных запросов)...")
    results.append(test_stability(base_url, auth_header, model))

    print("\n[3/5] Parallel test (3 параллельных запроса)...")
    results.append(test_parallel(base_url, auth_header, model))

    print("\n[4/5] Max context test (~3000 токенов)...")
    results.append(test_max_context(base_url, auth_header, model))

    print("\n[5/5] Rate limit test (15 быстрых запросов)...")
    results.append(test_rate_limit(base_url, auth_header, model))

    print_results(results)

    # Подсказка по настройке rate limit если не настроен
    for r in results:
        if "Rate limit" in r.name and "не настроен" in r.details:
            print("  💡 Для продакшна рекомендуется nginx с rate limiting:")
            print("     python demo_vps_service.py --deploy-guide")
            print()
            break


if __name__ == "__main__":
    main()
