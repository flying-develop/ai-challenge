"""Совместимый слой для векторных операций.

Использует numpy если доступен, иначе — чистый Python + stdlib.
Это позволяет запускать пайплайн без установки numpy (тестовый режим).

В продакшене numpy всегда предпочтительнее: быстрее и точнее.
"""

from __future__ import annotations

import math
import struct
from typing import Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Детерминированный псевдо-рандом (для LocalRandomEmbedder)
# ---------------------------------------------------------------------------

def make_random_vector(seed: int, dimension: int) -> list[float]:
    """Сгенерировать нормализованный случайный вектор с заданным seed.

    Использует numpy.RandomState если доступен, иначе — Mersenne Twister.
    Результат: L2-нормализованный вектор размерности dimension.
    """
    if _HAS_NUMPY:
        rng = np.random.RandomState(seed % (2**31))
        vec = rng.randn(dimension).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
    else:
        import random
        rng = random.Random(seed % (2**31))
        # Box-Muller для нормального распределения
        vec = []
        for _ in range(dimension):
            u1 = rng.random() or 1e-12
            u2 = rng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            vec.append(z)
        # L2-нормализация
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# ---------------------------------------------------------------------------
# Сериализация векторов (float32 blob)
# ---------------------------------------------------------------------------

def vector_to_blob(vec: list[float]) -> bytes:
    """Упаковать список float в bytes (float32, little-endian)."""
    if _HAS_NUMPY:
        return np.array(vec, dtype=np.float32).tobytes()
    else:
        return struct.pack(f"<{len(vec)}f", *vec)


def blob_to_vector(blob: bytes) -> list[float]:
    """Распаковать bytes (float32) в список float."""
    if _HAS_NUMPY:
        return np.frombuffer(blob, dtype=np.float32).tolist()
    else:
        n = len(blob) // 4
        return list(struct.unpack(f"<{n}f", blob))


# ---------------------------------------------------------------------------
# Косинусное сходство
# ---------------------------------------------------------------------------

def cosine_similarities(query: list[float], matrix: list[list[float]]) -> list[float]:
    """Вычислить косинусное сходство между query и каждой строкой matrix.

    Args:
        query:  Вектор запроса (размерность D).
        matrix: Список векторов (N x D).

    Returns:
        Список float длиной N, каждый в [-1, 1].
    """
    if _HAS_NUMPY:
        q = np.array(query, dtype=np.float32)
        m = np.array(matrix, dtype=np.float32)
        # Нормализация
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        m = m / norms
        scores = (m @ q).tolist()
        return scores
    else:
        # Pure Python реализация
        q_norm = math.sqrt(sum(x * x for x in query))
        if q_norm > 0:
            q = [x / q_norm for x in query]
        else:
            q = list(query)

        scores = []
        for row in matrix:
            row_norm = math.sqrt(sum(x * x for x in row))
            if row_norm > 0:
                row_n = [x / row_norm for x in row]
            else:
                row_n = list(row)
            dot = sum(a * b for a, b in zip(q, row_n))
            scores.append(dot)
        return scores


def argsort_desc(values: list[float]) -> list[int]:
    """Вернуть индексы, отсортированные по убыванию значений."""
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)
