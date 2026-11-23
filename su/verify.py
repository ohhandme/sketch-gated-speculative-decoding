"""Acceptance, residual sampling and token-only verification (spec §2.2/§2.5)."""
from __future__ import annotations

import numpy as np


def accept_probability(p_x: float, q_x: float) -> float:
    """min(1, P(x)/Q(x)), with the ratio understood in the limit Q(x) -> 0."""
    p_x = float(p_x)
    q_x = float(q_x)
    if p_x <= 0.0:
        return 0.0
    if q_x <= 0.0:
        return 1.0
    return float(min(1.0, p_x / q_x))


def residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """norm(max(0, P - Q)); falls back to P when the residual vanishes."""
    r = np.clip(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64), 0.0, None)
    total = r.sum()
    if total <= 0:
        p = np.asarray(p, dtype=np.float64)
        s = p.sum()
        return p / s if s > 0 else np.full(p.shape, 1.0 / p.size)
    return r / total


def sample(prob: np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(prob, dtype=np.float64)
    c = np.cumsum(p)
    if not np.isfinite(c[-1]) or c[-1] <= 0:
        return int(rng.integers(0, p.size))
    u = rng.random() * c[-1]
    return int(np.searchsorted(c, u, side="right").clip(0, p.size - 1))


def token_only_token(p: np.ndarray) -> int:
    """Target's exact-match token: the argmax of P (ties go to the lowest id)."""
    return int(np.argmax(np.asarray(p)))


def support_preserved(p: np.ndarray, tokens: list[int]) -> bool:
    p = np.asarray(p)
    return all(0 <= t < p.size and p[t] > 0 for t in tokens)
