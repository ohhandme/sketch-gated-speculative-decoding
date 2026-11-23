"""Acceptance, distribution and ranking metrics (spec §4.2 and §5.1)."""
from __future__ import annotations

import numpy as np


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    if (q[mask] <= 0).any():
        return float("inf")
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p, np.float64) - np.asarray(q, np.float64)).sum())


def ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    out = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        out[order[i:j + 1]] = avg
        i = j + 1
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if s.size != y.size:
        raise ValueError("scores and labels disagree")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    r = ranks(s)
    return float((r[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = ranks(x)
    ry = ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float((rx @ ry) / den) if den > 0 else float("nan")


def acceptance_length(tokens_per_step: list[int]) -> float:
    return float(np.mean(tokens_per_step)) if tokens_per_step else 0.0


def acceptance_rate(accepted: int, proposed: int) -> float:
    return float(accepted / proposed) if proposed else 0.0


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def crosspoint(xs: list[float], a: list[float], b: list[float]) -> float | None:
    """First x where curve a drops below curve b (linear interpolation)."""
    for i in range(1, len(xs)):
        d0 = a[i - 1] - b[i - 1]
        d1 = a[i] - b[i]
        if d0 > 0 >= d1:
            if d0 == d1:
                return float(xs[i])
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * d0 / (d0 - d1))
    return None
