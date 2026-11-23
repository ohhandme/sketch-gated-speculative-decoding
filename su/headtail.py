"""Exact head plus low-rank tail reconstruction of the draft law (spec §2.4)."""
from __future__ import annotations

import numpy as np


def head_indices(q: np.ndarray, tokens: int) -> np.ndarray:
    """Indices of the `tokens` largest probabilities, sorted by probability desc.

    Ties are broken by ascending token id, so the selected head (and therefore
    the reconstructed payload) is bit-for-bit reproducible across BLAS builds
    and hardware. ``np.argpartition`` alone is not deterministic in that case.
    """
    q = np.asarray(q)
    if q.ndim != 1:
        raise ValueError("expected a single distribution")
    tokens = int(max(0, min(tokens, q.size)))
    if tokens == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((np.arange(q.size), -q))
    return order[:tokens]


def merge(head_idx: np.ndarray, head_prob: np.ndarray, tail_logits: np.ndarray) -> np.ndarray:
    """Combine exact head masses with a normalised low-rank tail.

    The tail is renormalised over the vocabulary minus the head, then scaled by
    the mass the head leaves over. Exact head probabilities are preserved.
    """
    tail_logits = np.asarray(tail_logits, dtype=np.float64)
    vocab = tail_logits.size
    if head_idx.size != head_prob.size:
        raise ValueError("head indices and probabilities disagree")
    if head_prob.size and (head_prob < 0).any():
        raise ValueError("head probabilities must be non-negative")
    head_mass = float(np.clip(head_prob.sum(), 0.0, 1.0))
    logits = tail_logits.copy()
    logits[head_idx] = -np.inf
    shift = logits.max() if np.isfinite(logits).any() else 0.0
    weights = np.exp(logits - shift)
    weights[head_idx] = 0.0
    total = float(weights.sum())
    out = np.zeros(vocab, dtype=np.float64)
    if total > 0:
        out[:] = (1.0 - head_mass) * weights / total
    elif head_mass < 1.0:
        # Degenerate tail: spread the remaining mass uniformly over non-head ids.
        free = vocab - head_idx.size
        if free > 0:
            mask = np.ones(vocab, dtype=bool)
            mask[head_idx] = False
            out[mask] = (1.0 - head_mass) / free
    out[head_idx] = head_prob
    s = out.sum()
    if s <= 0:
        raise ValueError("reconstruction has non-positive mass")
    return out / s


def log_of(q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(q, dtype=np.float64))
