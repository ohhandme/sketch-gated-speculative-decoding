"""Warm-started low-rank basis for the draft log-probabilities (spec §2.1).

The basis always contains the all-ones direction as its first column. Softmax is
invariant to per-row constants, so that column only carries the offset; the
remaining `rank - 1` columns are kept orthogonal to it and model the shape.
"""
from __future__ import annotations

import numpy as np


def ones_column(vocab: int) -> np.ndarray:
    return (np.ones((vocab, 1)) / np.sqrt(vocab)).astype(np.float32)


def init_basis(vocab: int, rank: int, seed: int) -> np.ndarray:
    if rank < 1 or rank > vocab:
        raise ValueError("rank must be in [1, vocab]")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((vocab, rank))
    ones = ones_column(vocab)
    raw = raw - ones @ (ones.T @ raw)
    q, _ = np.linalg.qr(raw)
    if rank == 1:
        return ones
    return np.hstack([ones, q[:, :rank - 1].astype(np.float32)])


def centre(log_probs: np.ndarray) -> np.ndarray:
    """Remove the per-row constant that softmax ignores."""
    x = np.asarray(log_probs, dtype=np.float32)
    return x - x.mean(axis=-1, keepdims=True)


def project(log_probs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """c = X B for X of shape (N, V) and basis (V, r)."""
    return np.asarray(log_probs, dtype=np.float32) @ basis


def reconstruct(coefs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """X_hat = c B^T, returned as logits of shape (N, V)."""
    return np.asarray(coefs, dtype=np.float32) @ basis.T


def _align_signs(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    signs = np.sign(np.einsum("vr,vr->r", new, old))
    signs[signs == 0] = 1.0
    return new * signs


def power_update(basis: np.ndarray, log_probs: np.ndarray, iters: int = 1,
                 damping: float = 0.5) -> np.ndarray:
    """One damped power iteration on the second moment of the draft logits.

    basis <- orthonormalise(basis * damping + X^T (X basis))
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    x = centre(log_probs)
    b = np.asarray(basis, dtype=np.float32)
    ones = b[:, :1].copy()
    tail = b[:, 1:].copy()
    for _ in range(iters):
        update = x.T @ (x @ tail)
        mixed = damping * tail + (1.0 - damping) * update
        mixed = mixed - ones @ (ones.T @ mixed)
        new, _ = np.linalg.qr(mixed.astype(np.float64))
        tail = _align_signs(new.astype(np.float32), tail)
    if tail.shape[1] == 0:
        return ones
    return np.hstack([ones, tail])


def basis_delta(old: np.ndarray, new: np.ndarray, threshold: float,
                max_entries: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sparse delta: indices and values of entries that moved more than threshold."""
    diff = (new - old).astype(np.float32)
    flat = np.abs(diff).ravel()
    idx = np.flatnonzero(flat > threshold)
    if max_entries is not None and idx.size > max_entries:
        keep = np.argsort(-flat[idx])[:max_entries]
        idx = np.sort(idx[keep])
    return idx.astype(np.int32), diff.ravel()[idx]


def apply_delta(basis: np.ndarray, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.array(basis, dtype=np.float32, copy=True)
    if idx.size:
        out.ravel()[idx] += values
    return out


def quantise_coefs(coefs: np.ndarray, dtype: str = "float16") -> np.ndarray:
    return np.asarray(coefs, dtype=np.dtype(dtype))


def explained_energy(log_probs: np.ndarray, basis: np.ndarray) -> float:
    x = centre(log_probs)
    total = float((x ** 2).sum())
    if total == 0:
        return 1.0
    approx = reconstruct(project(log_probs, basis), basis)
    residual = centre(approx) - x
    return float(1.0 - (residual ** 2).sum() / total)
