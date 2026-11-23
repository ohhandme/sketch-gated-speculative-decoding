"""Shared-seed random sketches and the two gate scores from spec §4.1."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sketch:
    """A k x V projection with a fixed number of nonzeros per row."""

    vocab: int
    dim: int
    nnz: int
    kind: str
    seed: int
    columns: np.ndarray  # (dim, nnz) int32
    values: np.ndarray  # (dim, nnz) float32

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Return s = R p for p of shape (V,) or (N, V)."""
        p = np.asarray(probs, dtype=np.float32)
        single = p.ndim == 1
        if single:
            p = p[None, :]
        if p.shape[-1] != self.vocab:
            raise ValueError(f"expected last dim {self.vocab}, got {p.shape}")
        gathered = np.take_along_axis(p[:, None, :], self.columns[None, :, :], axis=2)
        out = np.einsum("nkd,kd->nk", gathered, self.values)
        return out[0] if single else out


def build_sketch(vocab: int, dim: int, nnz: int, kind: str, seed: int) -> Sketch:
    if vocab <= 0 or dim <= 0:
        raise ValueError("vocab and dim must be positive")
    if not 1 <= nnz <= vocab:
        raise ValueError("nnz must be in [1, vocab]")
    if kind not in ("sparse_pm1", "sparse_gauss"):
        raise ValueError(f"unknown sketch kind {kind!r}")
    rng = np.random.default_rng(seed)
    # Independent uniform sample per row without replacement.
    keys = rng.random((dim, vocab), dtype=np.float64)
    columns = np.argpartition(keys, nnz - 1, axis=1)[:, :nnz].astype(np.int32)
    columns.sort(axis=1)
    if kind == "sparse_pm1":
        signs = rng.integers(0, 2, size=(dim, nnz), dtype=np.int8) * 2 - 1
        values = signs.astype(np.float32)
    else:
        values = (rng.standard_normal((dim, nnz)) / np.sqrt(nnz)).astype(np.float32)
    return Sketch(vocab=vocab, dim=dim, nnz=nnz, kind=kind, seed=seed,
                  columns=columns, values=values)


def l1_score(sq: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """a = ||s_Q - s_P||_1 / k, per position."""
    sq = np.asarray(sq, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)
    if sq.shape != sp.shape:
        raise ValueError(f"shape mismatch {sq.shape} vs {sp.shape}")
    return np.abs(sq - sp).sum(axis=-1) / sq.shape[-1]


def cosine_score(sq: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """a = 1 - cos(s_Q, s_P), per position."""
    sq = np.asarray(sq, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)
    if sq.shape != sp.shape:
        raise ValueError(f"shape mismatch {sq.shape} vs {sp.shape}")
    num = (sq * sp).sum(axis=-1)
    den = np.linalg.norm(sq, axis=-1) * np.linalg.norm(sp, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - np.where(den > 0, num / den, 0.0)
    return out


def l1_tv_bound(nnz: int, vocab: int, tv: float) -> float:
    """Upper bound on expected score over random projections, given TV."""
    return 2.0 * nnz / vocab * tv


def deviation(sq, sp, metric):
    if metric == "l1":
        return l1_score(sq, sp)
    if metric == "cosine":
        return cosine_score(sq, sp)
    raise ValueError(f"unknown sketch metric {metric}")
