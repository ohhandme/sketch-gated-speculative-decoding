"""Offline calibration: collect gate scores and truth costs, pick theta (spec §3.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import kl
from . import lowrank
from .config import SystemConfig
from .headtail import head_indices, merge
from .models import DistributionSource
from .engine import tail_features, softmax
from .sketch import build_sketch, deviation
from .verify import accept_probability


@dataclass
class CalibrationData:
    scores: np.ndarray
    flips: np.ndarray
    accept_gap: np.ndarray
    tv: np.ndarray

    block_scores: np.ndarray | None = None
    kl_pq: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.scores.size)


def collect(cfg: SystemConfig, draft: DistributionSource, target: DistributionSource,
            prompt: list[int], steps: int | None = None,
            rng: np.random.Generator | None = None) -> CalibrationData:
    """Measure the gate score and the accept-flip indicator at every draft position.

    Positions are measured on the "all previous tokens accepted" path, which is
    the counterfactual the gate is supposed to predict. No fallback is applied.
    """
    steps = cfg.run.max_steps if steps is None else steps
    rng = rng or np.random.default_rng(cfg.run.seed)
    rng_accept = np.random.default_rng(cfg.run.seed + 1)
    vocab = int(target.vocab)
    if int(draft.vocab) != vocab:
        raise ValueError("draft and target vocabularies differ")
    sketch = build_sketch(vocab, cfg.sketch.dim, cfg.sketch.nnz,
                          cfg.sketch.kind, cfg.sketch.seed)
    basis = lowrank.init_basis(vocab, min(cfg.lowrank.rank, vocab),
                               cfg.sketch.seed + 1)
    kt = min(cfg.head.tokens, vocab)
    prefix = list(prompt)
    scores: list[float] = []
    flips: list[int] = []
    gaps: list[float] = []
    tvs: list[float] = []
    kls = []
    block_scores = []
    for step_index in range(steps):
        tokens: list[int] = []
        q_rows = []
        for _ in range(cfg.run.draft_len):
            q = softmax(draft.log_probs(prefix + tokens).astype(np.float16))
            c = np.cumsum(q)
            tokens.append(int(np.searchsorted(c, rng.random() * c[-1]).clip(0, q.size - 1)))
            q_rows.append(q)
        q_block = np.stack(q_rows)
        heads = np.stack([head_indices(q, kt) for q in q_block])
        log_q = tail_features(q_block, heads)
        if cfg.lowrank.update_every and step_index % cfg.lowrank.update_every == 0:
            updated = lowrank.power_update(basis, log_q, cfg.lowrank.power_iters,
                                           cfg.lowrank.damping)
            idx, val = lowrank.basis_delta(basis, updated, cfg.lowrank.delta_threshold,
                                           cfg.lowrank.max_delta_entries)
        else:
            idx, val = np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        basis = lowrank.apply_delta(basis, idx, val.astype(np.float16))
        coefs = lowrank.project(log_q, basis)
        tail = lowrank.reconstruct(coefs.astype(np.float16), basis)
        for i, token in enumerate(tokens):
            head = heads[i]
            q_tilde = merge(head, q_block[i][head], tail[i])
            p = softmax(target.log_probs(prefix + tokens[:i]))
            s_q = sketch.apply(q_block[i])
            s_p = sketch.apply(p)
            scores.append(float(deviation(s_q.astype(np.float16), s_p, cfg.gate.score)))
            u = rng_accept.random()
            ok_approx = u < accept_probability(p[token], q_tilde[token])
            ok_full = u < accept_probability(p[token], q_block[i][token])
            flips.append(int(ok_approx != ok_full))
            alpha = np.minimum(p, q_block[i]).sum()
            ratio = np.divide(p, q_tilde, out=np.ones_like(p), where=q_tilde > 0)
            gaps.append(float(alpha - q_block[i] @ np.minimum(1.0, ratio)))
            kls.append(kl(p, q_block[i]))
            tvs.append(float(0.5 * np.abs(p - q_block[i]).sum()))
        block_scores.append(max(scores[-len(tokens):]))
        prefix.extend(tokens)
    return CalibrationData(scores=np.asarray(scores), flips=np.asarray(flips),
                           accept_gap=np.asarray(gaps), tv=np.asarray(tvs), block_scores=np.asarray(block_scores), kl_pq=np.asarray(kls))


def quantiles(data: CalibrationData, values=(0.90, 0.95, 0.99)) -> dict[str, float]:
    return {f"q{int(v * 100)}": float(np.quantile(data.scores if data.block_scores is None else data.block_scores, v)) for v in values}


def sweep_table(data: CalibrationData, thresholds: list[float]) -> list[dict]:
    rows = []
    for theta in thresholds:
        blocks = data.scores if data.block_scores is None else data.block_scores
        over = np.repeat(blocks > theta, len(data.scores) // len(blocks))
        rows.append({
            "theta": float(theta),
            "fallback_rate": float(over.mean()) if over.size else 0.0,
            "flip_rate_on_gated": float(data.flips[~over].mean()) if (~over).any() else 0.0,
            "flip_rate_overall": float(data.flips.mean()) if data.flips.size else 0.0,
            "mean_tv": float(data.tv.mean()) if data.tv.size else 0.0,
        })
    return rows
