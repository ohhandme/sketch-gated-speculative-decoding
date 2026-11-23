"""Reference end-to-end loop: edge proposal, uplink payload, cloud verification."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lowrank, metrics
from .channel import StepTiming, downlink_ms
from .config import SystemConfig
from .gate import AdaptiveGate, build_gate
from .headtail import head_indices, merge
from .models import DistributionSource, Timer
from .serialize import Payload, pack, unpack
from .sketch import deviation, build_sketch
from .verify import accept_probability, residual, sample

EPS = 1e-30

TIMING_FIELDS = (
    "t_draft", "t_sketch", "t_prepare_edge", "t_serialize", "t_upload", "t_propagation",
    "t_queue_edge", "t_queue_cloud", "t_prepare_cloud", "t_gate", "t_verify",
    "t_resample", "t_accept", "t_fallback_trigger", "t_fallback_upload", "t_downlink",
)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def safe_log(prob: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(prob, dtype=np.float64), EPS))


def tail_features(q_block, head_idx):
    log_q = safe_log(q_block)
    mask = np.ones(log_q.shape, dtype=bool)
    np.put_along_axis(mask, head_idx, False, axis=1)
    mean = (log_q * mask).sum(axis=1, keepdims=True) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return np.where(mask, log_q - mean, 0.0)


@dataclass
class PositionTrace:
    index: int
    score: float
    accept_approx: bool
    accept_full: bool
    p_taken: float
    q_tilde_taken: float
    q_full_taken: float


@dataclass
class StepRecord:
    index: int
    mode: str
    accepted: int
    emitted: int
    fell_back: bool
    gate_score: float
    theta: float
    bytes_small: int
    bytes_full: int
    bytes_total: int
    timings: dict = field(default_factory=dict)
    e2e_ms: float = 0.0
    per_token_ms: float = 0.0
    tokens: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    fallback_kind: str = ""  # "" | "A" | "B"
    proposed: int = 0
    link_carry_ms: float = 0.0
    t_fallback_trigger_ms: float = 0.0


@dataclass
class RunResult:
    mode: str
    tokens: list
    steps: list
    prompt: list
    meta: dict = field(default_factory=dict)

    @property
    def accepted(self) -> int:
        return sum(s.accepted for s in self.steps)

    def summary(self) -> dict:
        emitted = [s.emitted for s in self.steps]
        per_token = [s.per_token_ms for s in self.steps]
        e2e = [s.e2e_ms for s in self.steps]
        tau = metrics.acceptance_length(emitted)
        bytes_per_step = float(np.mean([s.bytes_total for s in self.steps])) if self.steps else 0.0
        fields = {}
        if self.steps:
            for key in TIMING_FIELDS:
                values = [s.timings.get(key, 0.0) for s in self.steps]
                fields[key] = metrics.percentile_summary(values)
        proposed = sum(s.proposed for s in self.steps)
        return {
            "mode": self.mode,
            "steps": len(self.steps),
            "tokens": len(self.tokens),
            "tau": self.accepted / len(self.steps) if self.steps else 0.0,
            "tokens_per_step": tau,
            "aggregate_ms_per_token": sum(e2e) / sum(emitted) if sum(emitted) else 0.0,
            "accept_rate": metrics.acceptance_rate(self.accepted, proposed),
            "fallback_rate": float(np.mean([s.fell_back for s in self.steps]))
            if self.steps else 0.0,
            "bytes_per_step": bytes_per_step,
            "bytes_per_token": bytes_per_step / tau if tau else 0.0,
            "sketch_over_draft": _ratio(fields, "t_sketch", "t_draft"),
            "per_token_ms": metrics.percentile_summary(per_token),
            "e2e_ms": metrics.percentile_summary(e2e),
            "gate_scores": metrics.percentile_summary([s.gate_score for s in self.steps]),
            "timing_fields": fields,
            "wall_seconds": None,
        }


def _ratio(fields: dict, numerator: str, denominator: str) -> float | None:
    a = fields.get(numerator, {}).get("mean", 0.0)
    b = fields.get(denominator, {}).get("mean", 0.0)
    return float(a / b) if b else None


class Engine:
    def __init__(self, cfg: SystemConfig, draft: DistributionSource, target: DistributionSource,
                 prompt: list[int], theta: float | None = None, gate: AdaptiveGate | None = None):
        if cfg.run.fallback not in ("A", "B") or cfg.run.draft_len < 1 or cfg.run.max_steps < 1:
            raise ValueError("expected fallback A/B and positive block/step counts")
        self.cfg = cfg
        self.draft = draft
        self.target = target
        self.prompt = list(prompt)
        vocab = target.vocab
        if draft.vocab != vocab:
            raise ValueError("draft and target vocabularies differ")
        self.vocab = vocab
        self.sketch = build_sketch(vocab, cfg.sketch.dim, cfg.sketch.nnz, cfg.sketch.kind,
                                   cfg.sketch.seed)
        rank = min(cfg.lowrank.rank, vocab)
        self.basis = lowrank.init_basis(vocab, rank, cfg.sketch.seed + 1)
        # The cloud keeps its own copy: it only ever sees the quantised delta.
        self.cloud_basis = self.basis.copy()
        self.theta = cfg.gate.theta if theta is None else float(theta)
        self.gate = gate
        if self.gate is None and cfg.gate.adaptive and cfg.run.mode == "ours":
            self.gate = build_gate(cfg.gate, self.theta)

        seed = cfg.run.seed
        self.rng_draft = np.random.default_rng(seed)
        self.rng_accept = np.random.default_rng(seed + 1)
        self.rng_resample = np.random.default_rng(seed + 2)
        self._link_carry_ms = 0.0
        self._emitted: list[int] = []

    # ------------------------------------------------------------------ helpers
    def _probs(self, source: DistributionSource, prefix: list[int], timers: dict,
               key: str) -> np.ndarray:
        with Timer(timers, key):
            logp = source.log_probs(prefix)
        return softmax(logp)

    def _timing(self, cfg, payload: Payload, fell_back: bool, on_wire_full: bool,
                decision_ms: float, tail_ms: float, backlog_ms: float,
                full_blocking: bool) -> StepTiming:
        return StepTiming(
            payload_small=payload.small_bytes(),
            payload_full=(payload.sizes()["full_q"] if on_wire_full else 0),
            fell_back=fell_back, pipeline=cfg.run.pipeline, rtt_ms=cfg.channel.rtt_ms,
            overhead=cfg.channel.protocol_overhead_bytes, backlog_ms=backlog_ms,
            decision_ms=decision_ms, tail_ms=tail_ms, full_blocking=full_blocking)

    # ------------------------------------------------------------------ the loop
    def run(self, calibrate: bool = False) -> RunResult:
        cfg = self.cfg
        if cfg.run.mode not in ("cloud_only", "full_q", "token_only", "ours"):
            raise ValueError(f"unknown mode {cfg.run.mode}")
        prefix = list(self.prompt)
        generated: list[int] = []
        steps: list[StepRecord] = []
        for index in range(cfg.run.max_steps):
            if cfg.run.mode == "cloud_only":
                record = self._step_cloud_only(prefix, index)
            elif cfg.run.mode == "token_only":
                record = self._step_token_only(prefix, index)
            elif cfg.run.mode == "full_q":
                record = self._step_full_q(prefix, index)
            else:
                record = self._step_ours(prefix, index, calibrate)
            prefix.extend(record.tokens)
            generated.extend(record.tokens)
            steps.append(record)
            if self.gate is not None:
                self.gate.observe(record.fell_back)
                self.theta = self.gate.theta
        return RunResult(mode=cfg.run.mode, tokens=generated, steps=steps, prompt=self.prompt,
                         meta={"theta": self.theta, "rank": cfg.lowrank.rank,
                               "sketch_dim": cfg.sketch.dim, "vocab": self.vocab})

    def _step_cloud_only(self, prefix: list[int], index: int) -> StepRecord:
        timers: dict[str, float] = {}
        p = self._probs(self.target, prefix, timers, "t_verify")
        with Timer(timers, "t_resample"):
            token = sample(p, self.rng_resample)
        timers["t_downlink"] = downlink_ms(self.cfg.channel)
        e2e = sum(timers.values())
        return StepRecord(index=index, mode="cloud_only", accepted=0, emitted=1,
                          fell_back=False, gate_score=0.0, theta=self.theta,
                          bytes_small=0, bytes_full=0, bytes_total=0, timings=timers,
                          e2e_ms=e2e, per_token_ms=e2e, tokens=[token])

    def _draft_block(self, prefix: list[int], timers: dict):
        rows = self.cfg.run.draft_len
        tokens: list[int] = []
        q_rows = []
        wire_logits = []
        for _ in range(rows):
            with Timer(timers, "t_draft"):
                logq = self.draft.log_probs(prefix + tokens)
            wire_row = np.asarray(logq, dtype=np.float16)
            wire_logits.append(wire_row)
            q = softmax(wire_row)
            tokens.append(sample(q, self.rng_draft))
            q_rows.append(q)
        return tokens, np.stack(q_rows), np.stack(wire_logits)

    def _start_backlog(self, timers: dict, pre_keys) -> float:
        pre = sum(timers.get(k, 0.0) for k in pre_keys)
        return max(0.0, self._link_carry_ms - pre)

    # ------------------------------------------------------------------ B2
    def _step_token_only(self, prefix: list[int], index: int) -> StepRecord:
        cfg = self.cfg
        timers: dict[str, float] = {}
        tokens, _, _ = self._draft_block(prefix, timers)
        with Timer(timers, "t_serialize"):
            payload = Payload(coefs=np.empty((0, 0), dtype=np.float16),
                              sketch=np.empty(0, dtype=np.float16),
                              head_idx=np.empty((0, 0), dtype=np.int32),
                              head_prob=np.empty((0, 0), dtype=np.float16),
                              basis_idx=np.empty(0, dtype=np.int32),
                              basis_val=np.empty(0, dtype=np.float16),
                              token_ids=np.asarray(tokens, dtype=np.int32))
            payload = unpack(pack(payload))
            tokens = payload.token_ids.tolist()
        p0 = [self._target_probs(prefix, tokens[:i], timers) for i in range(len(tokens))]
        accepted, emitted, positions = self._verify_token_only(prefix, tokens, timers, p0)
        return self._finish(index, cfg.run.mode, tokens, accepted, emitted, positions, payload,
                            fell_back=False, score=0.0, timers=timers, decision_ms=0.0,
                            tail_ms=0.0, on_wire_full=False, full_blocking=False,
                            timers_pre=("t_draft", "t_serialize"), fallback_kind="")

    # ------------------------------------------------------------------ B1
    def _step_full_q(self, prefix: list[int], index: int) -> StepRecord:
        cfg = self.cfg
        timers: dict[str, float] = {}
        tokens, q_block, wire_logits = self._draft_block(prefix, timers)
        with Timer(timers, "t_serialize"):
            payload = Payload(coefs=np.empty((len(tokens), 0), dtype=np.float16),
                              sketch=np.empty(0, dtype=np.float16),
                              head_idx=np.empty((0, 0), dtype=np.int32),
                              head_prob=np.empty((0, 0), dtype=np.float16),
                              basis_idx=np.empty(0, dtype=np.int32),
                              basis_val=np.empty(0, dtype=np.float16),
                              token_ids=np.asarray(tokens), full_q=wire_logits)
            payload = unpack(pack(payload))
            tokens = payload.token_ids.tolist()
            q_block = [softmax(row) for row in payload.full_q]
        p0 = [self._target_probs(prefix, tokens[:i], timers) for i in range(len(tokens))]
        accepted, emitted, positions = self._verify_full(prefix, tokens, q_block, timers, p0)
        return self._finish(index, "full_q", tokens, accepted, emitted, positions, payload,
                            fell_back=False, score=0.0, timers=timers, decision_ms=0.0,
                            tail_ms=0.0, on_wire_full=True, full_blocking=True,
                            timers_pre=("t_draft", "t_serialize"), fallback_kind="")

    # ------------------------------------------------------------------ ours
    def _step_ours(self, prefix: list[int], index: int, calibrate: bool) -> StepRecord:
        cfg = self.cfg
        timers: dict[str, float] = {}
        self._emitted = []
        tokens, q_block, wire_logits = self._draft_block(prefix, timers)

        with Timer(timers, "t_sketch"):
            s_q = self.sketch.apply(q_block)

        with Timer(timers, "t_prepare_edge"):
            kt = min(cfg.head.tokens, self.vocab)
            head_idx = np.stack([head_indices(q, kt) for q in q_block])
            head_prob = np.take_along_axis(q_block, head_idx, axis=1)
            log_q = tail_features(q_block, head_idx)
            if cfg.lowrank.update_every and index % cfg.lowrank.update_every == 0:
                updated = lowrank.power_update(self.basis, log_q, cfg.lowrank.power_iters,
                                               cfg.lowrank.damping)
                delta_idx, delta_val = lowrank.basis_delta(
                    self.basis, updated, cfg.lowrank.delta_threshold,
                    cfg.lowrank.max_delta_entries)
            else:
                delta_idx = np.empty(0, dtype=np.int32)
                delta_val = np.empty(0, dtype=np.float32)
            # Both ends apply the *quantised* delta, so their bases never drift.
            delta_val16 = delta_val.astype(np.float16).astype(np.float32)
            basis_sent = lowrank.apply_delta(self.basis, delta_idx, delta_val16)
            coefs = lowrank.project(log_q, basis_sent)
            self.basis = basis_sent

        optimistic = cfg.run.fallback == "A" and cfg.run.pipeline
        with Timer(timers, "t_serialize"):
            payload = unpack(pack(Payload(coefs=coefs, sketch=s_q.astype(np.float16),
                head_idx=head_idx, head_prob=head_prob, token_ids=np.asarray(tokens),
                basis_idx=delta_idx, basis_val=delta_val16,
                full_q=wire_logits if optimistic else None)))
            tokens = payload.token_ids.tolist()

        with Timer(timers, "t_prepare_cloud"):
            self.cloud_basis = lowrank.apply_delta(
                self.cloud_basis, payload.basis_idx, payload.basis_val.astype(np.float32))
            tail_logits = lowrank.reconstruct(payload.coefs.astype(np.float64), self.cloud_basis)
            q_tilde = np.stack([
                merge(payload.head_idx[i].astype(np.int64),
                      payload.head_prob[i].astype(np.float64), tail_logits[i])
                for i in range(len(tokens))
            ])

        for key in TIMING_FIELDS:
            timers.setdefault(key, 0.0)
        target_rows = [self._target_probs(prefix, tokens[:i], timers) for i in range(len(tokens))]
        with Timer(timers, "t_gate"):
            received = payload.sketch.reshape(len(tokens), cfg.sketch.dim)
            scores = deviation(received, self.sketch.apply(np.stack(target_rows)), cfg.gate.score)
            score = float(np.max(scores))
        with Timer(timers, "t_fallback_trigger"):
            fell_back = score > self.theta
        decision_ms = sum(timers[k] for k in ("t_prepare_cloud", "t_verify", "t_gate", "t_fallback_trigger"))
        if fell_back and cfg.run.fallback == "A" and not optimistic:
            with Timer(timers, "t_serialize"):
                payload.full_q = np.frombuffer(wire_logits.tobytes(), np.float16).reshape(wire_logits.shape)
        if fell_back and cfg.run.fallback == "A":
            exact_q = [softmax(row) for row in payload.full_q]
            accepted, emitted, positions = self._verify_full(prefix, tokens, exact_q, timers, target_rows)
        elif fell_back:
            accepted, emitted, positions = self._verify_token_only(prefix, tokens, timers, target_rows)
        else:
            accepted, emitted, positions = self._verify_approx(
                prefix, tokens, q_tilde, q_block if calibrate else None, timers, target_rows,
                calibrate, scores=scores)

        on_wire_full = (cfg.run.fallback == "A") and (cfg.run.pipeline or fell_back)
        cloud_ms = sum(timers[k] for k in ("t_verify", "t_prepare_cloud", "t_gate",
                                            "t_fallback_trigger", "t_accept", "t_resample"))
        tail_ms = cloud_ms - decision_ms + cfg.channel.rtt_ms
        record = self._finish(
            index, "ours", tokens, accepted, emitted, positions, payload, fell_back=fell_back,
            score=score, timers=timers, decision_ms=decision_ms, tail_ms=tail_ms,
            on_wire_full=on_wire_full, full_blocking=False,
            timers_pre=("t_draft", "t_sketch", "t_prepare_edge", "t_serialize"),
            fallback_kind=("A" if (fell_back and cfg.run.fallback == "A") else
                           ("B" if fell_back else "")))
        return record

    # ------------------------------------------------------------------ record
    def _finish(self, index, mode, tokens, accepted, emitted, positions, payload, fell_back,
                score, timers, decision_ms, tail_ms, on_wire_full, full_blocking,
                timers_pre, fallback_kind) -> StepRecord:
        cfg = self.cfg
        timers["t_downlink"] = downlink_ms(cfg.channel)
        backlog = self._start_backlog(timers, timers_pre)
        timing = self._timing(cfg, payload, fell_back, on_wire_full, decision_ms, tail_ms,
                              backlog, full_blocking)
        upload = timing.upload_ms(cfg.channel.bandwidth_mbps)
        timers["t_upload"] = upload["upload"]
        timers["t_propagation"] = upload["propagation"]
        timers["t_fallback_upload"] = upload["wait_fallback"]
        self._link_carry_ms = upload["carry"]

        for key in TIMING_FIELDS:
            timers.setdefault(key, 0.0)
        e2e = sum(timers.values())
        sizes = payload.sizes()
        small_bytes = payload.small_bytes()
        total_bytes = (small_bytes + (sizes["full_q"] if on_wire_full else 0) +
                       cfg.channel.protocol_overhead_bytes)
        return StepRecord(index=index, mode=mode, accepted=accepted, emitted=emitted,
                          fell_back=fell_back, gate_score=score, theta=self.theta,
                          proposed=len(tokens), bytes_small=small_bytes, bytes_full=sizes["full_q"] if on_wire_full else 0,
                          bytes_total=total_bytes, timings=timers, e2e_ms=e2e,
                          per_token_ms=e2e / max(1, emitted), tokens=self._emitted[:],
                          positions=positions, fallback_kind=fallback_kind,
                          link_carry_ms=upload["carry"],
                          t_fallback_trigger_ms=timers["t_fallback_trigger"])

    # ------------------------------------------------------------------ verifiers
    def _target_probs(self, prefix, extra, timers) -> np.ndarray:
        return self._probs(self.target, prefix + list(extra), timers, "t_verify")

    def _verify_full(self, prefix, tokens, q_block, timers, p0):
        accepted, emitted, positions = 0, [], []
        for i, token in enumerate(tokens):
            p = p0[i]
            with Timer(timers, "t_accept"):
                u = self.rng_accept.random()
                prob = accept_probability(p[token], q_block[i][token])
            if u < prob:
                accepted += 1
                emitted.append(token)
                continue
            with Timer(timers, "t_resample"):
                nxt = sample(residual(p, q_block[i]), self.rng_resample)
            emitted.append(nxt)
            self._emitted = emitted
            return accepted, len(emitted), positions
        if self.cfg.run.bonus:
            p = self._target_probs(prefix, emitted, timers)
            with Timer(timers, "t_resample"):
                emitted.append(sample(p, self.rng_resample))
        self._emitted = emitted
        return accepted, len(emitted), positions

    def _verify_token_only(self, prefix, tokens, timers, p0):
        emitted: list[int] = []
        for i, token in enumerate(tokens):
            p = p0[i]
            with Timer(timers, "t_accept"):
                exact = sample(p, self.rng_resample)
            if token == exact:
                emitted.append(token)
                continue
            emitted.append(exact)
            self._emitted = emitted
            return len(emitted) - 1, len(emitted), []
        if self.cfg.run.bonus:
            p = self._target_probs(prefix, emitted, timers)
            emitted.append(sample(p, self.rng_resample))
        self._emitted = emitted
        return len(tokens), len(emitted), []

    def _verify_approx(self, prefix, tokens, q_tilde, q_block, timers, p0, calibrate,
                       scores):
        emitted: list[int] = []
        positions: list[PositionTrace] = []
        for i, token in enumerate(tokens):
            p = p0[i]
            with Timer(timers, "t_accept"):
                u = self.rng_accept.random()
                prob_approx = accept_probability(p[token], q_tilde[i][token])
                ok_approx = u < prob_approx
            if calibrate:
                with Timer(timers, "t_gate"):
                    pos_score = float(scores[i])
                    ok_full = u < accept_probability(p[token], q_block[i][token])
                positions.append(PositionTrace(index=i, score=pos_score, accept_approx=ok_approx,
                                               accept_full=ok_full, p_taken=float(p[token]),
                                               q_tilde_taken=float(q_tilde[i][token]),
                                               q_full_taken=float(q_block[i][token])))
            if ok_approx:
                emitted.append(token)
                continue
            # The compressed path discarded full Q; use the target-only correction.
            with Timer(timers, "t_resample"):
                nxt = sample(p, self.rng_resample)
            emitted.append(nxt)
            self._emitted = emitted
            return len(emitted) - 1, len(emitted), positions
        if self.cfg.run.bonus:
            p = self._target_probs(prefix, emitted, timers)
            with Timer(timers, "t_resample"):
                emitted.append(sample(p, self.rng_resample))
        self._emitted = emitted
        return len(tokens), len(emitted), positions
