"""Link model and per-step latency accounting (spec §2.6 and §6).

The model is a *serialised* uplink: bytes are put on the wire in the order
small-payload -> optional full-Q, and a step cannot start its upload while the
previous step's optimistic full-Q chunk is still draining (``backlog_ms``).
It is not a packet-level simulator, and queueing on either host is zero
(``t_queue_*``), which the README states explicitly.

Three transfer shapes are supported:

* ``full_blocking`` - B1: the N x V logits must arrive before verification.
* ``pipeline``      - optimistic full-Q streaming behind the small payload; it
  is discarded when the gate passes and otherwise costs only whatever has not
  arrived by the gate decision.
* ``sync``          - the full-Q is requested only after the gate rejects, so it
  pays a full RTT plus the transfer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ChannelConfig


def bytes_per_ms(bandwidth_mbps: float) -> float:
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth must be positive")
    return bandwidth_mbps * 1e6 / 8.0 / 1e3


@dataclass
class StepTiming:
    payload_small: int
    payload_full: int
    fell_back: bool
    pipeline: bool
    rtt_ms: float
    overhead: int
    backlog_ms: float = 0.0
    decision_ms: float = 0.0
    tail_ms: float = 0.0
    full_blocking: bool = False

    def upload_ms(self, bandwidth_mbps: float) -> dict[str, float]:
        rate = bytes_per_ms(bandwidth_mbps)
        small = (self.payload_small + self.overhead) / rate
        full = self.payload_full / rate if self.payload_full else 0.0
        prop = self.rtt_ms / 2.0
        out = {"backlog": self.backlog_ms, "small_tx": small, "full_tx": full,
               "propagation": prop, "upload": 0.0, "wait_fallback": 0.0, "carry": 0.0}
        start = self.backlog_ms

        if self.full_blocking and self.payload_full:
            out["upload"] = start + small + full
            return out

        if self.pipeline:
            out["upload"] = start + small
            if self.payload_full:
                if self.fell_back:
                    # The chunk started right after the small payload; the cloud
                    # only waits for whatever has not arrived at decision time.
                    out["wait_fallback"] = max(0.0, full - self.decision_ms)
                out["carry"] = max(0.0, full - (self.decision_ms + out["wait_fallback"] + self.tail_ms))
            return out

        out["upload"] = start + small
        if self.fell_back and self.payload_full:
            out["wait_fallback"] = self.rtt_ms + full
        return out


def downlink_ms(cfg: ChannelConfig) -> float:
    return cfg.rtt_ms / 2.0 + (cfg.downlink_bytes + cfg.protocol_overhead_bytes) / bytes_per_ms(
        cfg.bandwidth_mbps)
