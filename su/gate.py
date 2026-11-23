"""Threshold quantiles, gate decisions and the optional slow adaptation (spec §3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GateConfig


def quantile_theta(scores: np.ndarray, quantile: float) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("no calibration scores")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    return float(np.quantile(s, quantile))


def fallback_rate(scores: np.ndarray, theta: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    return float((s > theta).mean()) if s.size else 0.0


@dataclass
class AdaptiveGate:
    """Slow EMA controller: only nudges theta on long windows (spec §3.2)."""

    theta: float
    target_fallback: float
    step: float
    window: int = 500
    lower: float | None = None
    upper: float | None = None
    _seen: int = 0
    _fallbacks: int = 0
    _history: list[float] = field(default_factory=list)

    def observe(self, fell_back: bool) -> float:
        self._seen += 1
        self._fallbacks += int(fell_back)
        if self._seen % max(1, self.window) == 0:
            rate = self._fallbacks / self._seen
            # Fall back too often -> raise theta; too rarely -> lower it.
            self.theta += self.step * (1.0 if rate > self.target_fallback else -1.0)
            if self.lower is not None:
                self.theta = max(self.theta, self.lower)
            if self.upper is not None:
                self.theta = min(self.theta, self.upper)
            self._history.append(self.theta)
            self._seen = 0
            self._fallbacks = 0
        return self.theta


def build_gate(cfg: GateConfig, theta: float | None = None) -> AdaptiveGate:
    return AdaptiveGate(theta=cfg.theta if theta is None else theta,
                        target_fallback=cfg.target_fallback,
                        step=cfg.adapt_step,
                        window=cfg.ema_window,
                        lower=cfg.theta_min,
                        upper=cfg.theta_max)
