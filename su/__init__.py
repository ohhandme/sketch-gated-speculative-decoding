"""Sketch-gated low-rank uplink for speculative decoding (reference implementation).

Implements the spec in `program.md`: the edge sends a low-rank representation of
its draft log-probabilities, a random sketch of the draft distribution, exact
head probabilities and — when the gate fires — an optional full-Q payload.
"""

from .config import SystemConfig, load_config

__all__ = ["SystemConfig", "load_config"]
