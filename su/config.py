"""Configuration objects and a YAML/JSON loader with sane defaults."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

try:  # pyyaml is optional for JSON configs
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class SketchConfig:
    dim: int = 128
    nnz: int = 32
    kind: str = "sparse_pm1"  # sparse_pm1 | sparse_gauss
    seed: int = 1234


@dataclass
class LowRankConfig:
    rank: int = 4
    power_iters: int = 1
    damping: float = 0.5
    delta_threshold: float = 1e-3
    update_every: int = 8
    max_delta_entries: int = 512


@dataclass
class HeadConfig:
    tokens: int = 32


@dataclass
class GateConfig:
    theta: float = 0.02
    quantile: float = 0.95
    score: str = "l1"  # l1 | cosine
    adaptive: bool = False
    ema_window: int = 500
    adapt_step: float = 1e-3
    theta_min: float | None = None
    theta_max: float | None = None
    target_fallback: float = 0.10


@dataclass
class ChannelConfig:
    bandwidth_mbps: float = 10.0
    rtt_ms: float = 30.0
    protocol_overhead_bytes: int = 64
    downlink_bytes: int = 8


@dataclass
class RunConfig:
    draft_len: int = 4
    seed: int = 42
    max_steps: int = 32
    mode: str = "ours"  # cloud_only | full_q | token_only | ours
    fallback: str = "B"  # A | B
    pipeline: bool = True  # pipelined vs sync_fallback
    bonus: bool = True
    q_dtype: str = "float16"


@dataclass
class ModelConfig:
    draft: str = "tiny"  # "tiny" or a HF model directory
    target: str = "tiny"
    vocab_size: int = 2048
    draft_hidden: int = 64
    draft_layers: int = 2
    target_hidden: int = 96
    target_layers: int = 3
    heads: int = 4
    kv_heads: int = 2
    intermediate: int = 128
    prompt_len: int = 8


@dataclass
class SystemConfig:
    sketch: SketchConfig = field(default_factory=SketchConfig)
    lowrank: LowRankConfig = field(default_factory=LowRankConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "sketch": SketchConfig,
    "lowrank": LowRankConfig,
    "head": HeadConfig,
    "gate": GateConfig,
    "channel": ChannelConfig,
    "run": RunConfig,
    "model": ModelConfig,
}


def _build(section: str, values: dict[str, Any] | None):
    cls = _SECTIONS[section]
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {section} keys: {sorted(unknown)}")
    return cls(**values)


def from_dict(raw: dict[str, Any]) -> SystemConfig:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return SystemConfig(**{name: _build(name, raw.get(name)) for name in _SECTIONS})


def load_config(path: str | Path | None) -> SystemConfig:
    if path is None:
        return SystemConfig()
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML configs")
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    return from_dict(raw)


def describe(cfg: SystemConfig) -> dict[str, Any]:
    """Flat summary used in run metadata."""
    return {
        "mode": cfg.run.mode,
        "rank": cfg.lowrank.rank,
        "sketch_dim": cfg.sketch.dim,
        "sketch_nnz": cfg.sketch.nnz,
        "head_tokens": cfg.head.tokens,
        "theta": cfg.gate.theta,
        "quantile": cfg.gate.quantile,
        "bandwidth_mbps": cfg.channel.bandwidth_mbps,
        "rtt_ms": cfg.channel.rtt_ms,
        "draft_len": cfg.run.draft_len,
        "fallback": cfg.run.fallback,
        "pipeline": cfg.run.pipeline,
    }
