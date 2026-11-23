"""Experiment tables, the rate-latency curve and the §5.2 ablations."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

from . import metrics
from .config import SystemConfig
from .engine import Engine
from .models import DistributionSource


def run_mode(cfg: SystemConfig, draft: DistributionSource, target: DistributionSource,
             prompt: list[int], mode: str, theta: float | None = None) -> dict:
    local = replace(cfg, run=replace(cfg.run, mode=mode))
    engine = Engine(local, draft, target, prompt, theta=theta)
    result = engine.run()
    summary = result.summary()
    return summary


def bandwidth_sweep(cfg: SystemConfig, draft: DistributionSource, target: DistributionSource,
                    prompt: list[int], bandwidths=(1, 5, 10, 50, 100),
                    modes=("cloud_only", "full_q", "token_only", "ours"),
                    theta: float | None = None) -> dict:
    curves: dict[str, list[dict]] = {}
    for bw in bandwidths:
        local = replace(cfg, channel=replace(cfg.channel, bandwidth_mbps=float(bw)))
        for mode in modes:
            summary = run_mode(local, draft, target, prompt, mode, theta=theta)
            curves.setdefault(mode, []).append({
                "bandwidth_mbps": float(bw),
                "per_token_ms": summary["aggregate_ms_per_token"],
                "tau": summary["tau"],
                "bytes_per_step": summary["bytes_per_step"],
                "fallback_rate": summary["fallback_rate"],
            })
    xs = [float(b) for b in bandwidths]
    cross = {}
    if "ours" in curves and "token_only" in curves:
        cross["ours_below_token_only_mbps"] = metrics.crosspoint(
            xs, [r["per_token_ms"] for r in curves["ours"]],
            [r["per_token_ms"] for r in curves["token_only"]])
    if "ours" in curves and "full_q" in curves:
        cross["ours_below_full_q_mbps"] = metrics.crosspoint(
            xs, [r["per_token_ms"] for r in curves["ours"]],
            [r["per_token_ms"] for r in curves["full_q"]])
        cross["full_q_below_ours_mbps"] = metrics.crosspoint(
            xs, [r["per_token_ms"] for r in curves["full_q"]],
            [r["per_token_ms"] for r in curves["ours"]])
    return {"curves": curves, "crosspoints": cross}


def main_table(results: dict[str, dict]) -> list[dict]:
    rows = []
    for mode, summary in results.items():
        rows.append({
            "mode": mode,
            "tau": summary["tau"],
            "per_token_ms": summary["aggregate_ms_per_token"],
            "bytes_per_step": summary["bytes_per_step"],
            "bytes_per_token": summary["bytes_per_token"],
            "accept_rate": summary["accept_rate"],
            "sketch_over_draft": summary["sketch_over_draft"],
            "fallback_rate": summary["fallback_rate"],
            "tokens": summary["tokens"],
            "tokens_per_step": summary["tokens_per_step"],
        })
    full = next((r for r in rows if r["mode"] == "full_q"), None)
    if full and full["bytes_per_token"]:
        for row in rows:
            row["uplink_ratio_vs_full_q"] = row["bytes_per_token"] / full["bytes_per_token"]
    return rows


def _variant(cfg: SystemConfig, *, rank: int | None = None, sketch_dim: int | None = None,
             pipeline: bool | None = None, fallback: str | None = None) -> SystemConfig:
    local = cfg
    if rank is not None:
        local = replace(local, lowrank=replace(local.lowrank, rank=rank))
    if sketch_dim is not None:
        nnz = min(local.sketch.nnz, sketch_dim, local.model.vocab_size)
        local = replace(local, sketch=replace(local.sketch, dim=sketch_dim, nnz=nnz))
    run = local.run
    if pipeline is not None:
        run = replace(run, pipeline=pipeline)
    if fallback is not None:
        run = replace(run, fallback=fallback)
    return replace(local, run=run)


def ablation(cfg: SystemConfig, draft: DistributionSource, target: DistributionSource,
             prompt: list[int], thetas: dict[str, float] | None = None,
             ranks=(2, 4, 8), sketch_dims=(64, 128, 256)) -> dict:
    """The four ablations the spec marks as mandatory (§5.2).

    ``thetas`` maps a label (``q90``/``q95``/``q99``) to a threshold measured on
    the calibration split; with no mapping the config theta is used everywhere.
    """
    thetas = dict(thetas or {})
    base_theta = thetas.get("q95", cfg.gate.theta)
    out: dict[str, list[dict]] = {}

    def row(summary: dict, **extra) -> dict:
        return {"tau": summary["tau"],
                "per_token_ms": summary["aggregate_ms_per_token"],
                "bytes_per_step": summary["bytes_per_step"],
                "fallback_rate": summary["fallback_rate"],
                "accept_rate": summary["accept_rate"],
                **extra}

    # 1. gate on/off: theta = inf disables the gate (always trust Q-tilde).
    off = run_mode(_variant(cfg), draft, target, prompt, "ours", theta=float("inf"))
    on = run_mode(_variant(cfg), draft, target, prompt, "ours", theta=base_theta)
    out["gate_on_off"] = [row(off, variant="gate_off", theta=float("inf")),
                          row(on, variant="gate_on", theta=float(base_theta))]

    # 2. sync vs pipelined with forced fallback, so the hidden RTT is the only
    # difference. Fallback-A is the only strategy that ever ships a full Q.
    sync_rows = []
    for pipeline in (False, True):
        forced = run_mode(_variant(cfg, pipeline=pipeline, fallback="A"), draft, target, prompt,
                          "ours", theta=0.0)
        sync_rows.append(row(forced, variant="pipelined" if pipeline else "sync_fallback",
                             theta=0.0))
    out["sync_vs_pipelined"] = sync_rows

    # 3. quantile curve.
    theta_rows = []
    for label, theta in sorted(thetas.items()):
        summary = run_mode(_variant(cfg), draft, target, prompt, "ours", theta=theta)
        theta_rows.append(row(summary, variant=label, theta=float(theta)))
    out["theta_quantiles"] = theta_rows

    # 4. rank x sketch-dim rate-distortion grid.
    grid_rows = []
    for rank, dim in product(ranks, sketch_dims):
        summary = run_mode(_variant(cfg, rank=rank, sketch_dim=dim), draft, target, prompt,
                           "ours", theta=base_theta)
        grid_rows.append(row(summary, variant=f"r{rank}_k{dim}", rank=rank, sketch_dim=dim,
                             theta=float(base_theta)))
    out["rank_x_sketch"] = grid_rows
    return out
