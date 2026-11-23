"""Command line entry points: selftest, calibrate, run, sweep, ablation, all."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import calibration, metrics, tables
from .config import describe, load_config
from .engine import Engine
from .models import build_pair


def _prompt(vocab: int, length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=length).tolist()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def _make(args, cfg, calibration_split=False):
    draft, target = build_pair(cfg.model, cfg.run.seed)
    prompt = _prompt(draft.vocab, cfg.model.prompt_len, cfg.run.seed)
    if calibration_split:
        heldout = _prompt(draft.vocab, cfg.model.prompt_len, cfg.run.seed + 100003)
        if heldout == prompt:
            raise ValueError("calibration and evaluation prompts overlap")
        prompt = heldout
    return draft, target, prompt


def cmd_selftest(args, cfg, theta: float | None = None) -> dict:
    started = time.perf_counter()
    draft, target, prompt = _make(args, cfg)
    theta = cfg.gate.theta if theta is None else theta
    results = {}
    for mode in ("cloud_only", "token_only", "full_q", "ours"):
        local = replace(cfg, run=replace(cfg.run, mode=mode))
        result = Engine(local, draft, target, prompt, theta=theta).run()
        results[mode] = result.summary()
    return {"config": describe(cfg), "results": results,
            "wall_seconds": round(time.perf_counter() - started, 2)}


def cmd_run(args, cfg, theta: float | None = None) -> dict:
    local = replace(cfg, run=replace(cfg.run, mode=args.mode))
    draft, target, prompt = _make(args, local)
    engine = Engine(local, draft, target, prompt, theta=args.theta if theta is None else theta)
    result = engine.run(calibrate=args.calibrate)
    return {"config": describe(local), "summary": result.summary(),
            "steps": [asdict(s) for s in result.steps], "tokens": result.tokens}


def cmd_calibrate(args, cfg, theta: float | None = None) -> dict:
    draft, target, prompt = _make(args, cfg, calibration_split=True)
    data = calibration.collect(cfg, draft, target, prompt, steps=args.steps)
    qs = calibration.quantiles(data)
    thresholds = [qs[k] for k in sorted(qs)]
    table = calibration.sweep_table(data, thresholds)
    baseline = tables.run_mode(cfg, draft, target, prompt, "full_q")["accept_rate"]
    for row in table:
        trial = tables.run_mode(cfg, draft, target, prompt, "ours", row["theta"])
        row.update(accept_rate=trial["accept_rate"], bytes_per_step=trial["bytes_per_step"],
                   ms_per_token=trial["aggregate_ms_per_token"], observed_fallback=trial["fallback_rate"])
    feasible = [row for row in table if row["observed_fallback"] <= 0.10
                and row["accept_rate"] >= 0.95 * baseline]
    best = min(feasible, key=lambda row: row["ms_per_token"]) if feasible else None
    return {
        "config": describe(cfg),
        "samples": len(data), "calibration_prompt": prompt,
        "operating_theta": best["theta"] if best else None,
        "selection_budget": "fallback <= 10%, acceptance >= 95% of B1 (smoke proxy)",
        "mean_kl_pq": float(data.kl_pq.mean()),
        "quantiles": qs,
        "auc_flip": metrics.auc(data.scores, data.flips),
        "spearman_gap": metrics.spearman(data.scores, np.abs(data.accept_gap)),
        "flip_rate": float(data.flips.mean()) if len(data) else 0.0,
        "mean_tv": float(data.tv.mean()) if len(data) else 0.0,
        "sweep": table,
    }


def cmd_sweep(args, cfg, theta: float | None = None) -> dict:
    draft, target, prompt = _make(args, cfg)
    bandwidths = [float(b) for b in args.bandwidths]
    out = tables.bandwidth_sweep(cfg, draft, target, prompt, bandwidths=bandwidths,
                                 theta=theta)
    single = {}
    for mode in ("cloud_only", "full_q", "token_only", "ours"):
        single[mode] = tables.run_mode(cfg, draft, target, prompt, mode, theta=theta)
    out["evaluation_prompt"] = prompt
    out["theta"] = cfg.gate.theta if theta is None else theta
    out["main_table"] = tables.main_table(single)
    out["subsets"] = {k: v["timing_fields"] for k, v in single.items()}
    return out


def cmd_ablation(args, cfg, theta: float | None = None) -> dict:
    draft, target, prompt = _make(args, cfg)
    cal_prompt = _prompt(draft.vocab, cfg.model.prompt_len, cfg.run.seed + 100003)
    if cal_prompt == prompt:
        raise ValueError("calibration and evaluation prompts overlap")
    data = calibration.collect(cfg, draft, target, cal_prompt, steps=args.steps)
    thetas = calibration.quantiles(data)

    return {"config": describe(cfg), "thetas": thetas,
            "groups": tables.ablation(cfg, draft, target, prompt, thetas=thetas)}


def spec_checks(bundle: dict, cfg) -> dict:
    """Acceptance checklist from §9. ``None`` means "requires a real run"."""
    calib = bundle["calibration"]
    main = {row["mode"]: row for row in bundle["sweep"]["main_table"]}
    curves = bundle["sweep"]["curves"]
    ours, full_q, token = main.get("ours"), main.get("full_q"), main.get("token_only")
    sketch_bytes = cfg.run.draft_len * cfg.sketch.dim * 2
    full_bytes = cfg.run.draft_len * cfg.model.vocab_size * 2
    checks = {
        "sketch_overhead_le_0_1pct": (None if not ours or ours["sketch_over_draft"] is None
                                      else ours["sketch_over_draft"] <= 1e-3),
        "sketch_bytes_le_1pct_full_q": sketch_bytes <= 0.01 * full_bytes,
        "gate_auc_ge_0_9": calib["auc_flip"] >= 0.9 if np.isfinite(calib["auc_flip"]) else None,
        "fallback_le_10pct": ours["fallback_rate"] <= 0.10 if ours else None,
        "accept_loss_le_5pct": (None if not (ours and full_q and full_q["tau"])
                                else (full_q["tau"] - ours["tau"]) / full_q["tau"] <= 0.05),
        "ours_beats_token_only_somewhere": any(
            a["per_token_ms"] < b["per_token_ms"]
            for a, b in zip(curves["ours"], curves["token_only"])),
        "crosspoint_with_full_q": any(bundle["sweep"]["crosspoints"].get(k) is not None
                                        for k in ("ours_below_full_q_mbps", "full_q_below_ours_mbps")),
        "ablation_complete": all(bundle.get("ablation", {}).get("groups", {}).get(k)
                                 for k in ("gate_on_off", "sync_vs_pipelined",
                                           "theta_quantiles", "rank_x_sketch")),
        "quality_metrics": None,
    }
    checks["_note"] = ("None = not measurable in this CPU reference run. "
                       "quality_metrics needs a real task harness (§5.1).")
    checks["_sketch_bytes"] = sketch_bytes
    checks["_full_q_bytes"] = full_bytes
    return checks


def cmd_all(args, cfg, theta=None) -> dict:
    out = Path(args.out)
    bundle = {}
    bundle["runtime"] = {"device": "cpu", "batch_size": 1, "warmup_steps": 0,
                         "rtt_configured_ms": cfg.channel.rtt_ms, "rtt_measured_ms": None,
                         "network": "serialized link model", "cuda_graph": False,
                         "run_config": asdict(cfg)}
    bundle["selftest"] = cmd_selftest(args, cfg)
    _write(out / "selftest.json", bundle["selftest"])
    bundle["calibration"] = cmd_calibrate(args, cfg)
    _write(out / "calibration.json", bundle["calibration"])
    if theta is None:
        theta = bundle["calibration"]["operating_theta"]
    theta = (bundle["calibration"]["quantiles"][f"q{round(cfg.gate.quantile * 100)}"]
             if theta is None else theta)
    bundle["sweep"] = cmd_sweep(args, cfg, theta=theta)
    _write(out / "sweep.json", bundle["sweep"])
    bundle["ablation"] = cmd_ablation(args, cfg, theta=theta)
    _write(out / "ablation.json", bundle["ablation"])
    bundle["checks"] = spec_checks(bundle, cfg)
    _write(out / "summary.json", bundle)
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["selftest", "run", "calibrate", "sweep",
                                            "ablation", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument("--mode", default="ours")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--bandwidths", nargs="*", type=float, default=[1, 5, 10, 50, 100])
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.steps is not None:
        cfg = replace(cfg, run=replace(cfg.run, max_steps=args.steps))
    handler = {"selftest": cmd_selftest, "run": cmd_run, "calibrate": cmd_calibrate,
               "sweep": cmd_sweep, "ablation": cmd_ablation, "all": cmd_all}[args.command]
    result = handler(args, cfg, theta=args.theta)
    if args.command not in ("all",):
        _write(Path(args.out) / f"{args.command}.json", result)
    print(f"{args.command} complete: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
