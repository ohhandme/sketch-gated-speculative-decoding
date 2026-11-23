import unittest

from dataclasses import replace

from su.config import HeadConfig, LowRankConfig, ModelConfig, RunConfig, SketchConfig, SystemConfig
from su.models import build_pair
from su.tables import ablation, main_table, run_mode


def small_cfg():
    return SystemConfig(
        model=ModelConfig(draft="synthetic", target="synthetic", vocab_size=64, prompt_len=4),
        sketch=SketchConfig(dim=16, nnz=4, seed=7),
        lowrank=LowRankConfig(rank=3, delta_threshold=0.0),
        head=HeadConfig(tokens=8),
        run=RunConfig(draft_len=3, seed=3, max_steps=3, mode="ours"),
    )


class TableTests(unittest.TestCase):
    def test_main_table_has_uplink_ratios(self):
        cfg = replace(small_cfg(), model=ModelConfig(draft="synthetic", target="synthetic",
                                                     vocab_size=1024, prompt_len=4))
        draft, target = build_pair(cfg.model, cfg.run.seed)
        rows = {}
        for mode in ("cloud_only", "full_q", "token_only", "ours"):
            rows[mode] = run_mode(cfg, draft, target, [1, 2, 3, 4], mode)
        table = {r["mode"]: r for r in main_table(rows)}
        self.assertLess(table["token_only"]["bytes_per_step"],
                        table["full_q"]["bytes_per_step"] / 50.0)
        self.assertLess(table["ours"]["bytes_per_step"],
                        table["full_q"]["bytes_per_step"] / 2.0)
        self.assertAlmostEqual(table["ours"]["uplink_ratio_vs_full_q"],
                               table["ours"]["bytes_per_token"]
                               / table["full_q"]["bytes_per_token"], places=9)

    def test_all_four_ablations_have_rows(self):
        cfg = replace(small_cfg(), lowrank=replace(small_cfg().lowrank, update_every=1))
        draft, target = build_pair(cfg.model, cfg.run.seed)
        out = ablation(cfg, draft, target, [1, 2, 3, 4],
                       thetas={"q90": 0.003, "q95": 0.004, "q99": 0.005},
                       ranks=(2, 4), sketch_dims=(8, 16))
        self.assertEqual(len(out["gate_on_off"]), 2)
        self.assertEqual(len(out["sync_vs_pipelined"]), 2)
        self.assertEqual(len(out["theta_quantiles"]), 3)
        self.assertEqual(len(out["rank_x_sketch"]), 4)
        self.assertLess(out["sync_vs_pipelined"][1]["per_token_ms"],
                        out["sync_vs_pipelined"][0]["per_token_ms"])


if __name__ == "__main__":
    unittest.main()
