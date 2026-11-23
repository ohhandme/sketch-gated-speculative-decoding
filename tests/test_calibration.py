import unittest
from dataclasses import replace

import numpy as np

from su import calibration
from su.config import HeadConfig, LowRankConfig, ModelConfig, RunConfig, SketchConfig, SystemConfig
from su.models import build_pair


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SystemConfig(
            model=ModelConfig(draft="synthetic", target="synthetic", vocab_size=48, prompt_len=3),
            sketch=SketchConfig(dim=12, nnz=3, seed=2),
            lowrank=LowRankConfig(rank=2),
            head=HeadConfig(tokens=6),
            run=RunConfig(draft_len=3, seed=4, max_steps=3),
        )

    def test_collect_shape_and_ranges(self):
        draft, target = build_pair(self.cfg.model, self.cfg.run.seed)
        data = calibration.collect(self.cfg, draft, target, [1, 2, 3], steps=3)
        self.assertEqual(len(data), 3 * 3)
        self.assertTrue((data.scores >= 0).all())
        self.assertTrue(set(np.unique(data.flips)).issubset({0, 1}))
        self.assertTrue((data.tv >= 0).all() and (data.tv <= 1).all())

    def test_quantiles_and_sweep(self):
        rng = np.random.default_rng(0)
        data = calibration.CalibrationData(scores=rng.random(500),
                                           flips=(rng.random(500) < 0.2).astype(int),
                                           accept_gap=rng.random(500),
                                           tv=rng.random(500))
        qs = calibration.quantiles(data)
        self.assertAlmostEqual(qs["q95"], float(np.quantile(data.scores, 0.95)), places=12)
        rows = calibration.sweep_table(data, sorted(qs.values()))
        rates = [r["fallback_rate"] for r in rows]
        self.assertEqual(rates, sorted(rates, reverse=True))


if __name__ == "__main__":
    unittest.main()
