import unittest

import numpy as np

from su.config import ModelConfig
from su.models import SyntheticOracle, build_pair, build_tiny_qwen


class ModelTests(unittest.TestCase):
    def test_synthetic_is_deterministic_and_normalised(self):
        a = SyntheticOracle(vocab=32, seed=1)
        b = SyntheticOracle(vocab=32, seed=1)
        lp1 = a.log_probs([1, 2, 3])
        lp2 = b.log_probs([1, 2, 3])
        np.testing.assert_array_equal(lp1, lp2)
        self.assertAlmostEqual(float(np.exp(lp1).sum()), 1.0, places=9)
        self.assertFalse(np.allclose(lp1, a.log_probs([3, 2, 1])))

    def test_tiny_qwen_forward(self):
        model = build_tiny_qwen(vocab=128, hidden=32, layers=1, heads=2, kv_heads=1,
                                intermediate=64, seed=0)
        lp = model.log_probs([1, 2, 3, 4])
        self.assertEqual(lp.shape, (128,))
        self.assertAlmostEqual(float(np.exp(lp).sum()), 1.0, places=9)
        self.assertTrue(np.isfinite(lp).all())

    def test_pair_uses_different_weights(self):
        cfg = ModelConfig(draft="tiny", target="tiny", vocab_size=64, draft_hidden=32,
                          draft_layers=1, target_hidden=32, target_layers=1, heads=2,
                          kv_heads=1, intermediate=64)
        draft, target = build_pair(cfg, seed=0)
        lp_d = draft.log_probs([1, 2, 3])
        lp_t = target.log_probs([1, 2, 3])
        self.assertEqual(lp_d.shape, lp_t.shape)
        self.assertGreater(float(np.abs(lp_d - lp_t).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
