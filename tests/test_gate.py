import unittest

import numpy as np

from su.config import GateConfig
from su.gate import AdaptiveGate, build_gate, fallback_rate, quantile_theta


class GateTests(unittest.TestCase):
    def test_quantile_matches_numpy(self):
        s = np.linspace(0, 1, 101)
        for q in (0.5, 0.9, 0.95, 0.99):
            self.assertAlmostEqual(quantile_theta(s, q), float(np.quantile(s, q)), places=12)
        with self.assertRaises(ValueError):
            quantile_theta(np.array([]), 0.9)
        with self.assertRaises(ValueError):
            quantile_theta(s, 1.2)

    def test_fallback_rate_is_monotone(self):
        s = np.random.default_rng(0).random(1000)
        thetas = np.quantile(s, [0.5, 0.9, 0.95, 0.99])
        rates = [fallback_rate(s, t) for t in thetas]
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertAlmostEqual(rates[0], 0.5, places=2)
        self.assertAlmostEqual(rates[-1], 0.01, places=2)

    def test_adaptive_gate_moves_theta(self):
        gate = AdaptiveGate(theta=0.1, target_fallback=0.1, step=0.01, window=4)
        for _ in range(4):
            gate.observe(True)  # 100% fallback -> raise theta
        self.assertGreater(gate.theta, 0.1)
        before = gate.theta
        for _ in range(4):
            gate.observe(False)  # 0% fallback -> lower theta
        self.assertLess(gate.theta, before)

    def test_bounds_are_enforced(self):
        cfg = GateConfig(theta=0.5, theta_min=0.4, theta_max=0.6, adapt_step=0.2,
                         target_fallback=0.0, ema_window=2)
        gate = build_gate(cfg)
        for _ in range(12):
            gate.observe(True)
        self.assertLessEqual(gate.theta, 0.6)


if __name__ == "__main__":
    unittest.main()
