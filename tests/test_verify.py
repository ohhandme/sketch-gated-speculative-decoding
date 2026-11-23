import unittest

import numpy as np

from su.verify import (accept_probability, residual, sample, support_preserved,
                       token_only_token)


class VerifyTests(unittest.TestCase):
    def test_accept_probability_edges(self):
        self.assertEqual(accept_probability(0.0, 0.2), 0.0)
        self.assertEqual(accept_probability(0.2, 0.0), 1.0)
        self.assertEqual(accept_probability(0.3, 0.6), 0.5)
        self.assertEqual(accept_probability(0.9, 0.2), 1.0)

    def test_residual_is_a_distribution(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        r = residual(p, q)
        self.assertAlmostEqual(float(r.sum()), 1.0)
        self.assertTrue((r >= 0).all())
        np.testing.assert_allclose(r, np.array([0.4, 0.0, 0.0]) / 0.4)

    def test_residual_falls_back_to_p(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.0, 0.0])
        np.testing.assert_allclose(residual(p, q), p)

    def test_sampling_hits_support(self):
        rng = np.random.default_rng(0)
        p = np.array([0.0, 0.5, 0.0, 0.5])
        draws = {sample(p, rng) for _ in range(200)}
        self.assertEqual(draws, {1, 3})

    def test_token_only_rule_and_support(self):
        p = np.array([0.1, 0.6, 0.3, 0.0])
        self.assertEqual(token_only_token(p), 1)
        self.assertTrue(support_preserved(p, [1]))
        self.assertFalse(support_preserved(p, [3]))


if __name__ == "__main__":
    unittest.main()
