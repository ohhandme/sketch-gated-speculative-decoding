import unittest

import numpy as np

from su.headtail import head_indices, merge


class HeadTailTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.q = rng.dirichlet(np.ones(40) * 0.4)
        self.logits = rng.standard_normal(40)

    def test_head_selection_order(self):
        idx = head_indices(self.q, 5)
        self.assertEqual(idx.size, 5)
        self.assertTrue(np.all(np.diff(self.q[idx]) <= 0))
        self.assertEqual(set(idx.tolist()), set(np.argsort(-self.q)[:5].tolist()))

    def test_head_masses_preserved_and_normalised(self):
        idx = head_indices(self.q, 6)
        q_tilde = merge(idx, self.q[idx], self.logits)
        self.assertAlmostEqual(float(q_tilde.sum()), 1.0, places=12)
        np.testing.assert_allclose(q_tilde[idx], self.q[idx] / q_tilde.sum() * 1.0, atol=1e-12)
        self.assertTrue((q_tilde >= 0).all())

    def test_tail_proportional_to_low_rank_logits(self):
        idx = head_indices(self.q, 3)
        q_tilde = merge(idx, self.q[idx], self.logits)
        mask = np.ones(40, dtype=bool)
        mask[idx] = False
        ratio = q_tilde[mask] / np.exp(self.logits[mask] - self.logits[mask].max())
        self.assertLess(float(np.ptp(ratio)), 1e-9)

    def test_extremes(self):
        zero = merge(np.empty(0, dtype=int), np.empty(0), self.logits)
        self.assertAlmostEqual(float(zero.sum()), 1.0)
        everything = np.arange(40)
        full = merge(everything, self.q, self.logits)
        np.testing.assert_allclose(full, self.q / self.q.sum(), atol=1e-12)

    def test_head_ties_break_by_lowest_id(self):
        q = np.zeros(10)
        q[[3, 7, 9]] = 0.3
        self.assertEqual(head_indices(q, 2).tolist(), [3, 7])

    def test_head_does_not_promote_equal_but_unselected_candidate(self):
        q = np.zeros(10)
        q[[1, 4]] = 0.5
        idx = head_indices(q, 1)
        self.assertEqual(idx.tolist(), [1])
        self.assertNotIn(4, idx.tolist())


if __name__ == "__main__":
    unittest.main()
