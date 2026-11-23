import unittest

import numpy as np

from su.sketch import build_sketch, cosine_score, l1_score, l1_tv_bound


class SketchTests(unittest.TestCase):
    def setUp(self):
        self.vocab = 97
        self.sketch = build_sketch(self.vocab, 16, 5, "sparse_pm1", 7)

    def test_deterministic_and_shaped(self):
        other = build_sketch(self.vocab, 16, 5, "sparse_pm1", 7)
        np.testing.assert_array_equal(self.sketch.columns, other.columns)
        np.testing.assert_array_equal(self.sketch.values, other.values)
        self.assertEqual(self.sketch.columns.shape, (16, 5))
        for row in self.sketch.columns:
            self.assertEqual(len(set(row.tolist())), 5)
            self.assertTrue(np.all(np.diff(row) >= 1))

    def test_apply_matches_dense_matrix(self):
        dense = np.zeros((16, self.vocab), dtype=np.float32)
        rows = np.repeat(np.arange(16), 5)
        dense[rows, self.sketch.columns.ravel()] = self.sketch.values.ravel()
        rng = np.random.default_rng(0)
        q = rng.random((3, self.vocab))
        q /= q.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(self.sketch.apply(q), q @ dense.T, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(self.sketch.apply(q[0]), (q @ dense.T)[0], rtol=1e-6, atol=1e-6)

    def test_l1_score_respects_tv_bound(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            p = rng.dirichlet(np.ones(self.vocab))
            q = rng.dirichlet(np.ones(self.vocab) * 0.5)
            a = l1_score(self.sketch.apply(p), self.sketch.apply(q))
            tv = 0.5 * np.abs(p - q).sum()
            self.assertLessEqual(a, l1_tv_bound(5, self.vocab, tv) + 1e-9)
            self.assertGreaterEqual(a, 0.0)

    def test_identical_distributions_score_zero(self):
        rng = np.random.default_rng(2)
        q = rng.dirichlet(np.ones(self.vocab))
        s = self.sketch.apply(q)
        self.assertEqual(float(l1_score(s, s)), 0.0)
        self.assertAlmostEqual(float(cosine_score(s, s)), 0.0, places=12)

    def test_gaussian_variant(self):
        g = build_sketch(self.vocab, 8, 6, "sparse_gauss", 3)
        q = np.full((1, self.vocab), 1 / self.vocab)
        out = g.apply(q)
        self.assertEqual(out.shape, (1, 8))
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
