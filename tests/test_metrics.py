import unittest

import numpy as np
from scipy import stats

from su import metrics


class MetricsTests(unittest.TestCase):
    def test_auc_known_cases(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(metrics.auc(scores, np.array([0, 0, 1, 1])), 1.0)
        self.assertAlmostEqual(metrics.auc(scores, np.array([1, 1, 0, 0])), 0.0)
        self.assertAlmostEqual(metrics.auc(scores, np.array([0, 1, 0, 1])), 0.75)
        self.assertAlmostEqual(metrics.auc(scores, np.array([1, 0, 1, 0])), 0.25)
        self.assertTrue(np.isnan(metrics.auc(scores, np.zeros(4))))

    def test_auc_matches_sklearn_on_random_data(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        for _ in range(20):
            s = rng.random(200)
            y = rng.random(200) < (s * 0.7 + 0.15)
            if y.all() or (~y).all():
                continue
            self.assertAlmostEqual(metrics.auc(s, y), roc_auc_score(y, s), places=9)

    def test_spearman_matches_scipy(self):
        rng = np.random.default_rng(1)
        x = rng.random(300)
        y = x * 0.8 + rng.random(300) * 0.2
        self.assertAlmostEqual(metrics.spearman(x, y), stats.spearmanr(x, y).statistic, places=9)

    def test_kl_tv(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        self.assertAlmostEqual(metrics.kl(p, q), 0.0, places=12)
        self.assertAlmostEqual(metrics.tv(p, np.array([1.0, 0.0])), 0.5)
        self.assertEqual(metrics.kl(np.array([0.5, 0.5]), np.array([1.0, 0.0])), float("inf"))

    def test_crosspoint(self):
        xs = [1, 5, 10, 50]
        a = [10, 8, 5, 4]
        b = [4, 5, 9, 20]
        cross = metrics.crosspoint(xs, a, b)
        self.assertIsNotNone(cross)
        self.assertTrue(5 < cross < 10)
        self.assertIsNone(metrics.crosspoint(xs, [1, 2, 3, 4], [5, 6, 7, 8]))

    def test_percentile_summary(self):
        out = metrics.percentile_summary([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(out["mean"], 2.5)
        self.assertEqual(out["n"], 4)
        self.assertEqual(metrics.percentile_summary([]), {})


if __name__ == "__main__":
    unittest.main()
