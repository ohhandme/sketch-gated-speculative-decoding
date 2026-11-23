import unittest

import numpy as np

from su import lowrank


class LowRankTests(unittest.TestCase):
    def test_basis_is_orthonormal(self):
        b = lowrank.init_basis(64, 4, 1)
        np.testing.assert_allclose(b.T @ b, np.eye(4), atol=1e-5)

    def test_exact_when_rank_matches_intrinsic_dimension(self):
        rng = np.random.default_rng(0)
        vocab, rank, rows = 50, 2, 5
        factors = rng.standard_normal((rank, vocab))
        x = rng.standard_normal((rows, rank)) @ factors
        logq = np.log(np.exp(x) / np.exp(x).sum(axis=1, keepdims=True))
        # rank - 1 shape columns plus the constant column span a rank-2 signal.
        basis = lowrank.init_basis(vocab, rank + 1, 1)
        for _ in range(12):
            basis = lowrank.power_update(basis, logq, iters=1, damping=0.2)
        rebuilt = lowrank.reconstruct(lowrank.project(logq, basis), basis)
        np.testing.assert_allclose(lowrank.centre(rebuilt), lowrank.centre(logq), atol=1e-3)
        self.assertGreater(lowrank.explained_energy(logq, basis), 0.999)

    def test_power_update_increases_energy(self):
        rng = np.random.default_rng(3)
        vocab, rank, rows = 80, 3, 6
        x = rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, vocab))
        logq = x - x.max(axis=1, keepdims=True)
        basis = lowrank.init_basis(vocab, rank, 4)
        before = lowrank.explained_energy(logq, basis)
        for _ in range(6):
            basis = lowrank.power_update(basis, logq, iters=1, damping=0.2)
        after = lowrank.explained_energy(logq, basis)
        self.assertGreater(after, before)
        self.assertLessEqual(after, 1.0 + 1e-6)

    def test_delta_roundtrip(self):
        rng = np.random.default_rng(5)
        old = lowrank.init_basis(30, 3, 6)
        new = lowrank.power_update(old, rng.standard_normal((4, 30)), iters=1, damping=0.5)
        idx, val = lowrank.basis_delta(old, new, 0.0)
        rebuilt = lowrank.apply_delta(old, idx, val)
        np.testing.assert_allclose(rebuilt, new, atol=1e-6)
        idx2, val2 = lowrank.basis_delta(old, new, 1e9)
        self.assertEqual(idx2.size, 0)
        np.testing.assert_allclose(lowrank.apply_delta(old, idx2, val2), old, atol=1e-7)

    def test_float16_coefficients(self):
        rng = np.random.default_rng(7)
        c = rng.standard_normal((3, 4)) * 5
        q = lowrank.quantise_coefs(c)
        self.assertEqual(q.dtype, np.float16)
        self.assertLess(np.abs(q.astype(np.float32) - c).max(), 0.01 * np.abs(c).max() + 1e-3)


if __name__ == "__main__":
    unittest.main()
