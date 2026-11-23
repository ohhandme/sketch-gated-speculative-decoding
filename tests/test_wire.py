import unittest

import numpy as np

from su.serialize import Payload, pack, unpack


def sample_payload(full=True):
    rng = np.random.default_rng(0)
    rows, rank, k, kt = 3, 4, 8, 5
    return Payload(
        coefs=rng.standard_normal((rows, rank)).astype(np.float16),
        sketch=rng.standard_normal(k).astype(np.float16),
        head_idx=np.tile(np.arange(kt, dtype=np.int32), (rows, 1)),
        head_prob=np.full((rows, kt), 0.01, dtype=np.float16),
        basis_idx=np.array([1, 5, 9], dtype=np.int32),
        basis_val=np.array([0.1, -0.2, 0.3], dtype=np.float16),
        full_q=(rng.random((rows, 11)).astype(np.float16) if full else None),
    )


class WireTests(unittest.TestCase):
    def test_roundtrip_without_full_q(self):
        p = sample_payload(full=False)
        blob = pack(p)
        back = unpack(blob)
        np.testing.assert_array_equal(back.coefs, p.coefs)
        np.testing.assert_array_equal(back.sketch, p.sketch)
        np.testing.assert_array_equal(back.head_idx, p.head_idx)
        np.testing.assert_array_equal(back.head_prob, p.head_prob)
        np.testing.assert_array_equal(back.basis_idx, p.basis_idx)
        self.assertIsNone(back.full_q)
        self.assertEqual(p.nbytes(), len(blob))

    def test_roundtrip_with_full_q(self):
        p = sample_payload(full=True)
        blob = pack(p)
        back = unpack(blob)
        np.testing.assert_array_equal(back.full_q, p.full_q)
        self.assertEqual(p.nbytes(), len(blob))
        self.assertEqual(p.sizes()["full_q"], p.full_q.size * 2)

    def test_truncated_payload_rejected(self):
        blob = pack(sample_payload(full=True))
        with self.assertRaises(ValueError):
            unpack(blob[:-3])
        with self.assertRaises(ValueError):
            unpack(b"XXXX" + blob[4:])

    def test_byte_accounting_fields(self):
        p = sample_payload(full=False)
        sizes = p.sizes()
        self.assertEqual(sizes["coefs"], p.coefs.size * 2)
        self.assertEqual(sizes["sketch"], p.sketch.size * 2)
        self.assertEqual(sizes["head"], p.head_idx.size * 4 + p.head_prob.size * 8)
        self.assertEqual(sum(sizes.values()), len(pack(p)))


if __name__ == "__main__":
    unittest.main()
