import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from su.engine import Engine, tail_features
from su.serialize import pack, unpack
from test_engine import small_cfg
from su.models import build_pair


class ProtocolRegressionTests(unittest.TestCase):
    def run_mode(self, cfg, theta=None):
        return Engine(cfg, *build_pair(cfg.model, cfg.run.seed), [1, 2, 3], theta=theta).run()

    def test_b2_matches_target_sample_stream_including_bonus(self):
        cfg = small_cfg(mode="token_only", max_steps=32)
        b2 = self.run_mode(cfg)
        b0 = self.run_mode(replace(cfg, run=replace(cfg.run, mode="cloud_only", max_steps=len(b2.tokens))))
        self.assertEqual(b0.tokens, b2.tokens)

    def test_full_fallback_matches_full_baseline_for_both_transports(self):
        full = self.run_mode(small_cfg(mode="full_q"))
        for pipeline in (False, True):
            trial = self.run_mode(small_cfg(fallback="A", pipeline=pipeline), theta=-1)
            self.assertTrue(all(s.fell_back for s in trial.steps))
            self.assertEqual(full.tokens, trial.tokens)
            self.assertEqual(full.accepted, trial.accepted)

    def test_full_block_sketch_and_candidate_ids_survive_wire(self):
        cfg = small_cfg(mode="ours", max_steps=1)
        cfg = replace(cfg, gate=replace(cfg.gate, score="cosine"))
        packets = []
        def capture(payload):
            blob = pack(payload)
            packets.append(unpack(blob))
            return blob
        with patch('su.engine.pack', side_effect=capture):
            result = self.run_mode(cfg, theta=2.1)
        packet, = packets
        self.assertEqual(packet.sketch.size, cfg.run.draft_len * cfg.sketch.dim)
        self.assertEqual(packet.token_ids.size, cfg.run.draft_len)
        self.assertEqual(packet.head_prob.dtype, np.float64)
        self.assertGreater(result.steps[0].gate_score, 0.1)
        self.assertIsNone(packet.full_q)

    def test_head_mask_excludes_head_from_features(self):
        features = tail_features(np.array([[.8, .1, .1], [.4, .3, .3]]), np.array([[0], [0]]))
        np.testing.assert_allclose(features, 0, atol=1e-12)
