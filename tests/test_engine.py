import unittest
from dataclasses import replace

import numpy as np

from su.config import ModelConfig, RunConfig, SketchConfig, SystemConfig, HeadConfig, LowRankConfig
from su.channel import bytes_per_ms
from su.engine import Engine
from su.models import SyntheticOracle, build_pair
from su.serialize import HEADER


def small_cfg(**run_overrides):
    cfg = SystemConfig(
        model=ModelConfig(draft="synthetic", target="synthetic", vocab_size=64, prompt_len=4),
        sketch=SketchConfig(dim=16, nnz=4, seed=11),
        lowrank=LowRankConfig(rank=3, delta_threshold=0.0),
        head=HeadConfig(tokens=8),
        run=RunConfig(draft_len=4, seed=5, max_steps=3, mode="ours", fallback="B",
                      pipeline=True),
    )
    if run_overrides:
        cfg = replace(cfg, run=replace(cfg.run, **run_overrides))
    return cfg


class EngineTests(unittest.TestCase):
    def _sources(self, cfg):
        return build_pair(cfg.model, cfg.run.seed)

    def test_all_modes_run_and_emit_tokens(self):
        for mode in ("cloud_only", "token_only", "full_q", "ours"):
            cfg = small_cfg(mode=mode)
            draft, target = self._sources(cfg)
            result = Engine(cfg, draft, target, [1, 2, 3, 4]).run()
            self.assertEqual(len(result.steps), cfg.run.max_steps)
            self.assertGreater(len(result.tokens), 0)
            for step in result.steps:
                self.assertGreaterEqual(step.emitted, 1)
                self.assertEqual(len(step.tokens), step.emitted)
                self.assertGreaterEqual(step.accepted, 0)

    def test_deterministic_with_fixed_seed(self):
        cfg = small_cfg()
        draft, target = self._sources(cfg)
        a = Engine(cfg, draft, target, [1, 2, 3, 4]).run()
        draft2, target2 = build_pair(cfg.model, cfg.run.seed)
        b = Engine(cfg, draft2, target2, [1, 2, 3, 4]).run()
        self.assertEqual(a.tokens, b.tokens)
        self.assertEqual([s.gate_score for s in a.steps], [s.gate_score for s in b.steps])

    def test_byte_accounting(self):
        cfg = small_cfg(mode="ours", fallback="B")
        draft, target = self._sources(cfg)
        result = Engine(cfg, draft, target, [0, 1, 2]).run()
        for step in result.steps:
            self.assertEqual(step.bytes_full, 0)
            expected = (cfg.run.draft_len * cfg.lowrank.rank * 2 + cfg.run.draft_len * cfg.sketch.dim * 2 +
                        cfg.run.draft_len * (cfg.head.tokens * 12 + 4) + HEADER.size)
            self.assertGreaterEqual(step.bytes_small, expected)
            self.assertEqual(step.bytes_total, step.bytes_small + cfg.channel.protocol_overhead_bytes)

    def test_full_q_payload_size(self):
        cfg = small_cfg(mode="full_q")
        draft, target = self._sources(cfg)
        result = Engine(cfg, draft, target, [0, 1]).run()
        for step in result.steps:
            self.assertEqual(step.bytes_full, cfg.run.draft_len * cfg.model.vocab_size * 2)

    def test_token_only_always_advances(self):
        cfg = small_cfg(mode="token_only")
        draft, target = self._sources(cfg)
        result = Engine(cfg, draft, target, [3, 1, 4]).run()
        for step in result.steps:
            self.assertGreaterEqual(step.emitted, 1)
            self.assertEqual(step.fell_back, False)

    def test_pipelined_beats_sync_on_fallback(self):
        cfg = small_cfg(mode="ours", fallback="A", pipeline=False)
        draft, target = self._sources(cfg)
        sync = Engine(cfg, draft, target, [0, 1]).run()
        cfg2 = replace(cfg, run=replace(cfg.run, pipeline=True))
        draft2, target2 = self._sources(cfg2)
        pipe = Engine(cfg2, draft2, target2, [0, 1]).run()
        self.assertLessEqual(sum(s.e2e_ms for s in pipe.steps), sum(s.e2e_ms for s in sync.steps))

    def test_support_preservation_with_zero_target_mass(self):
        class Sparse(SyntheticOracle):
            def log_probs(self, prefix):
                lp = super().log_probs(prefix)
                p = np.exp(lp)
                p[::3] = 0.0
                p = p / p.sum()
                with np.errstate(divide="ignore"):
                    return np.log(p)

        cfg = small_cfg()
        draft = SyntheticOracle(vocab=64, seed=3)
        target = Sparse(vocab=64, seed=9)
        result = Engine(cfg, draft, target, [1, 2, 3]).run()
        prefix = list(result.prompt)
        for step in result.steps:
            for token in step.tokens:
                p = np.exp(target.log_probs(prefix))
                self.assertGreater(p[token], 0.0)
                prefix.append(token)

    def test_timings_present(self):
        cfg = small_cfg()
        draft, target = self._sources(cfg)
        step = Engine(cfg, draft, target, [0, 0, 0]).run().steps[0]
        for key in ("t_draft", "t_sketch", "t_serialize", "t_upload", "t_prepare_cloud",
                    "t_gate", "t_verify", "t_resample", "t_accept", "t_downlink",
                    "t_propagation", "t_fallback_upload"):
            self.assertIn(key, step.timings)
        self.assertAlmostEqual(step.e2e_ms, sum(step.timings.values()), places=9)
        self.assertAlmostEqual(step.per_token_ms, step.e2e_ms / step.emitted, places=9)

    def test_token_only_payload_is_only_ids(self):
        cfg = small_cfg(mode="token_only")
        draft, target = self._sources(cfg)
        step = Engine(cfg, draft, target, [0, 1, 2]).run().steps[0]
        self.assertEqual(step.bytes_full, 0)
        self.assertEqual(step.bytes_small,
                         HEADER.size + cfg.run.draft_len * 4)
        self.assertLess(step.bytes_total, 200)
        self.assertEqual(step.gate_score, 0.0)

    def test_full_q_latency_tracks_bandwidth(self):
        cfg = replace(small_cfg(mode="full_q"),
                      model=ModelConfig(draft="synthetic", target="synthetic",
                                        vocab_size=1024, prompt_len=4))
        fast = replace(cfg, channel=replace(cfg.channel, bandwidth_mbps=100.0))
        slow = replace(cfg, channel=replace(cfg.channel, bandwidth_mbps=1.0))
        draft, target = self._sources(cfg)
        a = Engine(fast, draft, target, [0, 1]).run()
        b = Engine(slow, draft, target, [0, 1]).run()
        self.assertGreater(b.summary()["per_token_ms"]["mean"],
                           a.summary()["per_token_ms"]["mean"] + 40.0)
        for step in (a.steps[0], b.steps[0]):
            self.assertEqual(step.bytes_total,
                             cfg.run.draft_len * (1024 * 2 + 4)
                             + HEADER.size
                             + cfg.channel.protocol_overhead_bytes)

    def test_sync_fallback_wait_is_not_double_counted(self):
        cfg = small_cfg(mode="ours", fallback="A", pipeline=False)
        draft, target = self._sources(cfg)
        engine = Engine(cfg, draft, target, [0, 1, 2], theta=0.0)
        step = engine.run().steps[0]
        self.assertTrue(step.fell_back)
        self.assertEqual(step.fallback_kind, "A")
        # t_upload is the link time of the small payload only; the RTT and the
        # full-Q transfer live in t_fallback_upload and must not be repeated.
        self.assertLess(step.timings["t_upload"], cfg.channel.rtt_ms)
        self.assertGreaterEqual(
            step.timings["t_fallback_upload"],
            cfg.channel.rtt_ms + step.bytes_full / bytes_per_ms(cfg.channel.bandwidth_mbps)
            - 1e-6)

    def test_forced_fallback_pipeline_beats_sync(self):
        cfg = small_cfg(mode="ours", fallback="A", pipeline=True)
        draft, target = self._sources(cfg)
        pipe = Engine(cfg, draft, target, [0, 1, 2], theta=0.0).run()
        sync_cfg = replace(cfg, run=replace(cfg.run, pipeline=False))
        sync = Engine(sync_cfg, draft, target, [0, 1, 2], theta=0.0).run()
        self.assertLess(pipe.summary()["per_token_ms"]["mean"],
                        sync.summary()["per_token_ms"]["mean"] - cfg.channel.rtt_ms * 0.5)

    def test_optimistic_full_q_charges_bytes_and_carries_link_time(self):
        cfg = replace(small_cfg(mode="ours", fallback="A", pipeline=True),
                      model=ModelConfig(draft="synthetic", target="synthetic",
                                        vocab_size=1024, prompt_len=4),
                      channel=replace(small_cfg().channel, bandwidth_mbps=1.0))
        draft, target = self._sources(cfg)
        result = Engine(cfg, draft, target, [0, 1, 2], theta=float("inf")).run()
        self.assertEqual(result.summary()["fallback_rate"], 0.0)
        for step in result.steps:
            self.assertGreaterEqual(step.bytes_total,
                                    cfg.run.draft_len * 1024 * 2)
        first, second = result.steps[:2]
        self.assertGreater(first.link_carry_ms, 0.0)
        prep = sum(second.timings[k] for k in ("t_draft", "t_sketch", "t_prepare_edge", "t_serialize"))
        expected = max(0.0, first.link_carry_ms - prep) + (second.bytes_small + cfg.channel.protocol_overhead_bytes) / bytes_per_ms(cfg.channel.bandwidth_mbps)
        self.assertAlmostEqual(second.timings["t_upload"], expected)

    def test_edge_and_cloud_basis_stay_identical(self):
        cfg = small_cfg(mode="ours", fallback="B")
        cfg = replace(cfg, lowrank=replace(cfg.lowrank, update_every=1, delta_threshold=0.0,
                                           max_delta_entries=None))
        draft, target = self._sources(cfg)
        engine = Engine(cfg, draft, target, [0, 1, 2])
        engine.run()
        np.testing.assert_array_equal(engine.basis, engine.cloud_basis)


if __name__ == "__main__":
    unittest.main()
