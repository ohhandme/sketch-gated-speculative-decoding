import unittest

from su.channel import StepTiming, bytes_per_ms, downlink_ms
from su.config import ChannelConfig


class ChannelTests(unittest.TestCase):
    def test_bytes_per_ms(self):
        self.assertAlmostEqual(bytes_per_ms(8.0), 1000.0)  # 8 Mbps -> 1 kB/ms
        with self.assertRaises(ValueError):
            bytes_per_ms(0)

    def test_pipelined_saves_one_rtt(self):
        cfg = ChannelConfig(bandwidth_mbps=10.0, rtt_ms=30.0)
        timing = StepTiming(payload_small=2000, payload_full=400_000, fell_back=True,
                            pipeline=False, rtt_ms=cfg.rtt_ms, overhead=64)
        sync = timing.upload_ms(cfg.bandwidth_mbps)
        timing_pipe = StepTiming(payload_small=2000, payload_full=400_000, fell_back=True,
                                 pipeline=True, rtt_ms=cfg.rtt_ms, overhead=64)
        pipe = timing_pipe.upload_ms(cfg.bandwidth_mbps)
        self.assertLess(pipe["wait_fallback"], sync["wait_fallback"])
        self.assertAlmostEqual(sync["wait_fallback"] - pipe["wait_fallback"], cfg.rtt_ms, delta=1e-6)

    def test_no_fallback_has_no_wait(self):
        cfg = ChannelConfig(bandwidth_mbps=5.0, rtt_ms=50.0)
        for pipeline in (True, False):
            t = StepTiming(payload_small=1500, payload_full=0, fell_back=False,
                           pipeline=pipeline, rtt_ms=cfg.rtt_ms, overhead=64)
            self.assertEqual(t.upload_ms(cfg.bandwidth_mbps)["wait_fallback"], 0.0)

    def test_downlink_scales_with_rtt(self):
        cfg = ChannelConfig(rtt_ms=40.0)
        self.assertGreater(downlink_ms(cfg), 20.0)


if __name__ == "__main__":
    unittest.main()
