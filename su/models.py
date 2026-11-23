"""Distribution sources: tiny random Qwen3 for CI, HF checkpoints, synthetic oracle."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


class DistributionSource:
    vocab: int

    def log_probs(self, prefix: list[int]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class _TorchModel(DistributionSource):
    def __init__(self, model, vocab: int):
        self.model = model
        self.vocab = vocab

    def log_probs(self, prefix: list[int]) -> np.ndarray:
        import torch

        ids = torch.tensor([list(prefix)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        return torch.log_softmax(logits.double(), dim=-1).numpy()


def build_tiny_qwen(vocab: int, hidden: int, layers: int, heads: int, kv_heads: int,
                    intermediate: int, seed: int) -> _TorchModel:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    return _TorchModel(model, vocab)


def build_hf(path: str) -> _TorchModel:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    return _TorchModel(model, int(model.config.vocab_size))


@dataclass
class SyntheticOracle(DistributionSource):
    """Deterministic pseudo-distribution keyed by (seed, prefix); no torch required."""

    vocab: int
    seed: int
    concentration: float = 0.3

    def log_probs(self, prefix: list[int]) -> np.ndarray:
        digest = hashlib.blake2b(("%d:%s" % (self.seed, ",".join(map(str, prefix)))).encode(),
                                 digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        alpha = np.full(self.vocab, self.concentration)
        # A few hot ids keep the head/tail split meaningful.
        hot = rng.choice(self.vocab, size=max(1, self.vocab // 32), replace=False)
        alpha[hot] += 2.0
        p = rng.dirichlet(alpha)
        with np.errstate(divide="ignore"):
            return np.log(p)


def build_pair(cfg: ModelConfig, seed: int) -> tuple[DistributionSource, DistributionSource]:
    def make(spec: str, hidden: int, layers: int, sub_seed: int) -> DistributionSource:
        if spec == "tiny":
            return build_tiny_qwen(cfg.vocab_size, hidden, layers, cfg.heads, cfg.kv_heads,
                                   cfg.intermediate, sub_seed)
        if spec == "synthetic":
            return SyntheticOracle(vocab=cfg.vocab_size, seed=sub_seed)
        return build_hf(spec)

    draft = make(cfg.draft, cfg.draft_hidden, cfg.draft_layers, seed)
    target = make(cfg.target, cfg.target_hidden, cfg.target_layers, seed + 7919)
    return draft, target


class Timer:
    """Context manager that appends elapsed milliseconds to a dict of accumulators."""

    def __init__(self, sink: dict, key: str):
        self.sink = sink
        self.key = key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink[self.key] = self.sink.get(self.key, 0.0) + (time.perf_counter() - self.t0) * 1e3
        return False
