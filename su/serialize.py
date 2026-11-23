"""Byte-accurate payload for the uplink (spec §2.1 / §2.3).

The three baselines put only their own bytes on the wire:

* ``KIND_OURS``   - coefs + sketch + exact head + optional basis delta.
* ``KIND_TOKENS`` - the N draft token ids only (B2, "N x token").
* ``KIND_FULL_Q`` - the N x V logits only (B1).

The header carries the token count so the token-only payload needs no sketch,
coefficients or head table.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"SUP2"
HEADER = struct.Struct(">4sBBHHIHHII")
FLAG_FULL_Q = 1
FLAG_TOKENS = 2

KIND_OURS = 0
KIND_TOKENS = 1
KIND_FULL_Q = 2


@dataclass
class Payload:
    """Everything the edge sends for one proposal block."""

    coefs: np.ndarray  # (N, r) fp16
    sketch: np.ndarray  # (k,) fp16
    head_idx: np.ndarray  # (N, kt) int32
    head_prob: np.ndarray  # (N, kt) fp16
    basis_idx: np.ndarray  # (m,) int32
    basis_val: np.ndarray  # (m,) fp16
    full_q: np.ndarray | None = None  # (N, V) fp16
    method: int = 3
    token_ids: np.ndarray | None = None  # (N,) int32, B2 only
    kind: int = KIND_OURS

    def sizes(self) -> dict[str, int]:
        rows, rank = self.coefs.shape
        head = self.head_idx.size * 4 + self.head_prob.size * 8
        basis = self.basis_idx.size * 4 + self.basis_val.size * 2
        tokens = 0 if self.token_ids is None else self.token_ids.size * 4
        full = 0 if self.full_q is None else self.full_q.size * 2
        return {
            "coefs": rows * rank * 2,
            "sketch": self.sketch.size * 2,
            "head": head,
            "tokens": tokens,
            "basis": basis,
            "full_q": full,
            "header": HEADER.size,
        }

    def small_bytes(self) -> int:
        s = self.sizes()
        return (s["coefs"] + s["sketch"] + s["head"] + s["tokens"] + s["basis"] +
                s["header"])

    def nbytes(self) -> int:
        return sum(self.sizes().values())


def pack(payload: Payload) -> bytes:
    coefs = np.asarray(payload.coefs, dtype=np.float16)
    sketch = np.asarray(payload.sketch, dtype=np.float16)
    head_idx = np.asarray(payload.head_idx, dtype=np.int32)
    head_prob = np.asarray(payload.head_prob, dtype=np.float64)
    basis_idx = np.asarray(payload.basis_idx, dtype=np.int32)
    basis_val = np.asarray(payload.basis_val, dtype=np.float16)
    token_ids = (np.empty(0, dtype=np.int32) if payload.token_ids is None
                 else np.asarray(payload.token_ids, dtype=np.int32))
    if coefs.ndim != 2:
        raise ValueError("coefs must be (N, r)")
    rows, rank = coefs.shape
    if token_ids.size and rows and token_ids.size != rows:
        raise ValueError("token count differs from block length")
    head_tokens = head_idx.size // rows if rows else 0
    flags = FLAG_FULL_Q if payload.full_q is not None else 0
    if token_ids.size:
        flags |= FLAG_TOKENS
    header = HEADER.pack(MAGIC, payload.method, flags, rows, rank, sketch.size,
                         head_tokens, token_ids.size, basis_idx.size,
                         0 if payload.full_q is None else payload.full_q.shape[1])
    chunks = [header, coefs.tobytes(), sketch.tobytes(), head_idx.tobytes(),
              head_prob.tobytes(), token_ids.tobytes(), basis_idx.tobytes(),
              basis_val.tobytes()]
    if payload.full_q is not None:
        chunks.append(np.asarray(payload.full_q, dtype=np.float16).tobytes())
    return b"".join(chunks)


def unpack(data: bytes) -> Payload:
    if len(data) < HEADER.size:
        raise ValueError("truncated payload")
    (magic, method, flags, rows, rank, sketch_dim, head_tokens, token_count,
     basis_n, vocab) = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("bad payload magic")
    offset = HEADER.size

    def take(count: int, dtype) -> np.ndarray:
        nonlocal offset
        itemsize = np.dtype(dtype).itemsize
        need = count * itemsize
        if offset + need > len(data):
            raise ValueError("truncated payload body")
        out = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += need
        return out

    coefs = take(rows * rank, np.float16).reshape(rows, rank)
    sketch = take(sketch_dim, np.float16)
    head_idx = take(rows * head_tokens, np.int32).reshape(rows, head_tokens)
    head_prob = take(rows * head_tokens, np.float64).reshape(rows, head_tokens)
    token_ids = take(token_count, np.int32)
    basis_idx = take(basis_n, np.int32)
    basis_val = take(basis_n, np.float16)
    full_q = None
    if flags & FLAG_FULL_Q:
        full_q = take(rows * vocab, np.float16).reshape(rows, vocab)
    if offset != len(data):
        raise ValueError("trailing bytes in payload")
    if (flags & FLAG_TOKENS) and not token_count:
        raise ValueError("token payload without tokens")
    if rows and token_count and rows != token_count:
        raise ValueError("token count differs from block length")
    return Payload(coefs=coefs, sketch=sketch, head_idx=head_idx, head_prob=head_prob,
                   basis_idx=basis_idx, basis_val=basis_val, full_q=full_q, method=method,
                   token_ids=token_ids,
                   kind=(KIND_OURS if rank else KIND_FULL_Q if full_q is not None else KIND_TOKENS))


def bits_per_step(cfg) -> dict[str, float]:
    """Closed-form byte estimate used by the README and unit tests."""
    rows = cfg.run.draft_len
    rank = cfg.lowrank.rank
    return {
        "coefs": float(rows * rank * 2),
        "sketch": float(rows * cfg.sketch.dim * 2),
        "head": float(rows * cfg.head.tokens * 12),
        "header": float(HEADER.size),
    }
