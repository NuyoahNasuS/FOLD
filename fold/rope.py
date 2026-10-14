"""Rotary frequencies of a drafter layer, read from its vLLM rotary module."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rope:
    inv_freq: torch.Tensor  # [rotary_dim / 2], float32
    rotary_dim: int
    head_dim: int


def drafter_rope(rotary_emb, device=None) -> Rope:
    """Supports the default vLLM RotaryEmbedding and Gemma 4's proportional variant
    (neox layout). Scaled variants are rejected because FOLD re-rotates queries
    with the native frequencies."""
    name = type(rotary_emb).__name__
    if name not in {"RotaryEmbedding", "Gemma4RotaryEmbedding"}:
        raise ValueError(f"unsupported rotary embedding: {name}")
    if not rotary_emb.is_neox_style:
        raise ValueError("only neox-style rotary embeddings are supported")
    freq = rotary_emb._compute_inv_freq(float(rotary_emb.base)).detach().clone()
    freq = freq.to(device=device or rotary_emb.cos_sin_cache.device, dtype=torch.float32)
    dim = int(rotary_emb.rotary_dim)
    if dim != 2 * freq.numel() or not 0 < dim <= rotary_emb.head_size:
        raise ValueError("rotary and head dimensions disagree")
    return Rope(freq, dim, int(rotary_emb.head_size))
