"""Enlarge the RoPE cache of a DFlash drafter for long inputs.

Some released DFlash drafters declare a max_position_embeddings below the
context length of their target (40,960 for the Llama-3.1-8B drafter), and vLLM
sizes the drafter's cos/sin cache from it. This override only raises the
allocation length. The frequencies and positions are unchanged.

Pass it as EngineArgs(hf_overrides=DraftRopeCache(max_len)). vLLM also applies
the callable to the target config, which is returned unchanged.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftRopeCache:
    max_position_embeddings: int

    def __call__(self, config):
        if tuple(getattr(config, "architectures", None) or ()) != ("DFlashDraftModel",):
            return config
        values = config.to_dict()
        rope = values.get("rope_parameters") or values.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default" or rope.get("factor", 1.0) != 1.0:
            raise ValueError("only unscaled default RoPE can be enlarged without changing positions")
        config.max_position_embeddings = max(int(config.max_position_embeddings), self.max_position_embeddings)
        return config
