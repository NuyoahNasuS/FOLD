"""FOLD attention read for vLLM 0.28 DFlash and DSpark drafters.

Requirements: batch size 1, BF16, FlashAttention 3 or 4 (FlashAttentionImpl).

A wrapped drafter layer keeps its KV cache, its KV update and the absolute RoPE
of its keys. Only the attention read changes. For block position p = t + r:

  window  context entries with p - j < D (D = W + B) and the whole block, read
          from the drafter's KV cache with the FlashAttention window
          [W + B - 1, B - 1] at their true relative distances.
  memory  all older context entries, read from a side cache that stores their
          keys before rotation. The query is rotated once by (D - H), so every
          memory key is read at distance D + r (shared virtual position t - D).

Both reads return their softmax LSE. The memory read uses softmax scale
alpha * scale, and kernels.merge combines the two outputs by their masses.

Sliding-window layers (the Qwen3.5-9B drafter) either keep their native read
(sliding_layers="window") or additionally read the far memory
(sliding_layers="fold"). In the latter case the native window is the local
branch and every older entry is a memory entry at the same distance D + r as
in the full-attention layers.
"""
from __future__ import annotations

import copy
import math
from types import MethodType
from typing import Any

import torch

from .kernels import far_query, merge, rerope_queries
from .rope import drafter_rope

SUPPORTED_DRAFTERS = ("DFlashQwen3ForCausalLM", "Qwen3DSparkForCausalLM", "Gemma4DSparkForCausalLM")
MODES = ("fold", "window", "rerope")


class FoldError(ValueError):
    pass


class FoldWorkspace:
    """Per-layer state: the side cache of unrotated keys and values, indexed by
    the request-local position, and the device-side far length."""

    side_block = 256  # large side pages keep the identity block table short at 256K tokens

    def __init__(self, cache, window, block, max_model_len, *, inv_freq, far_distance, alpha=2.0,
                 mode="fold", logn_ref=0):
        if window < 1 or block < 1:
            raise FoldError("window and block size must be positive")
        if cache.ndim != 4 or cache.shape[-1] % 2:
            raise FoldError("expected a FlashAttention cache [pages, Hkv, block, 2 * head_dim]")
        if cache.shape[0] < 1 or cache.shape[2] % 16:
            raise FoldError("FlashAttention needs allocated pages and a page size divisible by 16")
        if mode not in MODES:
            raise FoldError(f"mode must be one of {MODES}")
        self.window, self.block, self.max_model_len = int(window), int(block), int(max_model_len)
        self.mode, self.alpha, self.logn_ref = mode, float(alpha), int(logn_ref)
        _, hkv, _, d2 = cache.shape
        # the Window ablation keeps no memory
        pages = 1 if mode == "window" else (max_model_len + self.side_block - 1) // self.side_block + 1
        self.side_pages = pages
        self.side = torch.zeros((pages, hkv, self.side_block, d2), device=cache.device, dtype=cache.dtype)
        self.side_table = torch.arange(pages, device=cache.device, dtype=torch.int32)[None]
        self.far_len = torch.zeros(1, device=cache.device, dtype=torch.int32)
        self.inv_freq = inv_freq.to(device=cache.device, dtype=torch.float32)
        self.far_distance = int(far_distance) or self.window + self.block
        # about 64 to 128 CTAs on the memory read: 16 splits for 8 KV heads, 64 for one KV head
        self.far_splits = max(16, 64 // hkv)
        self.local_splits = 16


def logn_scale_rows(q, query_start_loc, seq_lens, logn_ref):
    """ReRoPE query scale max(1, log(p + 1) / log(ref)) at absolute position p of each row."""
    n = q.shape[0]
    qlen = query_start_loc[1:2] - query_start_loc[0:1]
    hist = (seq_lens[0:1] - qlen).clamp_min(0)
    pos = hist.to(torch.float32) + torch.arange(n, device=q.device, dtype=torch.float32)
    s = (torch.log(pos + 1.0) / math.log(logn_ref)).clamp_min(1.0)
    return (q.float() * s[:, None, None]).to(q.dtype)


class _Wrapped:
    def __init__(self, original: Any):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _native(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale):
        return self.original.forward(layer, query, key, value, kv_cache, attn_metadata, output=output,
                                     output_scale=output_scale, output_block_scale=output_block_scale)


class NativeWindowAttention(_Wrapped):
    """Native sliding-window read, optionally with the ReRoPE query scale."""

    def __init__(self, original: Any, logn_ref: int = 0):
        super().__init__(original)
        self.logn_ref = int(logn_ref)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return self._native(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        o = self.original
        n = attn_metadata.num_actual_tokens
        if n == 0:
            return output.zero_()
        nk, nv = kv_cache.transpose(1, 2).split(o.head_size, dim=-1)
        q = logn_scale_rows(query[:n], attn_metadata.query_start_loc, attn_metadata.seq_lens, self.logn_ref) \
            if self.logn_ref else query[:n]
        out = flash_attn_varlen_func(
            q=q, k=nk, v=nv, cu_seqlens_q=attn_metadata.query_start_loc, max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens, max_seqlen_k=attn_metadata.max_seq_len, softmax_scale=o.scale,
            causal=attn_metadata.causal, window_size=list(o.sliding_window), block_table=attn_metadata.block_table,
            fa_version=o.vllm_flash_attn_version, num_splits=16)
        output[:n].copy_(out.reshape(output[:n].shape))
        return output


class FoldAttention(_Wrapped):
    """Full-attention drafter layer: window read plus memory read."""

    def __init__(self, original: Any, workspace: FoldWorkspace):
        super().__init__(original)
        self.workspace = workspace

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return self._native(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)
        if attn_metadata.use_cascade or not isinstance(attn_metadata.causal, bool):
            raise FoldError("cascade attention and dynamic causality are not supported")
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        w, o = self.workspace, self.original
        n = attn_metadata.num_actual_tokens
        if n == 0:
            return output.zero_()
        B = w.block
        qsl, seq = attn_metadata.query_start_loc, attn_metadata.seq_lens
        q = query[:n]
        nk, nv = kv_cache.transpose(1, 2).split(o.head_size, dim=-1)
        local = dict(k=nk, v=nv, cu_seqlens_q=qsl, max_seqlen_q=attn_metadata.max_query_len,
                     seqused_k=seq, max_seqlen_k=attn_metadata.max_seq_len, softmax_scale=o.scale,
                     causal=False, window_size=[w.window + B - 1, B - 1], block_table=attn_metadata.block_table,
                     fa_version=o.vllm_flash_attn_version, num_splits=w.local_splits)
        if w.mode == "window":
            out_l = flash_attn_varlen_func(q=q, **local)
            output[:n].copy_(out_l.reshape(output[:n].shape))
            return output
        if q.stride(-1) != 1:
            q = q.contiguous()
        if w.mode == "rerope":
            q, qf = rerope_queries(q, qsl, seq, w.inv_freq, w.far_len, far_distance=w.far_distance,
                                   window=w.window, block=B, constant=True, logn_ref=w.logn_ref)
        else:
            qf = far_query(q, qsl, seq, w.inv_freq, w.far_len, far_distance=w.far_distance, window=w.window, block=B)
        out_l, lse_l = flash_attn_varlen_func(q=q, return_softmax_lse=True, **local)
        # far(j) = [0, H - W - B + 1 + j): causal bottom-right alignment with seqused_k = far_len
        fk, fv = w.side.transpose(1, 2).split(o.head_size, dim=-1)
        out_f, lse_f = flash_attn_varlen_func(
            q=qf, k=fk, v=fv, cu_seqlens_q=qsl, max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=w.far_len, max_seqlen_k=w.max_model_len, softmax_scale=o.scale * w.alpha, causal=True,
            block_table=w.side_table, return_softmax_lse=True,
            fa_version=o.vllm_flash_attn_version, num_splits=w.far_splits)
        merge(out_l, lse_l, out_f, lse_f, output[:n].view(n, -1, o.head_size), qsl, seq,
              window=w.window, block=B, alpha=w.alpha)
        return output


class SlidingFoldAttention(_Wrapped):
    """Sliding-window drafter layer: native window read plus memory read.

    With native window (w, 0), row j at H + j reads [H + j - w, H + j] natively
    and [0, H + j - w) from the memory. The workspace window is w + 1 - B, so the
    kernels' rule "far iff distance >= window + B" gives exactly this split.
    """

    def __init__(self, original: Any, workspace: FoldWorkspace):
        super().__init__(original)
        if workspace.mode == "window":
            raise FoldError("the Window ablation keeps the native sliding-window read")
        self.workspace = workspace

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return self._native(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)
        if attn_metadata.use_cascade or not isinstance(attn_metadata.causal, bool):
            raise FoldError("cascade attention and dynamic causality are not supported")
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        w, o = self.workspace, self.original
        n = attn_metadata.num_actual_tokens
        if n == 0:
            return output.zero_()
        qsl, seq = attn_metadata.query_start_loc, attn_metadata.seq_lens
        B = w.block
        qn = logn_scale_rows(query[:n], qsl, seq, w.logn_ref) if w.logn_ref else query[:n]
        qp = qn if qn.stride(-1) == 1 else qn.contiguous()
        if w.mode == "rerope":
            _, qf = rerope_queries(qp, qsl, seq, w.inv_freq, w.far_len, far_distance=w.far_distance,
                                   window=w.window, block=B, constant=True, logn_ref=0)
        else:
            qf = far_query(qp, qsl, seq, w.inv_freq, w.far_len, far_distance=w.far_distance, window=w.window, block=B)
        nk, nv = kv_cache.transpose(1, 2).split(o.head_size, dim=-1)
        out_l, lse_l = flash_attn_varlen_func(
            q=qn, k=nk, v=nv, cu_seqlens_q=qsl, max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=seq, max_seqlen_k=attn_metadata.max_seq_len, softmax_scale=o.scale,
            causal=attn_metadata.causal, window_size=list(o.sliding_window), block_table=attn_metadata.block_table,
            return_softmax_lse=True, fa_version=o.vllm_flash_attn_version, num_splits=16)
        fk, fv = w.side.transpose(1, 2).split(o.head_size, dim=-1)
        out_f, lse_f = flash_attn_varlen_func(
            q=qf, k=fk, v=fv, cu_seqlens_q=qsl, max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=w.far_len, max_seqlen_k=w.max_model_len, softmax_scale=o.scale * w.alpha, causal=True,
            block_table=w.side_table, return_softmax_lse=True,
            fa_version=o.vllm_flash_attn_version, num_splits=w.far_splits)
        merge(out_l, lse_l, out_f, lse_f, output[:n].view(n, -1, o.head_size), qsl, seq,
              window=w.window, block=B, alpha=w.alpha)
        return output


def _wrap_context_kv(inner, workspaces):
    """Replace precompute_and_store_context_kv so that it also writes the keys
    (after the key norm, before RoPE) and values of every committed context
    entry into the side cache of each FOLD layer."""
    from vllm import _custom_ops as ops

    def wrapped(context_states, context_positions, context_slot_mapping=None):
        if not hasattr(inner, "_num_attn_layers"):
            inner._build_fused_kv_buffers()
        num_ctx = context_states.shape[0]
        L, kv, hd, nkv = inner._num_attn_layers, inner._kv_size, inner._head_dim, inner._num_kv_heads
        all_k, all_v = inner._project_context_kv(context_states, num_ctx, L, nkv, hd)
        all_k_normed = inner._normalize_context_k(all_k)
        raw = {i: all_k_normed[i].clone() for i in workspaces} if context_slot_mapping is not None else {}
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        cos_sin = inner._rope_cos_sin_cache
        if cos_sin.dtype != all_k_flat.dtype:
            cos_sin = cos_sin.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(context_positions.repeat(L), all_k_flat, None, inner._rope_head_size, cos_sin,
                             inner._rope_is_neox)
        if context_slot_mapping is None:
            return
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i in range(L):
            slots = context_slot_mapping[i] if per_layer else context_slot_mapping
            if slots is None:
                continue
            attn = inner._attn_layers[i]
            attn.impl.do_kv_cache_update(attn, all_k_final[i], all_v[i], attn.kv_cache, slots)
            ws = workspaces.get(i)
            if ws is None or ws.mode == "window":
                continue
            valid = slots[:num_ctx] >= 0
            pos = context_positions[:num_ctx].to(torch.int64)
            pos = torch.where(valid, pos, torch.full_like(pos, ws.side_pages * ws.side_block - 1))
            pg, off = torch.div(pos, ws.side_block, rounding_mode="floor"), pos % ws.side_block
            ws.side[pg, :, off, :hd] = raw[i].to(ws.side.dtype)
            ws.side[pg, :, off, hd:] = all_v[i][:num_ctx].to(ws.side.dtype)

    inner.precompute_and_store_context_kv = wrapped


def use_full_draft_storage(speculator: Any) -> dict:
    """Allocate full KV storage for sliding-window drafter layers so that vLLM
    does not recycle their old pages. Attention masks and windows are unchanged.
    Call after the drafter is loaded and before the KV cache is allocated."""
    old = getattr(speculator, "_fold_full_storage", None)
    if old is not None:
        return old
    names = list(speculator.model.get_draft_kv_cache_layer_names())
    context = speculator.vllm_config.compilation_config.static_forward_context
    changed = []
    for name in names:
        layer = context[name]
        if getattr(layer, "sliding_window", None) is None:
            continue
        cache = getattr(layer, "kv_cache", None)
        if isinstance(cache, torch.Tensor) and cache.numel():
            raise FoldError("full draft storage must be requested before KV allocation")
        original = layer.get_kv_cache_spec
        if not hasattr(original, "__func__"):
            raise FoldError("unexpected KV spec override")

        def full_spec(this, vllm_config, original_function=original.__func__):
            proxy = copy.copy(this)
            proxy.sliding_window = None
            return original_function(proxy, vllm_config)

        layer.get_kv_cache_spec = MethodType(full_spec, layer)
        changed.append(name)
    speculator._fold_full_storage = dict(layers=changed)
    return speculator._fold_full_storage


def install(speculator: Any, *, window: int, alpha: float = 2.0, mode: str = "fold",
            sliding_layers: str = "window", logn_ref: int = 0, local_splits: int = 16) -> dict:
    """Wrap the attention of every drafter layer. Call after KV allocation and
    before CUDA graph capture.

    window          W, the number of recent context entries kept at exact positions
    alpha           concentration factor of the memory
    mode            "fold", "window" (FOLD without memory) or "rerope" (baseline)
    sliding_layers  "window" keeps the native read of sliding-window layers,
                    "fold" lets them read the memory as well
    logn_ref        reference length of the ReRoPE query scale (rerope mode)
    local_splits    FlashAttention split count of the window read
    """
    previous = getattr(speculator, "_fold_installation", None)
    if previous is not None:
        return previous
    if mode not in MODES or sliding_layers not in ("window", "fold"):
        raise FoldError("invalid mode or sliding_layers")
    if mode == "rerope" and not logn_ref:
        raise FoldError("rerope mode needs logn_ref")
    if mode != "rerope":
        logn_ref = 0
    if type(speculator.model).__name__ not in SUPPORTED_DRAFTERS:
        raise FoldError(f"supported drafters: {SUPPORTED_DRAFTERS}")
    cfg = speculator.vllm_config
    if cfg.scheduler_config.max_num_seqs != 1:
        raise FoldError("batch size 1 only")
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    inner = speculator.model.model
    names = list(speculator.model.get_draft_kv_cache_layer_names())
    context = cfg.compilation_config.static_forward_context
    B = int(speculator.num_query_per_req)
    D = window + B
    report = dict(mode=mode, window=window, block=B, far_distance=D, alpha=alpha, logn_ref=logn_ref,
                  sliding_layers=sliding_layers, fold_layers=[], native_window_layers=[], side_cache_bytes=0)
    workspaces = {}
    for li, dl in enumerate(inner.layers):
        name = dl.self_attn.attn.layer_name
        if name not in names:
            raise FoldError(f"{name} is not a drafter KV layer")
        layer = context[name]
        impl = layer.impl
        if type(impl) is not FlashAttentionImpl or impl.vllm_flash_attn_version not in (3, 4):
            raise FoldError(f"{name}: FlashAttention 3 or 4 is required")
        native = int(impl.sliding_window[0])
        if native >= 0:
            if native > window:
                raise FoldError(f"{name}: native window is wider than W")
            if sliding_layers == "fold" and mode != "window":
                if impl.sliding_window[1] != 0 or native + 1 - B < 1:
                    raise FoldError(f"{name}: expected a causal native window wider than the block")
                rope = drafter_rope(dl.self_attn.rotary_emb)
                ws = FoldWorkspace(layer.kv_cache, native + 1 - B, B, speculator.max_model_len, inv_freq=rope.inv_freq,
                                   far_distance=D, alpha=alpha, mode=mode, logn_ref=logn_ref)
                layer.impl = SlidingFoldAttention(impl, ws)
                workspaces[li] = ws
                report["fold_layers"].append(name)
                report["side_cache_bytes"] += ws.side.numel() * ws.side.element_size()
            else:
                layer.impl = NativeWindowAttention(impl, logn_ref)
                report["native_window_layers"].append(name)
            continue
        rope = drafter_rope(dl.self_attn.rotary_emb)
        ws = FoldWorkspace(layer.kv_cache, window, B, speculator.max_model_len, inv_freq=rope.inv_freq,
                           far_distance=D, alpha=alpha, mode=mode, logn_ref=logn_ref)
        ws.local_splits = int(local_splits)
        layer.impl = FoldAttention(impl, ws)
        workspaces[li] = ws
        report["fold_layers"].append(name)
        report["side_cache_bytes"] += ws.side.numel() * ws.side.element_size()
    _wrap_context_kv(inner, workspaces)
    speculator._fold_workspaces = workspaces
    speculator._fold_installation = report
    return report
