"""Triton kernels used by the FOLD attention read (batch size 1).

far_query: rotates the block queries so that every far-memory key, stored
    without rotation, is read at relative distance D + r for block offset r.
    The query at absolute position H + r is rotated by angle (D - H) * freq.
    It also writes far_len = max(0, H - W - B + qlen), the number of far keys.
merge: combines the window and memory outputs by their masses (Eq. 4 of the
    paper). The memory LSE comes from a softmax over alpha * logits, so the
    memory log-mass is lse_far / alpha (Eq. 3).

H = seq_len - qlen is the number of committed entries. All lengths are read on
device, so both kernels can be captured in CUDA graphs.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _far_query_kernel(q_ptr, qf_ptr, qsl_ptr, seq_ptr, inv_ptr, farlen_ptr,
                      sq_t, sq_h, sf_t, sf_h, far_distance, window, block,
                      RH: tl.constexpr, D: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    qlen = tl.load(qsl_ptr + 1) - tl.load(qsl_ptr)
    hist = tl.maximum(tl.load(seq_ptr) - qlen, 0)
    if (t == 0) & (h == 0):
        tl.store(farlen_ptr, tl.maximum(hist - window - block + qlen, 0))
    d = tl.arange(0, D)
    # neox half split inside the rotary prefix of length 2 * RH, rest copied
    rot = d < 2 * RH
    first = d < RH
    k = tl.where(first, d, d - RH)
    k = tl.where(rot, k, 0)
    partner = tl.where(first, d + RH, d - RH)
    partner = tl.where(rot, partner, d)
    x = tl.load(q_ptr + t * sq_t + h * sq_h + d).to(tl.float32)
    xp = tl.load(q_ptr + t * sq_t + h * sq_h + partner).to(tl.float32)
    ang = (far_distance - hist).to(tl.float32) * tl.load(inv_ptr + k, mask=rot, other=0.0)
    c = tl.cos(ang)
    s = tl.sin(ang)
    sign = tl.where(first, -1.0, 1.0)
    y = tl.where(rot, x * c + sign * xp * s, x)
    tl.store(qf_ptr + t * sf_t + h * sf_h + d, y.to(qf_ptr.dtype.element_ty))


@triton.jit
def _rerope_query_kernel(q_ptr, qs_ptr, qf_ptr, qsl_ptr, seq_ptr, inv_ptr, farlen_ptr,
                         sq_t, sq_h, ss_t, ss_h, sf_t, sf_h, far_distance, window, block, inv_log_ref,
                         RH: tl.constexpr, D: tl.constexpr, CONST: tl.constexpr, LOGN: tl.constexpr):
    """ReRoPE baseline. LOGN scales both queries by max(1, log(H + t + 1) / log(ref)).
    CONST rotates the far query so that every row reads every far key at distance D."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    qlen = tl.load(qsl_ptr + 1) - tl.load(qsl_ptr)
    hist = tl.maximum(tl.load(seq_ptr) - qlen, 0)
    if (t == 0) & (h == 0):
        tl.store(farlen_ptr, tl.maximum(hist - window - block + qlen, 0))
    d = tl.arange(0, D)
    rot = d < 2 * RH
    first = d < RH
    k = tl.where(first, d, d - RH)
    k = tl.where(rot, k, 0)
    partner = tl.where(first, d + RH, d - RH)
    partner = tl.where(rot, partner, d)
    x = tl.load(q_ptr + t * sq_t + h * sq_h + d).to(tl.float32)
    xp = tl.load(q_ptr + t * sq_t + h * sq_h + partner).to(tl.float32)
    if LOGN:
        sc = tl.maximum(tl.log((hist + t + 1).to(tl.float32)) * inv_log_ref, 1.0)
        x = x * sc
        xp = xp * sc
    tl.store(qs_ptr + t * ss_t + h * ss_h + d, x.to(qs_ptr.dtype.element_ty))
    off = far_distance - hist
    if CONST:
        off = off - t
    ang = off.to(tl.float32) * tl.load(inv_ptr + k, mask=rot, other=0.0)
    c = tl.cos(ang)
    s = tl.sin(ang)
    sign = tl.where(first, -1.0, 1.0)
    y = tl.where(rot, x * c + sign * xp * s, x)
    tl.store(qf_ptr + t * sf_t + h * sf_h + d, y.to(qf_ptr.dtype.element_ty))


@triton.jit
def _merge_kernel(ol_ptr, ll_ptr, of_ptr, lf_ptr, out_ptr, qsl_ptr, seq_ptr,
                  sol_t, sol_h, sll_h, sll_t, sof_t, sof_h, slf_h, slf_t, so_t, so_h,
                  window, block, inv_alpha, D: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    qlen = tl.load(qsl_ptr + 1) - tl.load(qsl_ptr)
    hist = tl.maximum(tl.load(seq_ptr) - qlen, 0)
    # row t has far entries iff H - W - B + 1 + t > 0
    has_far = (hist - window - block + 1 + t) > 0
    d = tl.arange(0, D)
    ll = tl.load(ll_ptr + h * sll_h + t * sll_t).to(tl.float32)
    ol = tl.load(ol_ptr + t * sol_t + h * sol_h + d).to(tl.float32)
    if has_far:
        lf = tl.load(lf_ptr + h * slf_h + t * slf_t).to(tl.float32) * inv_alpha
        m = tl.maximum(ll, lf)
        wl = tl.exp(ll - m)
        wf = tl.exp(lf - m)
        of = tl.load(of_ptr + t * sof_t + h * sof_h + d).to(tl.float32)
        y = (wl * ol + wf * of) / (wl + wf)
    else:
        y = ol
    tl.store(out_ptr + t * so_t + h * so_h + d, y.to(out_ptr.dtype.element_ty))


def far_query(q, qsl, seq_lens, inv_freq, far_len, *, far_distance, window, block):
    """q [n, Hq, D] -> far query [n, Hq, D]. Writes far_len[0]."""
    n, hq, d = q.shape
    qf = torch.empty((n, hq, d), device=q.device, dtype=q.dtype)
    _far_query_kernel[(n, hq)](q, qf, qsl, seq_lens, inv_freq, far_len,
                               q.stride(0), q.stride(1), qf.stride(0), qf.stride(1),
                               int(far_distance), int(window), int(block),
                               RH=inv_freq.shape[0], D=d)
    return qf


def rerope_queries(q, qsl, seq_lens, inv_freq, far_len, *, far_distance, window, block, constant, logn_ref):
    """q [n, Hq, D] -> (window query, far query) for the ReRoPE baseline. Writes far_len[0]."""
    n, hq, d = q.shape
    qs = torch.empty((n, hq, d), device=q.device, dtype=q.dtype)
    qf = torch.empty((n, hq, d), device=q.device, dtype=q.dtype)
    _rerope_query_kernel[(n, hq)](q, qs, qf, qsl, seq_lens, inv_freq, far_len,
                                  q.stride(0), q.stride(1), qs.stride(0), qs.stride(1),
                                  qf.stride(0), qf.stride(1), int(far_distance), int(window), int(block),
                                  (1.0 / math.log(logn_ref)) if logn_ref else 0.0,
                                  RH=inv_freq.shape[0], D=d, CONST=bool(constant), LOGN=bool(logn_ref))
    return qs, qf


def merge(out_l, lse_l, out_f, lse_f, out, qsl, seq_lens, *, window, block, alpha):
    """out_l, out_f [n, Hq, D], lse_l, lse_f [Hq, n]. Writes the merged output into out [n, Hq, D]."""
    n, hq, d = out_l.shape
    _merge_kernel[(n, hq)](out_l, lse_l, out_f, lse_f, out, qsl, seq_lens,
                           out_l.stride(0), out_l.stride(1), lse_l.stride(0), lse_l.stride(1),
                           out_f.stride(0), out_f.stride(1), lse_f.stride(0), lse_f.stride(1),
                           out.stride(0), out.stride(1),
                           int(window), int(block), 1.0 / float(alpha), D=d)
