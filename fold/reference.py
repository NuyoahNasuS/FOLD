"""Dense PyTorch reference of the FOLD read for one draft block.

It follows the equations of the paper directly and is meant for reading, not
for speed. The vLLM path in attention.py computes the same output with two
FlashAttention calls and kernels.merge.
"""
import torch


def rotate(x, pos, inv_freq):
    """Neox rotary embedding of x [..., N, H, D] at positions pos [N] on the first 2 * len(inv_freq) dims."""
    r = 2 * inv_freq.numel()
    ang = pos.to(torch.float64)[:, None] * inv_freq.to(torch.float64)[None, :]
    c = torch.cat((ang.cos(), ang.cos()), -1)[:, None, :]
    s = torch.cat((ang.sin(), ang.sin()), -1)[:, None, :]
    a, b = x[..., :r].chunk(2, -1)
    y = x[..., :r] * c + torch.cat((-b, a), -1) * s
    return torch.cat((y, x[..., r:]), -1)


def fold_block(q, k_ctx, v_ctx, k_blk, v_blk, *, window, alpha, inv_freq, scale):
    """q, k_blk, v_blk [B, H, D]: unrotated queries, keys and values of the block at t..t+B-1.
    k_ctx, v_ctx [t, H, D]: unrotated keys and values of the context entries 0..t-1.
    Returns the attention output [B, H, D] (float64). GQA heads must be expanded by the caller."""
    q, k_ctx, v_ctx, k_blk, v_blk = (x.double() for x in (q, k_ctx, v_ctx, k_blk, v_blk))
    B, t = q.shape[0], k_ctx.shape[0]
    D = window + B
    qpos = torch.arange(t, t + B)
    kpos = torch.arange(t + B)
    keys = torch.cat((k_ctx, k_blk))
    vals = torch.cat((v_ctx, v_blk))
    out = torch.empty_like(q)
    for r in range(B):
        p = t + r
        near = (p - kpos < D)                              # window N_p, includes every block position
        far = ~near                                        # memory F_p = {j : p - j >= D}, context only
        qr = rotate(q[r:r + 1], qpos[r:r + 1], inv_freq)[0]
        kr = rotate(keys, kpos, inv_freq)
        s = torch.einsum("hd,nhd->hn", qr, kr) * scale    # true relative distances
        s_near = s[:, near]
        z_near = torch.logsumexp(s_near, -1)               # log Z_N
        o_near = torch.einsum("hn,nhd->hd", torch.softmax(s_near, -1), vals[near])
        if far.any():
            # shared virtual position t - D: every memory key at distance D + r (Eq. 1)
            qf = rotate(q[r:r + 1], torch.tensor([D + r]), inv_freq)[0]
            sf = torch.einsum("hd,nhd->hn", qf, keys[far]) * scale
            o_far = torch.einsum("hn,nhd->hd", torch.softmax(alpha * sf, -1), vals[far])  # Eq. 2
            z_far = torch.logsumexp(alpha * sf, -1) / alpha                               # log Z_F, Eq. 3
            w = torch.softmax(torch.stack((z_near, z_far)), 0)                            # Eq. 4
            out[r] = w[0, :, None] * o_near + w[1, :, None] * o_far
        else:
            out[r] = o_near
    return out
