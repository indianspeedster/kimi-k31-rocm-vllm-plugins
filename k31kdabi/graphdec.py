"""Sync-free BI KDA decode on the row engine (StaticRowDecoder, decode_step_static): device-side positions and static
shapes, so vLLM FULL CUDA graphs can capture it. Bitwise equal to the eager row decode and to the window path."""
from __future__ import annotations

import torch

from . import gated_rms_norm
from . import rowkern as RK
from .rowdec import BT, RowDecoder

import triton
import triton.language as tl
from fla.ops.utils.op import exp2 as _exp2


@triton.jit
def b_complete(SLOT, RSLOT, POS, S0, SH, KC, G, VN, H0, H, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr):
    """If row t == BT-1 closed the chunk: next boundary state in place in H0 (FLA [K, V] fp32), FLA's chunk_h op order
    for one chunk (h *= exp2(g_last); h += bf16(k * exp2(g_last - g))^T @ bf16(v_new)). Else a no-op."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.load(POS + r)
    if t != BT - 1:
        return
    s = tl.load(SLOT + r).to(tl.int64)
    b = s * H + h
    # H0 may be a strided view (vLLM paged state cache): slot stride S0, head stride SH; each [K, V] head block contiguous
    rb = tl.load(RSLOT + r).to(tl.int64) * S0 + h * SH
    kc = KC + b * BT * K
    gg = G + b * BT * K
    vn = VN + b * BT * V
    h0 = H0 + rb
    o_t = tl.arange(0, BT)
    o_v = tl.arange(0, V)
    b_v = tl.load(vn + o_t[:, None] * V + o_v[None, :])                       # [BT, V] bf16 (= bf16(fp32 v_new))
    for i_k in tl.static_range(K // 64):
        o_k = i_k * 64 + tl.arange(0, 64)
        b_h = tl.load(h0 + o_k[:, None] * V + o_v[None, :]).to(tl.float32)    # [64, V]
        b_gn = tl.load(gg + (BT - 1) * K + o_k).to(tl.float32)                # g at the chunk's last row
        b_h *= _exp2(b_gn)[:, None]
        b_k = tl.load(kc + o_k[:, None] + o_t[None, :] * K)                    # [64, BT] (k^T, bf16)
        b_gk = tl.load(gg + o_k[:, None] + o_t[None, :] * K).to(tl.float32)   # [64, BT]
        b_kg = (b_k * _exp2(b_gn[:, None] - b_gk)).to(b_k.dtype)
        b_h += tl.dot(b_kg, b_v)
        tl.store(h0 + o_k[:, None] * V + o_v[None, :], b_h.to(H0.dtype.element_ty))


@triton.jit
def _conv_last_kernel(HIST, SLOT, RAW, sr, W, ACC, C, BLOCK: tl.constexpr):
    """Pre-SiLU FP32 conv of the new row in the trainer's op order (each mul/add rounded; enable_fp_fusion=False), then
    the hist shift."""
    r = tl.program_id(0)
    s = tl.load(SLOT + r).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < C
    h = HIST + s * 3 * C
    x0 = tl.load(h + offs, mask=m, other=0.0)
    x1 = tl.load(h + C + offs, mask=m, other=0.0)
    x2 = tl.load(h + 2 * C + offs, mask=m, other=0.0)
    x3 = tl.load(RAW + r.to(tl.int64) * sr + offs, mask=m, other=0.0)
    w0 = tl.load(W + offs * 4 + 0, mask=m, other=0.0)
    w1 = tl.load(W + offs * 4 + 1, mask=m, other=0.0)
    w2 = tl.load(W + offs * 4 + 2, mask=m, other=0.0)
    w3 = tl.load(W + offs * 4 + 3, mask=m, other=0.0)
    acc = x3.to(tl.float32) * w3
    acc = acc + x2.to(tl.float32) * w2
    acc = acc + x1.to(tl.float32) * w1
    acc = acc + x0.to(tl.float32) * w0
    tl.store(ACC + r.to(tl.int64) * C + offs, acc, mask=m)
    tl.store(h + offs, x1, mask=m)
    tl.store(h + C + offs, x2, mask=m)
    tl.store(h + 2 * C + offs, x3, mask=m)


@triton.jit
def _tail_kernel(TAIL, SLOT, POS, QKV, sq, G1, sg, B1, sb, P3, P4, NC, C63, BLOCK: tl.constexpr):
    """tail[s, t, :] = (qkv | g1 | b1) of the new row when t < 63 (t = the position before this row)."""
    r = tl.program_id(0)
    s = tl.load(SLOT + r).to(tl.int64)
    t = tl.load(POS + r).to(tl.int64)
    if t >= C63:
        return
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < NC
    rr = r.to(tl.int64)
    a = tl.load(QKV + rr * sq + offs, mask=m & (offs < P3), other=0.0)
    b = tl.load(G1 + rr * sg + (offs - P3), mask=m & (offs >= P3) & (offs < P4), other=0.0)
    c = tl.load(B1 + rr * sb + (offs - P4), mask=m & (offs >= P4), other=0.0)
    v = tl.where(offs < P3, a, tl.where(offs < P4, b, c))
    tl.store(TAIL + (s * C63 + t) * NC + offs, v, mask=m)


@triton.jit
def b_scatter2(SLOT, POS, QN, KN, sqk, VV, sv, GRAW, sg, BETA_IN, Q, KC, VC, GR, BETA, H, D: tl.constexpr, BT: tl.constexpr):
    """Write the new row (q^, k^ from one [R, 2H, D] l2norm output, v, raw gate, beta) into the slot caches at pos."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.load(SLOT + r).to(tl.int64)
    t = tl.load(POS + r).to(tl.int64)
    o = tl.arange(0, D)
    rr = r.to(tl.int64)
    dst = ((s * H + h) * BT + t) * D + o
    tl.store(Q + dst, tl.load(QN + rr * sqk + h * D + o))
    tl.store(KC + dst, tl.load(KN + rr * sqk + h * D + o))
    tl.store(VC + dst, tl.load(VV + rr * sv + h * D + o))
    tl.store(GR + dst, tl.load(GRAW + rr * sg + h * D + o))
    tl.store(BETA + (s * H + h) * BT + t, tl.load(BETA_IN + r * H + h))


class StaticRowDecoder(RowDecoder):
    """RowDecoder with device-side positions and a sync-free decode step (graph-capturable)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pos_dev = torch.zeros(len(self.pos), dtype=torch.int32, device=self.Q.device)

    def rows_static(self, recurrent_state, SL, RSL, q, k, v, graw, braw, A_log, dt_bias):
        """Device-only version of RowDecoder._rows for R = SL.numel() requests (one new row each). Returns core [R, H, D] BF16."""
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.utils.constant import RCP_LN2

        H, D = self.H, self.D
        R = SL.shape[0]
        dev = q.device
        PO = self.pos_dev.index_select(0, SL)                   # int32 [R]
        grid = (R, H)
        from .rowdec import _gate_cfg

        # one l2norm over q|k, strided scatter, per-row gate cumsum
        qk = l2norm_fwd(q_k_pair(q, k).reshape(1, R, 2 * H, D))
        qk = qk[0] if isinstance(qk, tuple) else qk
        qk = qk.reshape(R, 2 * H, D)
        assert v.stride(1) == 1 and graw.stride(1) == 1
        beta_in = braw.reshape(R, H).float().sigmoid()
        b_scatter2[grid](SL, PO, qk, qk[:, H:], 2 * H * D, v, v.stride(0), graw, graw.stride(0), beta_in,
                         self.Q, self.KC, self.VC, self.GR, self.BETA, H, D=D, BT=BT, num_warps=1)
        BS, nw = _gate_cfg()
        RK.b_gate[(R, H, triton.cdiv(D, BS))](SL, PO, self.GR, self.G, A_log, dt_bias, self.lower, RCP_LN2, H, S=D, BT=BT,
                                               BS=BS, num_warps=nw)
        AQK = torch.zeros(R, H, BT, dtype=torch.bfloat16, device=dev)
        AKK = torch.zeros(R, H, BT, dtype=torch.bfloat16, device=dev)
        W = torch.empty(R, H, D, dtype=torch.bfloat16, device=dev)
        U = torch.empty(R, H, D, dtype=torch.bfloat16, device=dev)
        O = torch.empty(R, H, D, dtype=torch.bfloat16, device=dev)
        SL32 = SL.to(torch.int32)
        RSL32 = RSL.to(torch.int32)
        RK.b_diag[grid](SL32, PO, self.Q, self.KC, self.G, self.BETA, AQK, self.AD, self.AP, self.scale, H, K=D, BT=BT, BC=16,
                        num_warps=1)
        RK.b_inter[grid](SL32, PO, self.Q, self.KC, self.G, self.BETA, AQK, self.AD, self.AF, AKK, self.scale, H, K=D, BT=BT,
                         BC=16, num_warps=4)
        RK.b_wu[grid](SL32, PO, self.KC, self.VC, self.G, self.BETA, AKK, W, U, H, K=D, V=D, BT=BT, num_warps=2)
        if recurrent_state.stride(3) != 1 or recurrent_state.stride(2) != D:
            raise RuntimeError(f"K31 BI KDA static decode: per-head state must be a contiguous [K, V] block, strides {recurrent_state.stride()}")
        S0, SH = recurrent_state.stride(0), recurrent_state.stride(1)
        RK.b_stage_a[grid](SL32, RSL32, PO, S0, SH, self.Q, self.G, U, W, AQK, recurrent_state, self.VN, O, self.scale, H, K=D, V=D,
                           BT=BT, R=16, num_warps=4)
        # chunk completion: the new boundary state only where row t closed the chunk
        b_complete[grid](SL32, RSL32, PO, S0, SH, self.KC, self.G, self.VN, recurrent_state, H, K=D, V=D, BT=BT, num_warps=4)
        self.pos_dev.index_copy_(0, SL, ((PO + 1) % BT).to(torch.int32))
        return O


def decode_step_static(st, rows, recurrent_state, SL, RSL, raw, g1f, b1, g2, conv_w, A_log, dt_bias, norm_w, eps):
    """All-decode, all-warm batch, sync-free. st: BIKDAState, rows: its StaticRowDecoder. raw [R, 3P] pre-conv q|k|v,
    g1f [R, P], b1 [R, H], g2 [R, H, D]. Updates hist, tail (pos < 63) and rows.pos_dev; returns normed core [R, H, D] BF16."""
    return _decode_step_lean2(st, rows, recurrent_state, SL, RSL, raw, g1f, b1, g2, conv_w, A_log, dt_bias, norm_w, eps)


def q_k_pair(q, k):
    """q, k are adjacent column blocks of the post-conv qkv rows: return the contiguous [R, 2P] q|k block."""
    if q.data_ptr() + q.shape[1] * q.element_size() == k.data_ptr() and q.stride() == k.stride():
        return q.as_strided((q.shape[0], 2 * q.shape[1]), q.stride()).contiguous()
    return torch.cat([q, k], 1)


def _decode_step_lean2(st, rows, recurrent_state, SL, RSL, raw, g1f, b1, g2, conv_w, A_log, dt_bias, norm_w, eps):
    """Conv-last + hist shift in one kernel, tail update in one kernel (the trainer's conv op order)."""
    P = st.P
    R = SL.shape[0]
    C = 3 * P
    with torch.no_grad():
        acc = torch.empty((R, C), dtype=torch.float32, device=raw.device)
        cw = conv_w if (conv_w.dtype == torch.float32 and conv_w.is_contiguous()) else conv_w.float().contiguous()
        assert raw.stride(1) == 1 and st.hist.is_contiguous()
        _conv_last_kernel[(R, triton.cdiv(C, 1024))](st.hist, SL, raw, raw.stride(0), cw, acc, C, BLOCK=1024, num_warps=4,
                                                     enable_fp_fusion=False)
        qkv = torch.nn.functional.silu(acc).to(raw.dtype)
        PO = rows.pos_dev.index_select(0, SL)                                                   # int32, pos before this row
        core = rows.rows_static(recurrent_state, SL, RSL, qkv[:, :P], qkv[:, P:2 * P], qkv[:, 2 * P:], g1f, b1, A_log, dt_bias)
        NC = st.tail.shape[2]
        assert g1f.stride(1) == 1 and b1.stride(1) == 1 and st.tail.is_contiguous()
        _tail_kernel[(R, triton.cdiv(NC, 1024))](st.tail, SL, PO, qkv, qkv.stride(0), g1f, g1f.stride(0), b1, b1.stride(0),
                                                 C, 4 * P, NC, st.tail.shape[1], BLOCK=1024, num_warps=4)
        return gated_rms_norm(core, g2, norm_w, eps).to(torch.bfloat16)
