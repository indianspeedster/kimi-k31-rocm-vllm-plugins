"""BI KDA row-engine kernels: one new row of the open chunk, bitwise equal to FLA 0.5.2 chunk_kda (FLA's chunk kernels
restricted to row t; rows above t are zero-masked). b_* wrappers run them on a (request, head) grid over per-slot caches."""
import triton
import triton.language as tl

from fla.ops.utils.op import exp, exp2
from fla.ops.kda.chunk_intra import SOLVE_TRIL_DOT_PRECISION as SOLVE_PREC


@triton.jit
def row_stage_a_kernel(q, g, u, w, A, h0, vnew, o, t, scale,
                       K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, R: tl.constexpr):
    """Single head. q/g/u/w/A/vnew: [BT, .] row-major for the open chunk; h0 [K, V] fp32. Writes vnew[t], o[t]."""
    rr = tl.arange(0, R)
    ok = tl.arange(0, K)
    ov = tl.arange(0, V)
    row0 = rr == 0
    # chunk-start state as the h kernel stores it (k dtype = bf16), K split in 64-blocks like FLA
    hb1 = tl.load(h0 + tl.arange(0, 64)[:, None] * V + ov[None, :]).to(tl.bfloat16)
    hb2 = tl.load(h0 + (64 + tl.arange(0, 64))[:, None] * V + ov[None, :]).to(tl.bfloat16)
    # ---- v_new_t = u_t - w_t @ hb  (h kernel: b_v = dot(w1, h1); b_v += dot(w2, h2); b_v = load(u) - b_v)
    w1 = tl.load(w + t * K + tl.arange(0, 64)[None, :] + rr[:, None] * 0, mask=row0[:, None], other=0.0)
    w2 = tl.load(w + t * K + 64 + tl.arange(0, 64)[None, :] + rr[:, None] * 0, mask=row0[:, None], other=0.0)
    b_v = tl.dot(w1, hb1)
    b_v += tl.dot(w2, hb2)
    b_u = tl.load(u + t * V + ov[None, :] + rr[:, None] * 0, mask=row0[:, None], other=0.0)
    b_v = b_u - b_v
    tl.store(vnew + t * V + ov[None, :] + rr[:, None] * 0, b_v.to(vnew.dtype.element_ty), mask=row0[:, None])
    tl.debug_barrier()
    # ---- o_t  (o kernel: b_o += dot(bf16(q*exp2(g)), h) per K block; b_o *= scale; b_o += dot(bf16(mask(A)), v_new))
    b_o = tl.zeros([R, V], dtype=tl.float32)
    for i_k in tl.static_range(2):
        okk = i_k * 64 + tl.arange(0, 64)
        b_q = tl.load(q + t * K + okk[None, :] + rr[:, None] * 0, mask=row0[:, None], other=0.0)
        b_g = tl.load(g + t * K + okk[None, :] + rr[:, None] * 0, mask=row0[:, None], other=0.0).to(tl.float32)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        if i_k == 0:
            b_o += tl.dot(b_qg, hb1.to(b_qg.dtype))
        else:
            b_o += tl.dot(b_qg, hb2.to(b_qg.dtype))
    b_o *= scale
    oi = tl.arange(0, BT)
    b_A = tl.load(A + t * BT + oi[None, :] + rr[:, None] * 0, mask=row0[:, None], other=0.0)
    b_A = tl.where((oi[None, :] <= t) & row0[:, None], b_A, 0.)
    b_vn = tl.load(vnew + oi[:, None] * V + ov[None, :], mask=(oi[:, None] <= t), other=0.0)
    b_A = b_A.to(b_vn.dtype)
    b_o += tl.dot(b_A, b_vn)
    tl.store(o + t * V + ov[None, :] + rr[:, None] * 0, b_o.to(o.dtype.element_ty), mask=row0[:, None])


@triton.jit
def row_diag_kernel(q, k, g, beta, Aqk, Ainv, Apre, t, scale,
                    K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """FLA chunk_kda_fwd_kernel_intra_sub_chunk (SAFE_GATE, BI reference = sub-chunk start) VERBATIM on row t's sub-chunk with
    T = t + 1 (FLA is prefix-invariant here: measured, all chunk lengths give the same rows). Recomputes the sub-chunk's rows <= t
    (<= 16) so the forward substitution runs on the same register tensors / reduction layouts as FLA: a version that assembled b_Ai
    from cached rows changed tl.sum's reduction layout and the FP32 inverse by 1 ulp on some rows (tests/probe_rowdec.py).
    Writes Aqk[t, sub-chunk cols] and Ainv[rows <= t, :BC] (fp32, incl. identity; raw Akk is staged in the same rows as FLA
    stages it in Akkd). Apre is unused (kept for the launch signature)."""
    i_ti = (t // BC) * BC
    Tl = t - i_ti + 1                                # valid rows of this sub-chunk (FLA: min(BC, T - i_ti) with T = t + 1)
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, K)
    o_c = i_ti + o_i
    m_c = o_i < Tl
    b_q = tl.load(q + o_c[:, None] * K + o_k[None, :], mask=m_c[:, None], other=0.0)
    b_k = tl.load(k + o_c[:, None] * K + o_k[None, :], mask=m_c[:, None], other=0.0)
    b_g = tl.load(g + o_c[:, None] * K + o_k[None, :], mask=m_c[:, None], other=0.0)
    b_beta = tl.load(beta + o_c, mask=m_c, other=0.0)
    b_gn = tl.load(g + i_ti * K + o_k)[None, :]
    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)
    b_kgt = tl.trans(b_k * b_gk)
    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)
    is_t = o_c == t
    tl.store(Aqk + o_c[:, None] * BT + (i_ti + o_i)[None, :], b_Aqk.to(Aqk.dtype.element_ty), mask=is_t[:, None])
    p_Akk = Ainv + o_c[:, None] * BC + o_i[None, :]
    m_Akk_st = m_c[:, None] & (o_i[None, :] < BC)
    tl.store(p_Akk, b_Akk.to(Ainv.dtype.element_ty), mask=m_Akk_st)
    tl.debug_barrier()
    b_Ai = -b_Akk
    for i in range(2, Tl):
        b_a = -tl.load(Ainv + (i_ti + i) * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.debug_barrier()
    tl.store(p_Akk, b_Ai.to(Ainv.dtype.element_ty), mask=m_Akk_st)


@triton.jit
def _ld(ptr, rows, cols, ld, mrow):
    return tl.load(ptr + rows[:, None] * ld + cols[None, :], mask=mrow[:, None], other=0.0).to(tl.float32)


@triton.jit
def row_inter_kernel(q, k, g, beta, Aqk, Adiag, Afull, Akk_out, t, scale,
                     K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """Mirror of FLA chunk_kda_fwd_kernel_inter_solve_fused (USE_SAFE_GATE) for row t (sub-chunk c = t // BC >= 0).
    Rows of sub-chunk c above t are masked to 0 (exact: their inverse coefficients for row t are 0). Earlier sub-chunks are complete.
    Adiag [BT, BC] fp32: diagonal-block inverses (stage B1, incl. identity). Afull [BT, BT] fp32: merged-inverse cache (this kernel
    writes row t). Writes Aqk[t, off-diagonal cols] (scaled, Aqk dtype) and Akk_out[t, :] (merged inverse row, Akk dtype)."""
    c = t // BC
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, K)
    r0 = o_i
    r1 = BC + o_i
    r2 = 2 * BC + o_i
    r3 = 3 * BC + o_i
    full = o_i >= 0
    if c == 0:
        b_row = tl.load(Adiag + t * BC + o_i)
        tl.store(Afull + t * BT + o_i, b_row)
        tl.store(Akk_out + t * BT + o_i, b_row.to(Akk_out.dtype.element_ty))
    else:
        rc = c * BC + o_i
        m_c = rc <= t
        b_qc = _ld(q, rc, o_k, K, m_c)
        b_kc = _ld(k, rc, o_k, K, m_c)
        b_gc = _ld(g, rc, o_k, K, m_c)
        b_gn = tl.load(g + c * BC * K + o_k).to(tl.float32)
        b_gqn = tl.where(m_c[:, None], exp2(b_gc - b_gn[None, :]), 0)
        b_qg = b_qc * b_gqn
        b_kg = b_kc * b_gqn
        b_bc = tl.load(beta + rc, mask=m_c, other=0.0).to(tl.float32)
        is_t = rc == t
        # off-diagonal blocks vs sub-chunk 0 (always), 1 (c >= 2), 2 (c == 3)
        b_k0 = _ld(k, r0, o_k, K, full)
        b_g0 = _ld(g, r0, o_k, K, full)
        b_Aq0 = tl.dot(b_qg, tl.trans(b_k0 * exp2(b_gn[None, :] - b_g0)))
        b_Ak0 = tl.dot(b_kg, tl.trans(b_k0 * exp2(b_gn[None, :] - b_g0))) * b_bc[:, None]
        tl.store(Aqk + rc[:, None] * BT + r0[None, :], (b_Aq0 * scale).to(Aqk.dtype.element_ty), mask=is_t[:, None])
        b_Ai00 = _ld(Adiag, r0, o_i, BC, full)
        b_Aicc = _ld(Adiag, rc, o_i, BC, m_c)
        if c == 1:
            b_Ai10 = -tl.dot(tl.dot(b_Aicc, b_Ak0, input_precision=SOLVE_PREC), b_Ai00, input_precision=SOLVE_PREC)
            tl.store(Afull + rc[:, None] * BT + r0[None, :], b_Ai10, mask=is_t[:, None])
            tl.store(Afull + rc[:, None] * BT + r1[None, :], b_Aicc, mask=is_t[:, None])
            tl.store(Akk_out + rc[:, None] * BT + r0[None, :], b_Ai10.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
            tl.store(Akk_out + rc[:, None] * BT + r1[None, :], b_Aicc.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
        else:
            b_k1 = _ld(k, r1, o_k, K, full)
            b_g1 = _ld(g, r1, o_k, K, full)
            b_Aq1 = tl.dot(b_qg, tl.trans(b_k1 * exp2(b_gn[None, :] - b_g1)))
            b_Ak1 = tl.dot(b_kg, tl.trans(b_k1 * exp2(b_gn[None, :] - b_g1))) * b_bc[:, None]
            tl.store(Aqk + rc[:, None] * BT + r1[None, :], (b_Aq1 * scale).to(Aqk.dtype.element_ty), mask=is_t[:, None])
            b_Ai11 = _ld(Adiag, r1, o_i, BC, full)
            b_Ai10 = _ld(Afull, r1, r0, BT, full)
            if c == 2:
                b_Ai21 = -tl.dot(tl.dot(b_Aicc, b_Ak1, input_precision=SOLVE_PREC), b_Ai11, input_precision=SOLVE_PREC)
                b_Ai20 = -tl.dot(b_Aicc, tl.dot(b_Ak0, b_Ai00, input_precision=SOLVE_PREC) +
                                 tl.dot(b_Ak1, b_Ai10, input_precision=SOLVE_PREC), input_precision=SOLVE_PREC)
                tl.store(Afull + rc[:, None] * BT + r0[None, :], b_Ai20, mask=is_t[:, None])
                tl.store(Afull + rc[:, None] * BT + r1[None, :], b_Ai21, mask=is_t[:, None])
                tl.store(Afull + rc[:, None] * BT + r2[None, :], b_Aicc, mask=is_t[:, None])
                tl.store(Akk_out + rc[:, None] * BT + r0[None, :], b_Ai20.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
                tl.store(Akk_out + rc[:, None] * BT + r1[None, :], b_Ai21.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
                tl.store(Akk_out + rc[:, None] * BT + r2[None, :], b_Aicc.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
            else:
                b_k2 = _ld(k, r2, o_k, K, full)
                b_g2 = _ld(g, r2, o_k, K, full)
                b_Aq2 = tl.dot(b_qg, tl.trans(b_k2 * exp2(b_gn[None, :] - b_g2)))
                b_Ak2 = tl.dot(b_kg, tl.trans(b_k2 * exp2(b_gn[None, :] - b_g2))) * b_bc[:, None]
                tl.store(Aqk + rc[:, None] * BT + r2[None, :], (b_Aq2 * scale).to(Aqk.dtype.element_ty), mask=is_t[:, None])
                b_Ai22 = _ld(Adiag, r2, o_i, BC, full)
                b_Ai21 = _ld(Afull, r2, r1, BT, full)
                b_Ai20 = _ld(Afull, r2, r0, BT, full)
                b_Ai32 = -tl.dot(tl.dot(b_Aicc, b_Ak2, input_precision=SOLVE_PREC), b_Ai22, input_precision=SOLVE_PREC)
                b_Ai31 = -tl.dot(b_Aicc, tl.dot(b_Ak1, b_Ai11, input_precision=SOLVE_PREC) +
                                 tl.dot(b_Ak2, b_Ai21, input_precision=SOLVE_PREC), input_precision=SOLVE_PREC)
                b_Ai30 = -tl.dot(b_Aicc, tl.dot(b_Ak0, b_Ai00, input_precision=SOLVE_PREC) +
                                 tl.dot(b_Ak1, b_Ai10, input_precision=SOLVE_PREC) +
                                 tl.dot(b_Ak2, b_Ai20, input_precision=SOLVE_PREC), input_precision=SOLVE_PREC)
                tl.store(Afull + rc[:, None] * BT + r0[None, :], b_Ai30, mask=is_t[:, None])
                tl.store(Afull + rc[:, None] * BT + r1[None, :], b_Ai31, mask=is_t[:, None])
                tl.store(Afull + rc[:, None] * BT + r2[None, :], b_Ai32, mask=is_t[:, None])
                tl.store(Afull + rc[:, None] * BT + r3[None, :], b_Aicc, mask=is_t[:, None])
                tl.store(Akk_out + rc[:, None] * BT + r0[None, :], b_Ai30.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
                tl.store(Akk_out + rc[:, None] * BT + r1[None, :], b_Ai31.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
                tl.store(Akk_out + rc[:, None] * BT + r2[None, :], b_Ai32.to(Akk_out.dtype.element_ty), mask=is_t[:, None])
                tl.store(Akk_out + rc[:, None] * BT + r3[None, :], b_Aicc.to(Akk_out.dtype.element_ty), mask=is_t[:, None])


@triton.jit
def row_wu_kernel(k, v, g, beta, A, w, u, t, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr):
    """Mirror of FLA recompute_w_u_fwd_kda_kernel for row t: u_t = dot(A[t] (Akk dtype), bf16(v*beta)); w_t = dot(A[t], bf16(k*beta*exp2(g)))."""
    rr = tl.arange(0, 16)
    row0 = rr == 0
    o_t = tl.arange(0, BT)
    m_t = o_t <= t
    b_b = tl.load(beta + o_t, mask=m_t, other=0.0)
    b_A = tl.load(A + t * BT + o_t[None, :] + rr[:, None] * 0, mask=row0[:, None], other=0.0)
    o_v = tl.arange(0, V)
    b_v = tl.load(v + o_t[:, None] * V + o_v[None, :], mask=m_t[:, None], other=0.0)
    b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
    b_u = tl.dot(b_A, b_vb)
    tl.store(u + t * V + o_v[None, :] + rr[:, None] * 0, b_u.to(u.dtype.element_ty), mask=row0[:, None])
    o_k = tl.arange(0, K)
    b_k = tl.load(k + o_t[:, None] * K + o_k[None, :], mask=m_t[:, None], other=0.0)
    b_kb = b_k * b_b[:, None]
    b_gk = tl.load(g + o_t[:, None] * K + o_k[None, :], mask=m_t[:, None], other=0.0).to(tl.float32)
    b_kb *= exp2(b_gk)
    b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
    tl.store(w + t * K + o_k[None, :] + rr[:, None] * 0, b_w.to(w.dtype.element_ty), mask=row0[:, None])



# ------------------------------------------------------------------------------------------- batched wrappers
@triton.jit
def b_diag(SLOT, POS, Q, KC, G, BETA, AQK, AD, AP, scale, H, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.load(SLOT + r).to(tl.int64)
    t = tl.load(POS + r)
    b = s * H + h
    rq = (r * H + h).to(tl.int64)
    row_diag_kernel(Q + b * BT * K, KC + b * BT * K, G + b * BT * K, BETA + b * BT,
                    AQK + rq * BT - t * BT, AD + b * BT * BC, AP + b * BT * BC, t, scale, K=K, BT=BT, BC=BC)


@triton.jit
def b_inter(SLOT, POS, Q, KC, G, BETA, AQK, AD, AF, AKK, scale, H, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.load(SLOT + r).to(tl.int64)
    t = tl.load(POS + r)
    b = s * H + h
    rq = (r * H + h).to(tl.int64)
    row_inter_kernel(Q + b * BT * K, KC + b * BT * K, G + b * BT * K, BETA + b * BT,
                     AQK + rq * BT - t * BT, AD + b * BT * BC, AF + b * BT * BT, AKK + rq * BT - t * BT, t, scale,
                     K=K, BT=BT, BC=BC)


@triton.jit
def b_wu(SLOT, POS, KC, VC, G, BETA, AKK, W, U, H, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.load(SLOT + r).to(tl.int64)
    t = tl.load(POS + r)
    b = s * H + h
    rq = (r * H + h).to(tl.int64)
    row_wu_kernel(KC + b * BT * K, VC + b * BT * V, G + b * BT * K, BETA + b * BT, AKK + rq * BT - t * BT,
                  W + rq * K - t * K, U + rq * V - t * V, t, K=K, V=V, BT=BT)


@triton.jit
def b_stage_a(SLOT, RSLOT, POS, S0, SH, Q, G, U, W, AQK, H0, VN, O, scale, H, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
              R: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.load(SLOT + r).to(tl.int64)
    t = tl.load(POS + r)
    b = s * H + h
    rq = (r * H + h).to(tl.int64)
    # caller's recurrent_state may be a strided view (vLLM's paged state cache): slot stride S0, head stride SH (elements);
    # within a head the [K, V] block is contiguous (checked by the caller)
    rb = tl.load(RSLOT + r).to(tl.int64) * S0 + h * SH
    row_stage_a_kernel(Q + b * BT * K, G + b * BT * K, U + rq * V - t * V, W + rq * K - t * K, AQK + rq * BT - t * BT,
                       H0 + rb, VN + b * BT * V, O + rq * V - t * V, t, scale, K=K, V=V, BT=BT, R=R)


@triton.jit
def b_gate(SLOT, POS, GR, G, A_log, dt_bias, lower_bound, scale, H, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr):
    """FLA kda_gate_chunk_cumsum_vector_kernel (USE_LOWER_BOUND, HAS_BIAS, HAS_SCALE, forward) on one slot's open chunk with
    T = t + 1; stores g_cum row t only. Launch with FLA's pinned config (BS, num_warps), grid (R, H, cdiv(S, BS))."""
    r = tl.program_id(0)
    i_h = tl.program_id(1)
    i_s = tl.program_id(2)
    s = tl.load(SLOT + r).to(tl.int64)
    t = tl.load(POS + r)
    T = t + 1
    b = s * H + i_h
    o_t = tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
    b_s = tl.load(GR + b * BT * S + o_t[:, None] * S + o_s[None, :], mask=m_s, other=0.0).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * S + o_s, mask=o_s < S, other=0.0).to(tl.float32)
    b_s = b_s + b_bias[None, :]
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)
    b_o = tl.cumsum(b_gate, axis=0)
    b_o *= scale
    tl.store(G + b * BT * S + o_t[:, None] * S + o_s[None, :], b_o.to(G.dtype.element_ty), mask=m_s & (o_t[:, None] == t))
