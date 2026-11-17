"""Batch-invariant MXFP4 routed-expert MoE for Kimi-K3.1 on ROCm, shared by the vLLM rollout and the mLite trainer.

Per slot (token t, expert e): gu = bf16(x @ deq(W13_e)^T); act = bf16(situ(fp32(gu) + d13)); y = act @ deq(W2_e)^T over
8 canonical intermediate chunks (FP32 pair tree, the TP8 layout at every TP / EP); leaf = w * (fp32(bf16(y)) + d2);
out = bf16(expert-id-ascending FP32 pair tree over the k = 16 leaves), -0 -> +0. d13 / d2 = FP32 expert-LoRA deltas.
Every GEMM uses pinned Triton tiles, so a row's bits do not depend on the batch. EP: leaves are all-reduced in FP32.
No vLLM imports in the math (the trainer calls bi_moe_forward)."""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# pinned tile: do NOT autotune; trainer and rollout must use the same values
BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 32, 64, 64, 4, 2
N_CHUNKS = 8   # canonical split of the intermediate dim (= TP8 shards)


def _pick_cfg(rows, K, N, mx):
    """Tile per (shape, rows); every candidate gives the pinned reference's bits (speed only)."""
    if not mx:
        return None
    T = rows // 16 + 1
    if K == 384:                                        # W2 canonical chunk
        return (32, 64, 64, 4, 2) if T <= 1 else (64, 64, 64, 4, 2) if T <= 8 else \
            (16, 128, 64, 4, 2) if T <= 256 else (32, 128, 64, 4, 2)
    if N >= 4096:                                       # W13 at TP1/2
        return (16, 64, 128, 4, 2) if T <= 1 else (16, 256, 64, 8, 2) if T <= 256 else (64, 64, 64, 4, 2)
    return (64, 64, 256, 8, 2) if T <= 1 else (16, 128, 128, 8, 2) if T <= 8 else \
        (16, 64, 128, 4, 2) if T <= 64 else (16, 128, 64, 4, 2) if T <= 256 else (64, 128, 128, 8, 2)


@triton.jit
def _e2m1(c):
    sign = (c >> 3) & 1
    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    p = tl.where(e == 3, 4.0, tl.where(e == 2, 2.0, 1.0))
    mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * p)
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _grouped_gemm_kernel(A, A_ROWS, W, S, ADD, OUT, TILE_E, TILE_R0, TILE_R1, N, K,
                         stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_om, stride_addm, WSC,
                         MXFP4: tl.constexpr, HAS_ADD: tl.constexpr, SHARED_W: tl.constexpr, FP32_DOT: tl.constexpr,
                         W_TO_F32: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    if SHARED_W:
        e = e * 0
    r0 = tl.load(TILE_R0 + pid_t)
    r1 = tl.load(TILE_R1 + pid_t)
    if r0 >= r1:
        return                                    # empty (graph-padding) tile
    offs_m = r0 + tl.arange(0, BM)
    m_mask = offs_m < r1
    om = offs_m.to(tl.int64)
    rows = tl.load(A_ROWS + offs_m, mask=m_mask, other=0).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < K
        a = tl.load(A + rows[:, None] * stride_am + offs_k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        if MXFP4:
            offs_kb = k0 // 2 + tl.arange(0, BK // 2)
            b = tl.load(W + e * stride_we + offs_n[:, None] * stride_wn + offs_kb[None, :],
                        mask=n_mask[:, None] & (offs_kb[None, :] < K // 2), other=0)
            offs_ks = k0 // 32 + tl.arange(0, BK // 32)
            s = tl.load(S + e * stride_se + offs_n[:, None] * stride_sn + offs_ks[None, :],
                        mask=n_mask[:, None] & (offs_ks[None, :] < K // 32), other=127)
            lo = _e2m1(b & 0xF)
            hi = _e2m1((b >> 4) & 0xF)
            w = tl.reshape(tl.join(lo, hi), (BN, BK))                       # low nibble = even element
            sc = (s.to(tl.int32) << 23).to(tl.float32, bitcast=True)      # 2^(byte-127), exact
            w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
            w = w.to(tl.bfloat16)                                         # exact
        else:
            w = tl.load(W + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            if W_TO_F32:
                w = w.to(tl.float32) * WSC                 # == torch (B.float() * scaling): fp32 convert + fp32 mul
        if FP32_DOT:
            acc += tl.dot(a, tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(a, tl.trans(w))
    if HAS_ADD:
        acc += tl.load(ADD + om[:, None] * stride_addm + offs_n[None, :], mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    tl.store(OUT + om[:, None] * stride_om + offs_n[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gg_scaled_kernel(A, A_ROWS, W, S, OUT, TILE_E, TILE_R0, TILE_R1, N, K, stride_am, stride_we, stride_wn, stride_se,
                      stride_sn, stride_oc, stride_om, CH_A, CH_W, CH_S, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr):
    """MXFP4 grouped GEMM with tl.dot_scaled (same MFMA chain as the dequant kernel, same bits). program_id(2) = chunk c:
    A columns / W, S K-ranges offset by c * CH_*,
    output plane c (OUT + c * stride_oc). Row r of a chunk = A[A_ROWS[r], c*CH_A : c*CH_A + K] @ W[e, :, chunk]^T."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    c = tl.program_id(2).to(tl.int64)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    r0 = tl.load(TILE_R0 + pid_t)
    r1 = tl.load(TILE_R1 + pid_t)
    if r0 >= r1:
        return
    offs_m = r0 + tl.arange(0, BM)
    m_mask = offs_m < r1
    rows = tl.load(A_ROWS + offs_m, mask=m_mask, other=0).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    nn = tl.where(n_mask, offs_n, 0).to(tl.int64)
    pa = A + c * CH_A + rows[:, None] * stride_am + tl.arange(0, BK)[None, :]
    pw = W + e * stride_we + c * CH_W + nn[None, :] * stride_wn + tl.arange(0, BK // 2)[:, None]
    ps = S + e * stride_se + c * CH_S + nn[:, None] * stride_sn + tl.arange(0, BK // 32)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(pa, mask=m_mask[:, None], other=0.0)
        w = tl.load(pw)
        sc = tl.load(ps)
        acc = tl.dot_scaled(a, None, "bf16", w, sc, "e2m1", acc=acc, fast_math=True)
        pa += BK
        pw += BK // 2
        ps += BK // 32
    om = offs_m.to(tl.int64)
    tl.store(OUT + c * stride_oc + om[:, None] * stride_om + offs_n[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


def scaled_gemm(a, a_rows, w, scale, tiles, n_out, chunks=1, out=None, cfg=None):
    """FP32 [chunks, R, n_out]: plane c = a[a_rows, c*Kc:(c+1)*Kc] @ deq(w[e, :, chunk c])^T, Kc = K_total / chunks.
    w uint8 [E, n_out, K_total/2] (raw MXFP4, may be a slice with stride(-1) == 1), scale uint8 [E, n_out, K_total/32].
    Same values as grouped_gemm on each chunk (dequant engine / a16w4 engine); one launch for all chunks."""
    tile_e, r0, r1, bm = tiles
    R = a_rows.numel()
    Kt = w.shape[2] * 2
    Kc = Kt // chunks
    wv, sv = w.view(torch.uint8), scale.view(torch.uint8)
    assert a.stride(1) == 1 and wv.stride(2) == 1 and sv.stride(2) == 1 and Kc % 32 == 0
    if out is None:
        out = torch.empty((chunks, R, n_out), dtype=torch.float32, device=a.device)
    if R == 0 or tile_e.numel() == 0:
        return out
    if cfg is None:
        cfg = (128, 128 if Kc % 128 == 0 else 64, 4, 2)
    bn, bk, nw, ns = cfg
    if Kc % bk:
        bk = 64 if Kc % 64 == 0 else 32
    grid = (tile_e.numel(), triton.cdiv(n_out, bn), chunks)
    _gg_scaled_kernel[grid](a, a_rows, wv, sv, out, tile_e, r0, r1, n_out, Kc, a.stride(0), wv.stride(0), wv.stride(1),
                            sv.stride(0), sv.stride(1), out.stride(0), out.stride(1), Kc, Kc // 2, Kc // 32,
                            BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns)
    return out


@triton.jit
def _tree8_rows_kernel(P, OUT, NROWS, N, stride_pc, stride_pm, stride_om, BR: tl.constexpr, BN: tl.constexpr):
    """OUT[r] = ((P0+P1)+(P2+P3))+((P4+P5)+(P6+P7)) for rows < NROWS (== the progressive chunk tree)."""
    pid_r = tl.program_id(0)
    pid_n = tl.program_id(1)
    nrows = tl.load(NROWS)
    r0 = pid_r * BR
    if r0 >= nrows:
        return
    offs_r = r0 + tl.arange(0, BR)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m = (offs_r < nrows)[:, None] & (offs_n < N)[None, :]
    p = P + offs_r.to(tl.int64)[:, None] * stride_pm + offs_n[None, :]
    x0 = tl.load(p, mask=m, other=0.0)
    x1 = tl.load(p + stride_pc, mask=m, other=0.0)
    x2 = tl.load(p + 2 * stride_pc, mask=m, other=0.0)
    x3 = tl.load(p + 3 * stride_pc, mask=m, other=0.0)
    x4 = tl.load(p + 4 * stride_pc, mask=m, other=0.0)
    x5 = tl.load(p + 5 * stride_pc, mask=m, other=0.0)
    x6 = tl.load(p + 6 * stride_pc, mask=m, other=0.0)
    x7 = tl.load(p + 7 * stride_pc, mask=m, other=0.0)
    y = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))
    tl.store(OUT + offs_r.to(tl.int64)[:, None] * stride_om + offs_n[None, :], y, mask=m)


def tree8_rows(P, nrows):
    C, R, N = P.shape
    assert C == 8 and P.stride(2) == 1
    out = torch.empty((R, N), dtype=torch.float32, device=P.device)
    if R:
        BR, BN = 8, 512
        _tree8_rows_kernel[(triton.cdiv(R, BR), triton.cdiv(N, BN))](P, out, nrows, N, P.stride(0), P.stride(1),
                                                                    out.stride(0), BR=BR, BN=BN, num_warps=4)
    return out


@triton.jit
def _route_count_kernel(BUCKET, S, CNT, NB: tl.constexpr, BS: tl.constexpr):
    """block b: CNT[b, j] = #slots of bucket j in slots [b*BS, (b+1)*BS)."""
    b = tl.program_id(0)
    eb = tl.arange(0, NB)
    sb = b * BS + tl.arange(0, BS)
    bk = tl.load(BUCKET + sb, mask=sb < S, other=NB)
    tl.store(CNT + b * NB + eb, tl.sum((bk[:, None] == eb[None, :]).to(tl.int32), axis=0))


@triton.jit
def _route_scan_kernel(CNT, NBLK, E_LOC, BASE, COUNTS, TILE_E, TILE_R0, TILE_R1, MAX_TILES, NB: tl.constexpr,
                       BT: tl.constexpr, BMT: tl.constexpr):
    """program 0: BASE[b, j] = start[j] + #slots of bucket j in blocks < b (stable order), COUNTS; program p >= 1: tile
    entries [(p-1)*BT, p*BT) of the padded tile table (== _route_kernel)."""
    pid = tl.program_id(0)
    eb = tl.arange(0, NB)
    cnt = tl.zeros((NB,), dtype=tl.int32)
    for b in range(0, NBLK):
        cnt += tl.load(CNT + b * NB + eb)
    start = tl.cumsum(cnt, 0) - cnt
    loc = eb < NB - 1
    if pid == 0:
        run = start
        for b in range(0, NBLK):
            tl.store(BASE + b * NB + eb, run)
            run += tl.load(CNT + b * NB + eb)
        tl.store(COUNTS + eb, cnt, mask=loc)
    else:
        ntile = tl.where(loc, (cnt + BMT - 1) // BMT, 0)
        tcum = tl.cumsum(ntile, 0)
        tfirst = tcum - ntile
        total = tl.sum(ntile, 0)
        tt = (pid - 1) * BT + tl.arange(0, BT)
        e = tl.sum((tcum[None, :] <= tt[:, None]).to(tl.int32), axis=1)
        e = tl.minimum(e, E_LOC - 1)
        oh = (e[:, None] == eb[None, :]).to(tl.int32)
        e_start = tl.sum(oh * start[None, :], axis=1)
        e_cnt = tl.sum(oh * cnt[None, :], axis=1)
        e_first = tl.sum(oh * tfirst[None, :], axis=1)
        v = tt < total
        r0 = tl.where(v, e_start + (tt - e_first) * BMT, 0)
        r1 = tl.where(v, e_start + e_cnt, 0)
        m = tt < MAX_TILES
        tl.store(TILE_E + tt, e, mask=m)
        tl.store(TILE_R0 + tt, r0, mask=m)
        tl.store(TILE_R1 + tt, r1, mask=m)


@triton.jit
def _route_scatter_kernel(BUCKET, S, K, BASE, ORDER, ROWS_SLOT, TOK, NB: tl.constexpr, BS: tl.constexpr):
    b = tl.program_id(0)
    eb = tl.arange(0, NB)
    sb = b * BS + tl.arange(0, BS)
    valid = sb < S
    bk = tl.load(BUCKET + sb, mask=valid, other=NB)
    base = tl.load(BASE + b * NB + tl.minimum(bk, NB - 1), mask=valid, other=0)
    jj = tl.arange(0, BS)
    same = (bk[:, None] == bk[None, :]) & (jj[None, :] < jj[:, None])
    pos = base + tl.sum(same.to(tl.int32), axis=1)
    tl.store(ORDER + pos, sb, mask=valid)
    tl.store(ROWS_SLOT + sb, pos, mask=valid)
    tl.store(TOK + pos, sb // K, mask=valid)


def route_par(bucket, E, k, bm, max_tiles):
    """Device routing (stable counting sort) in three launches: count per block, scan + tiles, scatter."""
    S = bucket.numel()
    dev = bucket.device
    NB = triton.next_power_of_2(E + 1)
    b = bucket.to(torch.int32)
    if NB != E + 1:
        b = torch.where(b == E, NB - 1, b)
    BS = 64
    nblk = triton.cdiv(S, BS)
    cnt = torch.empty((nblk, NB), dtype=torch.int32, device=dev)
    base = torch.empty((nblk, NB), dtype=torch.int32, device=dev)
    order = torch.empty(S, dtype=torch.int32, device=dev)
    rows_slot = torch.empty(S, dtype=torch.int32, device=dev)
    tok = torch.empty(S, dtype=torch.int32, device=dev)
    counts = torch.empty(NB, dtype=torch.int32, device=dev)
    te = torch.empty(max_tiles, dtype=torch.int32, device=dev)
    t0 = torch.empty(max_tiles, dtype=torch.int32, device=dev)
    t1 = torch.empty(max_tiles, dtype=torch.int32, device=dev)
    _route_count_kernel[(nblk,)](b, S, cnt, NB=NB, BS=BS, num_warps=4)
    BT = 32
    _route_scan_kernel[(1 + triton.cdiv(max_tiles, BT),)](cnt, nblk, E, base, counts, te, t0, t1, max_tiles, NB=NB, BT=BT,
                                                         BMT=bm, num_warps=4)
    _route_scatter_kernel[(nblk,)](b, S, k, base, order, rows_slot, tok, NB=NB, BS=BS, num_warps=4)
    return order.to(torch.int64), rows_slot, tok, counts[:E].to(torch.int64), (te, t0, t1, bm)


def _tiles_device(counts: torch.Tensor, max_tiles: int, bm: int = None):
    """Graph-safe tiling (no host sync): a STATIC number of tiles (upper bound cdiv(S, bm) + E); tiles past the real
    count are empty (r0 == r1 -> fully masked in the kernel)."""
    bm = bm or BLOCK_M
    dev = counts.device
    ntile = (counts + bm - 1) // bm
    cum = torch.cumsum(ntile, 0)
    starts = torch.cumsum(counts, 0) - counts
    t = torch.arange(max_tiles, device=dev)
    e = torch.searchsorted(cum, t, right=True).clamp_(max=counts.numel() - 1)
    first = cum[e] - ntile[e]
    r0 = starts[e] + (t - first) * bm
    r1 = starts[e] + counts[e]
    valid = t < cum[-1]
    r0 = torch.where(valid, r0, torch.zeros_like(r0))
    r1 = torch.where(valid, r1, torch.zeros_like(r1))
    return e.to(torch.int32), r0.to(torch.int32), r1.to(torch.int32), bm


def grouped_gemm(a, a_rows, w, tiles, n_out, scale=None, add=None, shared_w=False, cfg=None, wscale=None, out=None):
    """out[r] = a[a_rows[r]] @ w[e(r)]^T in FP32 (+ add[r]). w: MXFP4 packed uint8 [E, N, K/2] with scale uint8
    [E, N, K/32], or BF16 [E, N, K] (scale None). a: BF16 [*, K]. Rows grouped by expert as described by tiles."""
    tile_e, r0, r1 = tiles[:3]
    R = a_rows.numel()
    K = a.shape[1]
    if out is None:
        out = torch.empty((R, n_out), dtype=torch.float32, device=a.device)
    else:   # caller-provided FP32 view (e.g. one half of the [R, 2N] LoRA delta buffer): same values, no torch.cat
        assert out.dtype == torch.float32 and tuple(out.shape) == (R, n_out) and out.stride(1) == 1
    if R == 0 or tile_e.numel() == 0:
        return out
    mx = scale is not None
    wv = w.view(torch.uint8) if mx else w
    sv = scale.view(torch.uint8) if mx else wv
    fp32_dot = a.dtype == torch.float32
    assert a.dtype in (torch.bfloat16, torch.float32) and a.stride(1) == 1 and wv.stride(-1) == 1
    w_to_f32 = (not mx) and wscale is not None
    if not mx:
        assert wv.dtype == a.dtype or (w_to_f32 and a.dtype == torch.float32 and wv.dtype == torch.bfloat16)
    tbm = tiles[3] if len(tiles) > 3 else BLOCK_M
    if cfg is None:
        p = _pick_cfg(R, a.shape[1], n_out, scale is not None)
        cfg = (tbm,) + p[1:] if p else (tbm, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES)
    bm, bn, bk0, nw, ns = cfg
    assert bm == tbm, "tile BM must match the tiling"
    bk = bk0 if K % bk0 == 0 else 32                    # LoRA expand has K = r = 32
    assert K % bk == 0, f"K={K} must be a multiple of 32"
    if add is not None:
        assert add.dtype == torch.float32 and add.shape == out.shape and add.stride(1) == 1
    grid = (tile_e.numel(), triton.cdiv(n_out, bn))
    _grouped_gemm_kernel[grid](
        a, a_rows, wv, sv, add if add is not None else out, out, tile_e, r0, r1, n_out, K,
        a.stride(0), wv.stride(0), wv.stride(1), sv.stride(0), sv.stride(1), out.stride(0),
        add.stride(0) if add is not None else 0, float(wscale) if w_to_f32 else 1.0,
        MXFP4=mx, HAS_ADD=add is not None, SHARED_W=shared_w, FP32_DOT=fp32_dot, W_TO_F32=w_to_f32,
        BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns)
    return out


def situ_fp32(gu: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """pre FP32 [R, 2N] = [gate | up] -> BF16 [R, N]. Same expression as the trainer (rocm_mxfp4_moe.py:149-153);
    the trainer should call this function so the torch kernels match."""
    d = gu.shape[-1] // 2
    g, u = gu[:, :d], gu[:, d:]
    g = (beta * torch.tanh(g / beta)) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (g * u).to(torch.bfloat16)


@triton.jit
def _pre_situ_kernel(GU, ADD, MASK, TOK, NROWS, OUT, N, stride_gu, stride_add, stride_o, INV_B, B, INV_L, L,
                     HAS_ADD: tl.constexpr, HAS_MASK: tl.constexpr, HAS_L: tl.constexpr, BR: tl.constexpr,
                     BN: tl.constexpr):
    """act = bf16(situ_fp32(fp32(bf16(gu)) + add * mask[tok])) for rows < NROWS (rows past it are never read: no tile
    covers them, the combines mask them). Same IEEE ops in the same order as the torch path; launched without FP fusion.
    x / beta == x * fl32(1/beta) (ATen div by a CPU scalar); sigmoid = 1 / (1 + exp(-x)) with a correctly rounded
    divide; tanh / exp = the OCML functions torch calls."""
    pid_r = tl.program_id(0)
    pid_n = tl.program_id(1)
    nrows = tl.load(NROWS)
    r0 = pid_r * BR
    if r0 >= nrows:
        return
    offs_r = r0 + tl.arange(0, BR)
    offs_n = pid_n * BN + tl.arange(0, BN)
    rm = offs_r < nrows
    m = rm[:, None] & (offs_n < N)[None, :]
    orr = offs_r.to(tl.int64)
    pg = GU + orr[:, None] * stride_gu + offs_n[None, :]
    g = tl.load(pg, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
    u = tl.load(pg + N, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32)
    if HAS_ADD:
        pa = ADD + orr[:, None] * stride_add + offs_n[None, :]
        ag = tl.load(pa, mask=m, other=0.0)
        au = tl.load(pa + N, mask=m, other=0.0)
        if HAS_MASK:
            mk = tl.load(MASK + tl.load(TOK + offs_r, mask=rm, other=0).to(tl.int64), mask=rm, other=0.0)
            ag = ag * mk[:, None]
            au = au * mk[:, None]
        g = g + ag
        u = u + au
    sg = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-g))
    gg = (B * libdevice.tanh(g * INV_B)) * sg
    if HAS_L:
        u = L * libdevice.tanh(u * INV_L)
    tl.store(OUT + orr[:, None] * stride_o + offs_n[None, :], (gg * u).to(tl.bfloat16), mask=m)


def _f32(v):
    return float(torch.tensor(v, dtype=torch.float32))


def pre_situ(gu, add13, mask, tok, nrows, beta, linear_beta):
    """Fused form of: pre = fp32(bf16(gu)) [+ add13 * mask[tok][:, None]]; situ_fp32(pre, beta, linear_beta). gu FP32
    [R, 2N] (raw GEMM accumulator), add13 FP32 [R, 2N] or None, mask FP32 [T] or None, tok int32 [R], nrows int64 [1]
    (device: rows >= nrows are skipped). -> BF16 [R, N] (rows >= nrows uninitialised)."""
    R, N2 = gu.shape
    N = N2 // 2
    out = torch.empty((R, N), dtype=torch.bfloat16, device=gu.device)
    if R == 0:
        return out
    assert gu.stride(1) == 1 and (add13 is None or (add13.stride(1) == 1 and add13.shape == gu.shape))
    inv = lambda b: _f32(_f32(1.0) / _f32(b))  # noqa: E731   fl32(1/fl32(b)) as ATen computes it on the host
    BR, BN = 16, 256
    _pre_situ_kernel[(triton.cdiv(R, BR), triton.cdiv(N, BN))](
        gu, add13 if add13 is not None else gu, mask if mask is not None else gu, tok, nrows, out, N, gu.stride(0),
        add13.stride(0) if add13 is not None else 0, out.stride(0), inv(beta), _f32(beta),
        inv(linear_beta) if linear_beta is not None else 1.0, _f32(linear_beta) if linear_beta is not None else 1.0,
        HAS_ADD=add13 is not None, HAS_MASK=add13 is not None and mask is not None, HAS_L=linear_beta is not None,
        BR=BR, BN=BN, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _add_rows_kernel(A, B, NROWS, N, stride_a, stride_b, BR: tl.constexpr, BN: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_n = tl.program_id(1)
    nrows = tl.load(NROWS)
    r0 = pid_r * BR
    if r0 >= nrows:
        return
    offs_r = r0 + tl.arange(0, BR)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m = (offs_r < nrows)[:, None] & (offs_n < N)[None, :]
    orr = offs_r.to(tl.int64)
    pa = A + orr[:, None] * stride_a + offs_n[None, :]
    a = tl.load(pa, mask=m, other=0.0)
    b = tl.load(B + orr[:, None] * stride_b + offs_n[None, :], mask=m, other=0.0)
    tl.store(pa, a + b, mask=m)


def add_rows_(a, b, nrows):
    """a[:nrows] += b[:nrows] (FP32, in place; nrows a device int64 [1]); rows past nrows untouched. Returns a."""
    R, N = a.shape
    if R:
        BR, BN = 16, 512
        _add_rows_kernel[(triton.cdiv(R, BR), triton.cdiv(N, BN))](a, b, nrows, N, a.stride(0), b.stride(0), BR=BR, BN=BN,
                                                                    num_warps=4)
    return a


def tree_dim0(x: torch.Tensor) -> torch.Tensor:
    """adjacent-pair FP32 tree over dim 0 (power of 2): ((x0+x1)+(x2+x3))+..."""
    while x.shape[0] > 1:
        assert x.shape[0] % 2 == 0
        x = x[0::2] + x[1::2]
    return x[0]


def _chunked_contract(a, ar, w, tiles, n_out, chunk, scale=None, gather=None, nrows=None):
    """FP32 [R, n_out] = tree over the N_CHUNKS global chunks of (a[:, chunk c] @ w[:, :, chunk c]^T); a holds this rank's
    contiguous N slice (C_local chunks); gather(P [C_local, R, n_out]) -> [N_CHUNKS, R, n_out] across TP ranks in order."""
    C = a.shape[1] // chunk

    def part(c):
        ac = a[:, c * chunk:(c + 1) * chunk]
        if scale is not None:
            return grouped_gemm(ac, ar, w[:, :, c * chunk // 2:(c + 1) * chunk // 2], tiles, n_out,
                                scale=scale[:, :, c * chunk // 32:(c + 1) * chunk // 32])
        return grouped_gemm(ac, ar, w[:, :, c * chunk:(c + 1) * chunk], tiles, n_out)

    if gather is not None:
        P = gather(torch.stack([part(c) for c in range(C)]))
        assert P.shape[0] == N_CHUNKS, f"{P.shape[0]} chunks, expected {N_CHUNKS}"
        return tree_dim0(P)
    assert C == N_CHUNKS, f"{C} local chunks, expected {N_CHUNKS} (or pass gather)"
    # progressive adjacent-pair tree (same sums as tree_dim0, peak = log2(C)+1 partials)
    stack = []
    for c in range(C):
        lvl, t = 0, part(c)
        while stack and stack[-1][0] == lvl:
            t = stack.pop()[1] + t if nrows is None else add_rows_(stack.pop()[1], t, nrows)
            lvl += 1
        stack.append((lvl, t))
    assert len(stack) == 1
    return stack[0][1]



@triton.jit
def _leaf(Y, ADD, W, ROWS, j, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD: tl.constexpr, MASK,
          HAS_MASK: tl.constexpr):
    r = tl.load(ROWS + t * 16 + j)                       # sorted-row index of this slot (-1: not local)
    w = tl.load(W + t * 16 + j)
    rr = tl.maximum(r, 0).to(tl.int64)
    y = tl.load(Y + rr * stride_y + offs_h, mask=h_mask & (r >= 0), other=0.0)
    yf = y.to(tl.bfloat16).to(tl.float32)
    if HAS_ADD:
        ad = tl.load(ADD + rr * stride_add + offs_h, mask=h_mask & (r >= 0), other=0.0)
        if HAS_MASK:
            ad = ad * tl.load(MASK + t)                  # == torch add2 * token_mask[tok][:, None] (launch: no FP fusion)
        yf = yf + ad
    w = tl.where(r >= 0, w, 0.0)
    return w * yf


@triton.jit
def _combine16_kernel(Y, ADD, W, ROWS, IDS, OUT, H, stride_y, stride_add, stride_o, MASK, HAS_ADD: tl.constexpr,
                      HAS_MASK: tl.constexpr, BH: tl.constexpr):
    """Fused leaf + expert-ascending adjacent-pair FP32 tree + BF16 + (-0 -> +0) for top-16. Same arithmetic as
    leaf/slots/tree_combine_expert_ascending in torch (bitwise)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BH + tl.arange(0, BH)
    h_mask = offs_h < H
    jj = tl.arange(0, 16)
    ids = tl.load(IDS + t * 16 + jj).to(tl.int64)
    # stable rank of slot j in expert-id order (ties by slot index: duplicate ids from the trainer's sentinel are safe)
    lt = ids[None, :] < ids[:, None]
    tie = (ids[None, :] == ids[:, None]) & (jj[None, :] < jj[:, None])
    rank = tl.sum((lt | tie).to(tl.int32), axis=1)
    j0 = tl.sum(tl.where(rank == 0, jj, 0))
    j1 = tl.sum(tl.where(rank == 1, jj, 0))
    j2 = tl.sum(tl.where(rank == 2, jj, 0))
    j3 = tl.sum(tl.where(rank == 3, jj, 0))
    j4 = tl.sum(tl.where(rank == 4, jj, 0))
    j5 = tl.sum(tl.where(rank == 5, jj, 0))
    j6 = tl.sum(tl.where(rank == 6, jj, 0))
    j7 = tl.sum(tl.where(rank == 7, jj, 0))
    j8 = tl.sum(tl.where(rank == 8, jj, 0))
    j9 = tl.sum(tl.where(rank == 9, jj, 0))
    j10 = tl.sum(tl.where(rank == 10, jj, 0))
    j11 = tl.sum(tl.where(rank == 11, jj, 0))
    j12 = tl.sum(tl.where(rank == 12, jj, 0))
    j13 = tl.sum(tl.where(rank == 13, jj, 0))
    j14 = tl.sum(tl.where(rank == 14, jj, 0))
    j15 = tl.sum(tl.where(rank == 15, jj, 0))
    l0 = _leaf(Y, ADD, W, ROWS, j0, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l1 = _leaf(Y, ADD, W, ROWS, j1, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l2 = _leaf(Y, ADD, W, ROWS, j2, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l3 = _leaf(Y, ADD, W, ROWS, j3, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l4 = _leaf(Y, ADD, W, ROWS, j4, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l5 = _leaf(Y, ADD, W, ROWS, j5, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l6 = _leaf(Y, ADD, W, ROWS, j6, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l7 = _leaf(Y, ADD, W, ROWS, j7, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l8 = _leaf(Y, ADD, W, ROWS, j8, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l9 = _leaf(Y, ADD, W, ROWS, j9, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l10 = _leaf(Y, ADD, W, ROWS, j10, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l11 = _leaf(Y, ADD, W, ROWS, j11, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l12 = _leaf(Y, ADD, W, ROWS, j12, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l13 = _leaf(Y, ADD, W, ROWS, j13, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l14 = _leaf(Y, ADD, W, ROWS, j14, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    l15 = _leaf(Y, ADD, W, ROWS, j15, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
    out = ((((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))) + (((l8 + l9) + (l10 + l11)) + ((l12 + l13) + (l14 + l15))))
    o = out.to(tl.bfloat16)
    o = tl.where(o == 0, 0.0, o).to(tl.bfloat16)
    tl.store(OUT + t * stride_o + offs_h, o, mask=h_mask)


def combine16(y, add2, wsel_tok, rows_tok, topk_ids, mask=None):
    """y FP32 [R, H] (sorted rows); add2 FP32 [R, H] or None; wsel_tok FP32 [T, 16] (0 for non-local); rows_tok int32
    [T, 16] (row of each slot, -1 = not local); topk_ids [T, 16]. -> BF16 [T, H]."""
    T = topk_ids.shape[0]
    H = y.shape[1]
    out = torch.empty((T, H), dtype=torch.bfloat16, device=y.device)
    BH = 256
    _combine16_kernel[(T, triton.cdiv(H, BH))](y, add2 if add2 is not None else y, wsel_tok.contiguous(),
                                               rows_tok.contiguous(), topk_ids.contiguous(), out, H, y.stride(0),
                                               add2.stride(0) if add2 is not None else 0, out.stride(0),
                                               mask if mask is not None else y, HAS_ADD=add2 is not None,
                                               HAS_MASK=mask is not None, BH=BH, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _leaf16_scatter_kernel(Y, ADD, W, ROWS, SLOTS, H, stride_y, stride_add, MASK, HAS_ADD: tl.constexpr,
                           HAS_MASK: tl.constexpr, BH: tl.constexpr):
    """EP: every slot's leaf (combine16's _leaf: w * (bf16(y) + add), exactly +0 for non-local slots) in SLOT order into
    SLOTS FP32 [T*16, H] (replaces zeros + index_put + multiply of the torch path; same values)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BH + tl.arange(0, BH)
    h_mask = offs_h < H
    for j in tl.static_range(16):
        lj = _leaf(Y, ADD, W, ROWS, j, t, offs_h, h_mask, stride_y, stride_add, HAS_ADD, MASK, HAS_MASK)
        tl.store(SLOTS + (t * 16 + j).to(tl.int64) * H + offs_h, lj, mask=h_mask)


@triton.jit
def _ld_slot(SLOTS, t, j, H, offs_h, h_mask):
    return tl.load(SLOTS + (t * 16 + j).to(tl.int64) * H + offs_h, mask=h_mask, other=0.0)


@triton.jit
def _tree16_slots_kernel(SLOTS, IDS, OUT, H, stride_o, BH: tl.constexpr):
    """EP: expert-ascending adjacent-pair FP32 tree over the 16 (all-reduced) leaves of each token + BF16 + (-0 -> +0);
    the tree half of _combine16_kernel (same order), reading precomputed leaves."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BH + tl.arange(0, BH)
    h_mask = offs_h < H
    jj = tl.arange(0, 16)
    ids = tl.load(IDS + t * 16 + jj).to(tl.int64)
    # stable rank of slot j in expert-id order (ties by slot index: duplicate ids from the trainer's sentinel are safe)
    lt = ids[None, :] < ids[:, None]
    tie = (ids[None, :] == ids[:, None]) & (jj[None, :] < jj[:, None])
    rank = tl.sum((lt | tie).to(tl.int32), axis=1)
    l0 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 0, jj, 0)), H, offs_h, h_mask)
    l1 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 1, jj, 0)), H, offs_h, h_mask)
    l2 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 2, jj, 0)), H, offs_h, h_mask)
    l3 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 3, jj, 0)), H, offs_h, h_mask)
    l4 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 4, jj, 0)), H, offs_h, h_mask)
    l5 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 5, jj, 0)), H, offs_h, h_mask)
    l6 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 6, jj, 0)), H, offs_h, h_mask)
    l7 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 7, jj, 0)), H, offs_h, h_mask)
    l8 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 8, jj, 0)), H, offs_h, h_mask)
    l9 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 9, jj, 0)), H, offs_h, h_mask)
    l10 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 10, jj, 0)), H, offs_h, h_mask)
    l11 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 11, jj, 0)), H, offs_h, h_mask)
    l12 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 12, jj, 0)), H, offs_h, h_mask)
    l13 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 13, jj, 0)), H, offs_h, h_mask)
    l14 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 14, jj, 0)), H, offs_h, h_mask)
    l15 = _ld_slot(SLOTS, t, tl.sum(tl.where(rank == 15, jj, 0)), H, offs_h, h_mask)
    out = ((((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))) + (((l8 + l9) + (l10 + l11)) + ((l12 + l13) + (l14 + l15))))
    o = out.to(tl.bfloat16)
    o = tl.where(o == 0, 0.0, o).to(tl.bfloat16)
    tl.store(OUT + t * stride_o + offs_h, o, mask=h_mask)


def combine16_ep(y, add2, w_tok, rows_tok, topk_ids, leaf_reduce, mask=None):
    """EP version of combine16: leaves in slot order (one kernel) -> leaf_reduce (FP32 all-reduce over the EP group, exact:
    one non-zero contributor per element) -> tree (one kernel). rows_tok int32 [T, 16]: sorted row of each slot, -1 = not
    on this rank; w_tok FP32 [T, 16] (masked to 0 for -1 rows inside). -> BF16 [T, H]."""
    T = topk_ids.shape[0]
    H = y.shape[1]
    BH = 256
    grid = (T, triton.cdiv(H, BH))
    slots = torch.empty((T * 16, H), dtype=torch.float32, device=y.device)
    _leaf16_scatter_kernel[grid](y, add2 if add2 is not None else y, w_tok.contiguous(), rows_tok.contiguous(), slots, H,
                                 y.stride(0), add2.stride(0) if add2 is not None else 0, mask if mask is not None else y,
                                 HAS_ADD=add2 is not None, HAS_MASK=mask is not None, BH=BH, num_warps=4,
                                 enable_fp_fusion=False)
    slots = leaf_reduce(slots)
    out = torch.empty((T, H), dtype=torch.bfloat16, device=y.device)
    _tree16_slots_kernel[grid](slots, topk_ids.contiguous(), out, H, out.stride(0), BH=BH, num_warps=4)
    return out


def tree_combine_expert_ascending(leaf_slots: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """leaf_slots FP32 [T, k, H] in slot order -> FP32 [T, H]: adjacent-pair tree over the k leaves sorted by expert id
    ascending (k a power of 2). Matches the trainer's tree16_root contract."""
    T, k, H = leaf_slots.shape
    perm = torch.argsort(topk_ids, dim=1, stable=True)
    x = torch.gather(leaf_slots, 1, perm[:, :, None].expand(T, k, H))
    while x.shape[1] > 1:
        assert x.shape[1] % 2 == 0
        x = x[:, 0::2] + x[:, 1::2]
    return x[:, 0]


@triton.jit
def _gg_bf16_chunks_kernel(A, A_ROWS, W, OUT, TILE_E, TILE_R0, TILE_R1, N, K, stride_am, stride_we, stride_wn, stride_oc,
                           stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """_grouped_gemm_kernel's BF16 path (same loads / dot chain) with a chunk grid axis: plane c = A[:, c*K:(c+1)*K] @
    W[e, :, c*K:(c+1)*K]^T -> OUT + c * stride_oc. One launch for the 8 canonical chunks of the expert-LoRA A2 shrink."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    c = tl.program_id(2).to(tl.int64)
    e = tl.load(TILE_E + pid_t).to(tl.int64)
    r0 = tl.load(TILE_R0 + pid_t)
    r1 = tl.load(TILE_R1 + pid_t)
    if r0 >= r1:
        return
    offs_m = r0 + tl.arange(0, BM)
    m_mask = offs_m < r1
    om = offs_m.to(tl.int64)
    rows = tl.load(A_ROWS + offs_m, mask=m_mask, other=0).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    A += c * K
    W += c * K
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < K
        a = tl.load(A + rows[:, None] * stride_am + offs_k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(W + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    tl.store(OUT + c * stride_oc + om[:, None] * stride_om + offs_n[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


def bf16_chunks_gemm(a, a_rows, w, tiles, n_out, chunks):
    """FP32 [chunks, R, n_out]; plane c == grouped_gemm(a[:, chunk c], a_rows, w[:, :, chunk c], tiles, n_out) (BF16 w)."""
    tile_e, r0, r1, bm = tiles
    R = a_rows.numel()
    Kc = w.shape[2] // chunks
    assert a.stride(1) == 1 and w.stride(2) == 1 and a.dtype == w.dtype == torch.bfloat16 and Kc % BLOCK_K == 0
    out = torch.empty((chunks, R, n_out), dtype=torch.float32, device=a.device)
    if R == 0 or tile_e.numel() == 0:
        return out
    _gg_bf16_chunks_kernel[(tile_e.numel(), triton.cdiv(n_out, BLOCK_N), chunks)](
        a, a_rows, w, out, tile_e, r0, r1, n_out, Kc, a.stride(0), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        BM=bm, BN=BLOCK_N, BK=BLOCK_K, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


_ARANGE = {}


def _arange_i32(n, dev):
    """Constant int32 arange, cached when created outside CUDA-graph capture (a capture-time tensor lives in the graph
    pool and is only written at replay, so it is never cached; a cached eager one is static memory -> capture-safe)."""
    key = (n, str(dev))
    t = _ARANGE.get(key)
    if t is None:
        t = torch.arange(n, device=dev, dtype=torch.int32)
        if n <= 65536 and not torch.cuda.is_current_stream_capturing():
            _ARANGE[key] = t
    return t


def _token_tiles(T, bm, dev):
    """Tiles covering rows 0..T-1 of ONE (shared) weight: (0, i*bm, min((i+1)*bm, T)). Constant -> cached like _arange_i32."""
    key = ("tt", T, bm, str(dev))
    t = _ARANGE.get(key)
    if t is None:
        nt = triton.cdiv(T, bm)
        r0 = _arange_i32(nt, dev) * bm
        t = (torch.zeros(nt, device=dev, dtype=torch.int32), r0, torch.clamp(r0 + bm, max=T), bm)
        if T <= 65536 and not torch.cuda.is_current_stream_capturing():
            _ARANGE[key] = t
    return t


def _lora_glue(a, a_rows, tiles, A, B, scaling, shared_A, shared_B, chunk=None, out=None, nrows=None, tok_tiles=None):
    """Expert-LoRA delta, same values as _expert_lora_delta:
    shared A (d1, d3): the shrink a @ A^T is the same for every slot of a token -> computed once per TOKEN (T rows, not
    T*k) and gathered by the expand (a_rows = tok); chunked A2 shrink: the 8 chunks in one launch + row-bounded tree8."""
    if chunk is None and shared_A and tok_tiles is not None:
        T = a.shape[0]
        t = grouped_gemm(a, _arange_i32(T, a.device), A, tok_tiles, A.shape[1], shared_w=True)
        rows = a_rows
    elif chunk is not None and not shared_A and nrows is not None and A.shape[2] == N_CHUNKS * chunk:
        t = tree8_rows(bf16_chunks_gemm(a, a_rows, A, tiles, A.shape[1], N_CHUNKS), nrows)
        rows = _arange_i32(t.shape[0], t.device)
    else:
        return _expert_lora_delta(a, a_rows, tiles, A, B, scaling, shared_A, shared_B, chunk=chunk, out=out, nrows=nrows)
    if B.dtype == torch.bfloat16 and B.stride(-1) == 1:
        return grouped_gemm(t, rows, B, tiles, B.shape[1], shared_w=shared_B, wscale=scaling, out=out)
    return grouped_gemm(t, rows, (B.float() * scaling).contiguous(), tiles, B.shape[1], shared_w=shared_B, out=out)


def _expert_lora_delta(a, a_rows, tiles, A, B, scaling, shared_A, shared_B, chunk=None, gather=None, out=None, nrows=None):
    """FP32 delta[r] = (a[a_rows[r]] @ A_e^T)_fp32 @ (B_e * s)^T_fp32  (A [E|1, r, K], B [E|1, N, r] BF16).
    chunk/gather: contraction over the (TP-sharded) intermediate dim is done in the canonical N_CHUNKS split."""
    if chunk is None:
        t = grouped_gemm(a, a_rows, A, tiles, A.shape[1], shared_w=shared_A)          # FP32, no BF16 cast
    else:
        assert not shared_A
        t = _chunked_contract(a, a_rows, A, tiles, A.shape[1], chunk, gather=gather, nrows=nrows)
    ar = torch.arange(t.shape[0], device=t.device, dtype=torch.int32)
    if B.dtype == torch.bfloat16 and B.stride(-1) == 1:   # B * s formed in the kernel's load (same values)
        return grouped_gemm(t, ar, B, tiles, B.shape[1], shared_w=shared_B, wscale=scaling, out=out)
    bs = (B.float() * scaling).contiguous()
    return grouped_gemm(t, ar, bs, tiles, B.shape[1], shared_w=shared_B, out=out)      # FP32-IEEE MFMA


SCALED_W2_MAX_ROWS = 2048   # bound of the [8, R, H] FP32 partials of the one-launch W2
ROUTE_MAX_SLOTS = 1024      # device routing (route_par) up to this many slots and ROUTE_MAX_E local experts; else torch
ROUTE_MAX_E = 128
TOK_CHUNK = int(os.environ.get("K31_BI_MOE_TOK_CHUNK", "1024"))   # token block per forward (memory bound; rows independent)
BM_LARGE = int(os.environ.get("K31_BI_MOE_BM_LARGE", "64"))       # row tile for blocks of >= 1024 tokens (rows independent)


@torch.no_grad()
def bi_moe_forward(x, w13, s13, w2, s2, topk_weights, topk_ids, beta, linear_beta=None, lora=None,
                   n_global=None, chunk_gather=None, expert_map=None, leaf_reduce=None):
    """Token-chunked driver of _bi_moe_forward_tokens (every rank of a TP/EP group sees the same T, so the per-chunk
    collectives stay matched). Arguments as in _bi_moe_forward_tokens."""
    T = x.shape[0]
    if T <= TOK_CHUNK:
        return _bi_moe_forward_tokens(x, w13, s13, w2, s2, topk_weights, topk_ids, beta, linear_beta, lora,
                                      n_global, chunk_gather, expert_map, leaf_reduce)
    def lsl(i):
        if lora is None or "token_mask" not in lora:
            return lora
        return dict(lora, token_mask=lora["token_mask"][i:i + TOK_CHUNK])
    outs = [_bi_moe_forward_tokens(x[i:i + TOK_CHUNK], w13, s13, w2, s2, topk_weights[i:i + TOK_CHUNK],
                                   topk_ids[i:i + TOK_CHUNK], beta, linear_beta, lsl(i), n_global, chunk_gather,
                                   expert_map, leaf_reduce) for i in range(0, T, TOK_CHUNK)]
    return torch.cat(outs)


def _bi_moe_forward_tokens(x, w13, s13, w2, s2, topk_weights, topk_ids, beta, linear_beta=None, lora=None,
                           n_global=None, chunk_gather=None, expert_map=None, leaf_reduce=None):
    """x BF16 [T, H]; w13 uint8 [E, 2N, H/2], s13 uint8 [E, 2N, H/32]; w2 uint8 [E, H, N/2], s2 [E, H, N/32];
    topk_weights FP32 [T, k]; topk_ids [T, k]. lora (optional): dict of BF16 factors A1, A3 [E|1, r, H], B1, B3 [E|1, N, r],
    A2 [E|1, r, N], B2 [E|1, H, r] (N = this rank's intermediate slice), "scaling" float, and shared_<name> bools
    (default: A1, A3, B2 shared [1, ...]; B1, B3, A2 per expert).
    n_global: full intermediate size (N_local * tp); chunk_gather: TP all-gather of FP32 chunk partials (TP>1);
    expert_map: global -> local expert id (-1 = not on this rank) for EP; leaf_reduce: FP32 sum over the EP group.
    Returns BF16 [T, H] (identical on every rank of the TP / EP group)."""
    T, H = x.shape
    k = topk_ids.shape[1]
    E = w13.shape[0]
    N2 = w13.shape[1]
    n_global = n_global or N2 // 2
    chunk = n_global // N_CHUNKS
    assert n_global % N_CHUNKS == 0 and (N2 // 2) % chunk == 0 and chunk % BLOCK_K == 0
    if k != 16:
        raise NotImplementedError(f"K31 BI MoE: top-k {k} (the fused combines need k = 16)")
    x = x.contiguous()
    gids = topk_ids.reshape(-1).to(torch.int64)
    ids = gids if expert_map is None else expert_map[gids].to(torch.int64)
    local = ids >= 0
    bucket = torch.where(local, ids, E)
    bm = 16 if T <= 256 else (BM_LARGE if T >= 1024 else 64)   # one BM per forward (tiles shared by all GEMMs)
    rows_slot = None
    if bucket.numel() <= ROUTE_MAX_SLOTS and E <= ROUTE_MAX_E:
        order, rows_slot, tok, counts, tiles = route_par(
            bucket, E, k, bm, triton.cdiv(bucket.numel(), bm) + min(E, bucket.numel()))
    else:
        # stable sort by local expert (non-local slots at the tail, never covered by a tile); no host sync
        order = torch.argsort(bucket, stable=True)
        counts = torch.zeros(E + 1, dtype=torch.int64, device=x.device).scatter_add_(
            0, bucket, torch.ones_like(bucket))[:E]
        tiles = _tiles_device(counts, triton.cdiv(order.numel(), bm) + min(E, order.numel()), bm)
        tok = (order // k).to(torch.int32)
    return _glue_tail(x, w13, s13, w2, s2, topk_weights, topk_ids, beta, linear_beta, lora, chunk_gather, expert_map,
                      leaf_reduce, local, order, rows_slot, tok, counts, tiles, chunk)


def _glue_tail(x, w13, s13, w2, s2, topk_weights, topk_ids, beta, linear_beta, lora, chunk_gather, expert_map,
               leaf_reduce, local, order, rows_slot, tok, counts, tiles, chunk):
    """After routing: LoRA deltas, w13, pre-SiTU, W2 and the k = 16 combine. Rows >= nrows (this rank's local slots)
    are never computed nor read."""
    T, H = x.shape
    k = topk_ids.shape[1]
    N2 = w13.shape[1]
    R = tok.numel()
    dev = x.device
    nrows = counts.sum().reshape(1)
    nr = nrows if chunk_gather is None else None          # TP gather path: keep the plain tree (all rows are local)
    mask = lora.get("token_mask") if lora is not None else None
    add13 = add2 = None
    if lora is not None:
        sh = lambda n, d: lora.get("shared_" + n, d)  # noqa: E731
        add13 = torch.empty((R, N2), dtype=torch.float32, device=dev)
        tt = _token_tiles(T, tiles[3], dev)
        _lora_glue(x, tok, tiles, lora["A1"], lora["B1"], lora["scaling"], sh("A1", True), sh("B1", False),
                   out=add13[:, :N2 // 2], tok_tiles=tt)
        _lora_glue(x, tok, tiles, lora["A3"], lora["B3"], lora["scaling"], sh("A3", True), sh("B3", False),
                   out=add13[:, N2 // 2:], tok_tiles=tt)
    gu = scaled_gemm(x, tok, w13, s13, tiles, N2)[0]
    act = pre_situ(gu, add13, mask, tok, nrows, beta, linear_beta)
    del gu, add13
    ar = _arange_i32(R, dev)
    if lora is not None:
        if chunk_gather is None:
            add2 = _lora_glue(act, ar, tiles, lora["A2"], lora["B2"], lora["scaling"], lora.get("shared_A2", False),
                              lora.get("shared_B2", True), chunk=chunk, nrows=nrows)
        else:
            add2 = _expert_lora_delta(act, ar, tiles, lora["A2"], lora["B2"], lora["scaling"], lora.get("shared_A2", False),
                                      lora.get("shared_B2", True), chunk=chunk, gather=chunk_gather, nrows=nr)
    if chunk_gather is None and R <= SCALED_W2_MAX_ROWS and w2.shape[2] * 2 == N_CHUNKS * chunk:
        y = tree8_rows(scaled_gemm(act, ar, w2, s2, tiles, H, chunks=N_CHUNKS), nrows)
    else:
        y = _chunked_contract(act, ar, w2, tiles, H, chunk, scale=s2, gather=chunk_gather, nrows=nr)   # FP32; leaf rounds
    m2 = mask if add2 is not None else None
    tw = topk_weights.to(torch.float32).reshape(T, k)
    if leaf_reduce is None:
        if rows_slot is None:
            rows_slot = torch.full((T * k,), -1, dtype=torch.int32, device=dev)
            rows_slot[order] = torch.arange(order.numel(), dtype=torch.int32, device=dev)
        if expert_map is not None:                       # (not a vLLM case) non-local slots -> row -1 -> exact +0
            rows_slot = torch.where(local, rows_slot, -1)
        return combine16(y, add2, tw, rows_slot.view(T, k), topk_ids, mask=m2)
    if rows_slot is not None:
        rows_ep = torch.where(local, rows_slot, -1)
    else:
        rows_ep = torch.full((T * k,), -1, dtype=torch.int32, device=dev)
        rows_ep[order] = torch.where(local[order], torch.arange(order.numel(), dtype=torch.int32, device=dev),
                                     torch.full_like(order, -1, dtype=torch.int32))
    return combine16_ep(y, add2, tw, rows_ep.view(T, k), topk_ids, leaf_reduce, mask=m2)


# ------------------------------------------------------------------------------------------------ vLLM integration
def _lora_from_context(ctx, num_tokens):
    """vLLM MoELoRAContext -> bi_moe_forward LoRA dict, without host syncs (graph-capturable).

    One adapter slot (max_loras = 1); tokens mapped to it get the delta (token_mask), others none. vLLM stores B
    pre-scaled (bf16(B * s)), equal to the trainer's B * s only for a power-of-two s = alpha / r (checked at adapter
    load and by the trainer), hence scaling = 1."""
    if ctx is None:
        return None
    if ctx.max_loras != 1:
        raise NotImplementedError(f"K31 BI MoE: expert LoRA supports one adapter slot (max_loras=1), got {ctx.max_loras}")
    meta = getattr(ctx.punica_wrapper, "token_mapping_meta", None)
    if meta is not None and bool(meta.no_lora_flag_cpu.item()):   # CPU flag: no device sync
        return None          # no LoRA in this batch (vLLM leaves the mapping stale then)
    m = getattr(ctx, "local_token_lora_mapping", None)
    if m is None:
        m = ctx.punica_wrapper.token_mapping_meta.token_lora_mapping
    m = m[:num_tokens]
    slot = m.amax().clamp(min=0).reshape(1).to(torch.int64)
    # a token mapped to a disabled adapter slot would silently get a zero delta
    torch._assert_async(((m < 0).all() | (ctx.adapter_enabled.index_select(0, slot) > 0).all()),
                        "K31 BI MoE: tokens mapped to a disabled expert-LoRA adapter slot")
    pick = lambda t: t[0] if t.shape[0] == 1 else t.index_select(0, slot)[0]  # noqa: E731
    a1, a3 = pick(ctx.w13_lora_a_stacked[0]), pick(ctx.w13_lora_a_stacked[1])
    b1, b3 = pick(ctx.w13_lora_b_stacked[0]), pick(ctx.w13_lora_b_stacked[1])
    a2, b2 = pick(ctx.w2_lora_a_stacked[0]), pick(ctx.w2_lora_b_stacked[0])
    return {"A1": a1, "A3": a3, "B1": b1, "B3": b3, "A2": a2, "B2": b2, "scaling": 1.0,
            "shared_A1": a1.shape[0] == 1, "shared_A3": a3.shape[0] == 1, "shared_B1": b1.shape[0] == 1,
            "shared_B3": b3.shape[0] == 1, "shared_A2": a2.shape[0] == 1, "shared_B2": b2.shape[0] == 1,
            "token_mask": (m == slot[0]).to(torch.float32)}


def _check_lora_graph_mode():
    """_lora_from_context's no-LoRA branch is taken at capture time: with CUDA graphs, vLLM must capture separate graphs
    for batches with and without LoRA (cudagraph_specialize_lora). Called at experts construction (config is set)."""
    from vllm.config import get_current_vllm_config

    cc = get_current_vllm_config().compilation_config
    mode = getattr(cc, "cudagraph_mode", None)
    graphs = mode is not None and str(getattr(mode, "name", mode)).upper() != "NONE"
    if graphs and not cc.cudagraph_specialize_lora:
        raise RuntimeError("K31 BI MoE expert LoRA under CUDA graphs needs cudagraph_specialize_lora=True")


def enabled() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


# Modules that import the MXFP4 oracle functions by name. Compressed-tensors is the K3.1 checkpoint's path (required);
# the others are rebound when this vLLM build has them.
# Kimi-K3.1's MXFP4 experts go through quantization.mxfp4 (Mxfp4MoEMethod: backend_to_kernel_cls for the K3 SiTU
# experts) and compressed-tensors; both are required.
_REQUIRED_QUANT_MODULES = {
    "vllm.model_executor.layers.quantization.mxfp4": ("convert_weight_to_mxfp4_moe_kernel_format", "backend_to_kernel_cls"),
    ("vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe."
     "compressed_tensors_moe_w4a4_mxfp4"): ("convert_weight_to_mxfp4_moe_kernel_format", "select_mxfp4_moe_backend"),
}
_OPTIONAL_QUANT_MODULES = ("vllm.model_executor.layers.quantization.online.mxfp4",
                           "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe",
                           "vllm.model_executor.layers.quantization.quark.quark_moe")


def _bi_experts_class():
    """AiterMxfp4Experts subclass whose apply is bi_moe_forward; LoRA-capable through vLLM's LoRAExpertsMixin."""
    from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import LoRAExpertsMixin
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import AiterMxfp4Experts

    class K31BIMxfp4Experts(LoRAExpertsMixin, AiterMxfp4Experts):
        consumes_expert_mask = False      # get the canonical -1 / local-slot expert_map

        def __init__(self, moe_config, *a, **kw):
            super().__init__(moe_config, *a, **kw)
            if moe_config.is_lora_enabled:
                _check_lora_graph_mode()

        @staticmethod
        def _supports_batch_invariance():
            return True

        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                  expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                  apply_router_weight_on_input):
            if hidden_states.dtype != torch.bfloat16 or a1q_scale is not None:
                raise RuntimeError("K31 BI MoE needs bf16 routed activations: set VLLM_ROCM_USE_AITER_MOE_SITUV2=0")
            if apply_router_weight_on_input:
                raise NotImplementedError("K31 BI MoE: apply_router_weight_on_input")
            mc = self.moe_config
            qc = self.quant_config
            pc = mc.moe_parallel_config
            if pc.dp_size != 1:
                raise NotImplementedError("K31 BI MoE: DP > 1 (token dispatch)")
            gather = reduce = None
            rank = 0
            if expert_map is not None and expert_map.numel() == global_num_experts + 1:
                # AITER-style 0/1 local-expert mask (+ sentinel) -> canonical -1 / local-slot map
                m = expert_map[:global_num_experts] > 0
                expert_map = torch.where(m, torch.cumsum(m.to(torch.int32), 0) - 1,
                                         torch.full_like(m, -1, dtype=torch.int32))
            n_global = None
            if pc.use_ep and pc.ep_size > 1:
                from vllm.distributed import get_ep_group
                grp = get_ep_group()
                rank = pc.ep_rank

                def reduce(t, grp=grp):
                    torch.distributed.all_reduce(t, group=grp.device_group)
                    return t
            elif pc.tp_size > 1:
                from vllm.distributed import get_tp_group
                grp = get_tp_group()
                rank = pc.tp_rank
                n_global = (w1.shape[1] // 2) * pc.tp_size
                unp = getattr(mc, "intermediate_size_per_partition_unpadded", None)
                if unp not in (None, w1.shape[1] // 2):
                    raise RuntimeError(f"K31 BI MoE: padded intermediate {unp} vs {w1.shape[1] // 2}")

                def gather(P, grp=grp, ws=pc.tp_size):
                    outp = torch.empty((ws * P.shape[0],) + tuple(P.shape[1:]), dtype=P.dtype, device=P.device)
                    torch.distributed.all_gather_into_tensor(outp, P.contiguous(), group=grp.device_group)
                    return outp
            lora = _lora_from_context(self._lora_context, hidden_states.shape[0])
            out = bi_moe_forward(hidden_states, w1, qc.w1_scale, w2, qc.w2_scale, topk_weights, topk_ids,
                                 mc.activation_situ_beta, mc.activation_situ_linear_beta,
                                 lora=lora, n_global=n_global, chunk_gather=gather,
                                 expert_map=expert_map if (pc.use_ep and pc.ep_size > 1) else None, leaf_reduce=reduce)
            # identical on every rank of the group, and the runner sums the group: rank 0 returns it, the others 0
            if rank == 0:
                output.copy_(out)
            else:
                output.zero_()

    return K31BIMxfp4Experts


def install():
    """Under VLLM_BATCH_INVARIANT=1: MXFP4 experts stay in raw checkpoint layout (no AITER shuffle) and the MXFP4 oracle
    selects K31BIMxfp4Experts (apply = bi_moe_forward) whatever backend was requested."""
    if not enabled():
        return "off"
    import importlib
    import logging
    O = importlib.import_module("vllm.model_executor.layers.fused_moe.oracle.mxfp4")
    if getattr(O, "_k31_bi_moe", False):
        return "on"
    orig_convert = O.convert_weight_to_mxfp4_moe_kernel_format
    experts_cls = _bi_experts_class()
    log = logging.getLogger("vllm")

    def convert(mxfp4_backend, layer, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale,
                w13_bias=None, w2_bias=None, *a, **kw):
        if mxfp4_backend == O.Mxfp4MoeBackend.AITER_MXFP4_BF16:
            return (w13_weight.data, w2_weight.data, w13_weight_scale.data, w2_weight_scale.data, w13_bias, w2_bias)
        return orig_convert(mxfp4_backend, layer, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale,
                            w13_bias, w2_bias, *a, **kw)

    def select(config, *a, **kw):
        # vLLM's own selection would refuse every MXFP4 backend under BI; make_mxfp4_moe_kernel still validates this class
        return O.Mxfp4MoeBackend.AITER_MXFP4_BF16, experts_cls

    orig_b2k = O.backend_to_kernel_cls

    def backend_to_kernel_cls(backend):
        return [experts_cls] if backend == O.Mxfp4MoeBackend.AITER_MXFP4_BF16 else orig_b2k(backend)

    hooks = {"convert_weight_to_mxfp4_moe_kernel_format": convert, "select_mxfp4_moe_backend": select,
             "select_deepseek_v4_mxfp4_moe_backend": select, "backend_to_kernel_cls": backend_to_kernel_cls}
    for attr, fn in hooks.items():
        if hasattr(O, attr):
            setattr(O, attr, fn)
    rebound = []
    for name in (*_REQUIRED_QUANT_MODULES, *_OPTIONAL_QUANT_MODULES):
        if name in _REQUIRED_QUANT_MODULES:
            m = importlib.import_module(name)
        else:
            try:
                m = importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001  an unused optional module may fail to import; say which
                log.warning("K31 BI MoE: optional module %s not rebound (%s: %s)", name, type(exc).__name__, exc)
                continue
        for attr, fn in hooks.items():
            if hasattr(m, attr):
                setattr(m, attr, fn)
                rebound.append(f"{name.rsplit('.', 1)[-1]}.{attr}")
    for name, attrs in _REQUIRED_QUANT_MODULES.items():
        short = name.rsplit(".", 1)[-1]
        missing = [a for a in attrs if f"{short}.{a}" not in rebound]
        if missing:
            raise RuntimeError(f"K31 BI MoE: {short} no longer imports {missing}; re-derive the hooks")
    O._k31_bi_moe = True
    log.warning("K31 BI MoE installed: %s (rebound: %s). xlora stays off under BI.", experts_cls.__name__,
                ", ".join(rebound))
    return "on"
