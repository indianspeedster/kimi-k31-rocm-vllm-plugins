"""aiter Triton `moe_gemm_a16w4` (tl.dot_scaled, MXFP4 weights read directly) as a drop-in engine for k31lora's BI MoE:
    K31_BI_MOE_ENGINE=a16w4  K31_BI_MOE_ENGINE_FN=k31xlora.a16w4_gemm:a16w4_grouped_gemm

Same signature and contract as k31lora.bi_moe.grouped_gemm for MXFP4 weights:
    out[r] = a[a_rows[r]] @ deq(w[e(r)])^T  (+ add[r])      -> FP32 [R, n_out]
  a      BF16 [*, K], stride(1) == 1 (W2 chunks: a column slice of act, row stride N_local; used as-is, no copy)
  w      uint8 [E, n_out, K/2] raw checkpoint MXFP4 (low nibble = even element); may be a last-dim slice (no copy)
  scale  uint8 [E, n_out, K/32] E8M0
  tiles  (tile_e int32, r0 int32, r1 int32, bm): rows grouped by expert ASCENDING, contiguous from row 0; padding tiles
         (r0 == r1) allowed; rows not covered by a tile come back as 0.
Row-invariant: every aiter heuristic config has split_k = 1 and a K-ordered dot_scaled chain (probe: bs1 == bsN
bitwise, negative control fails). No atomics. No host syncs: the routing tables are built on device from the tiles, so the
function is CUDA-graph safe (the grid is a static upper bound; tiles past the real count map to -1 = no work).
`add` is an FP32 add on the FP32 GEMM output (same single FP32 rounding as an epilogue add)."""
import torch

_A = None


def _aiter():
    global _A
    if _A is None:
        try:
            from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
        except ImportError:
            from aiter.ops.triton.moe_op_gemm_a16w4 import moe_gemm_a16w4
        try:
            from aiter.ops.triton.moe.moe_routing.routing import ExptData, RoutingData
        except ImportError:
            from aiter.ops.triton.moe_routing.routing import ExptData, RoutingData
        _A = (moe_gemm_a16w4, ExptData, RoutingData)
    return _A


def _next_pow2(x: int) -> int:
    return 1 << max(0, int(x) - 1).bit_length()


import os

import triton
import triton.language as tl


@triton.jit
def _route_tiles_kernel(TE, R0, R1, n_tiles_in, n_rows, n_experts, max_tiles, bm,
                        HIST, OFFS_RAW, OFFS_PAD, PID_MAP,
                        EB: tl.constexpr, TB: tl.constexpr, TC: tl.constexpr):
    """One program: the whole _routing_from_tiles table build (integer only -> bitwise equal to the torch version)."""
    offs_e = tl.arange(0, EB)
    emask = offs_e < n_experts
    rmax = tl.zeros((EB,), dtype=tl.int32)
    rmin = tl.zeros((EB,), dtype=tl.int32) + n_rows
    for t0 in range(0, n_tiles_in, TB):
        tt = t0 + tl.arange(0, TB)
        tm = tt < n_tiles_in
        te = tl.load(TE + tt, mask=tm, other=-1).to(tl.int32)
        a0 = tl.load(R0 + tt, mask=tm, other=0).to(tl.int32)
        a1 = tl.load(R1 + tt, mask=tm, other=0).to(tl.int32)
        ne = tm & (a1 > a0)
        match = (te[None, :] == offs_e[:, None]) & ne[None, :]
        rmax = tl.maximum(rmax, tl.max(tl.where(match, a1[None, :], 0), axis=1))
        rmin = tl.minimum(rmin, tl.min(tl.where(match, a0[None, :], n_rows), axis=1))
    hist = tl.where(emask, tl.maximum(rmax - rmin, 0), 0)
    cum = tl.cumsum(hist, axis=0)
    ntile = (hist + bm - 1) // bm
    cpad = tl.cumsum(ntile, axis=0)
    tl.store(HIST + offs_e, hist, mask=emask)
    tl.store(OFFS_RAW + 1 + offs_e, cum, mask=emask)
    tl.store(OFFS_PAD + 1 + offs_e, cpad, mask=emask)
    first = offs_e == 0
    tl.store(OFFS_RAW + offs_e, tl.zeros((EB,), dtype=tl.int32), mask=first)
    tl.store(OFFS_PAD + offs_e, tl.zeros((EB,), dtype=tl.int32), mask=first)
    total = tl.sum(ntile, axis=0)
    for c0 in range(0, max_tiles, TC):
        t = c0 + tl.arange(0, TC)
        # searchsorted(offs_pad[1:], t, right=True) == number of experts whose padded end <= t
        e = tl.sum(((cpad[None, :] <= t[:, None]) & emask[None, :]).to(tl.int32), axis=1)
        e_cl = tl.minimum(e, n_experts - 1)
        start = tl.sum(tl.where(offs_e[None, :] < e_cl[:, None], ntile[None, :], 0), axis=1)   # offs_pad[e_cl]
        b = t - start
        val = tl.where(t < total, (b << 16) + e_cl, -1)
        tl.store(PID_MAP + t, val, mask=t < max_tiles)


_GATE_SCAL = {}


def _routing_from_tiles_fused(tiles, n_rows: int, n_experts: int, device):
    _, ExptData, RoutingData = _aiter()
    tile_e, r0, r1 = tiles[0], tiles[1], tiles[2]
    bm = max(16, min(_next_pow2(max(1, n_rows // max(1, n_experts))), 128))
    if n_rows <= n_experts:
        max_tiles = max(1, n_rows)
    else:
        max_tiles = n_experts - 1 - ((n_experts - n_rows - 1) // bm)
    hist = torch.empty(n_experts, dtype=torch.int32, device=device)
    offs_raw = torch.empty(n_experts + 1, dtype=torch.int32, device=device)
    offs_pad = torch.empty(n_experts + 1, dtype=torch.int32, device=device)
    pid_map = torch.empty(max_tiles, dtype=torch.int32, device=device)
    _route_tiles_kernel[(1,)](tile_e, r0, r1, tile_e.numel(), n_rows, n_experts, max_tiles, bm,
                              hist, offs_raw, offs_pad, pid_map,
                              EB=max(16, _next_pow2(n_experts)), TB=32, TC=16, num_warps=8)
    key = (max(1, n_rows), device)
    gs = _GATE_SCAL.get(key)
    if gs is None:
        gs = _GATE_SCAL[key] = torch.ones(max(1, n_rows), dtype=torch.float32, device=device)  # unused (gammas=None)
    rd = RoutingData(bm, gs, hist, n_experts, 1, ExptData(hist, offs_raw, offs_pad, pid_map))
    return rd, offs_raw[-1]


_FUSED_ROUTE_MAX_E = int(os.environ.get("K31_A16W4_FUSED_ROUTE_MAX_E", "128"))


def _use_fused_route(n_experts: int) -> bool:
    """K31_A16W4_FUSED_ROUTE: auto (default) = fused only for n_experts <= 128 (EP-local layouts; measured 38 -> 6 kernels,
    211 -> 144 us per call at E=112, full-model bitwise). The one-program kernel scans [32 x next_pow2(E)] per tile chunk,
    so TP-wide E=896 keeps the torch path until timed. 1 = always fused, 0 = never."""
    v = os.environ.get("K31_A16W4_FUSED_ROUTE", "auto")
    if v == "1":
        return True
    if v == "0":
        return False
    return n_experts <= _FUSED_ROUTE_MAX_E


def _routing_from_tiles(tiles, n_rows: int, n_experts: int, device):
    """Device-only ExptData for rows already grouped by expert ascending (no sort, no atomics, no host sync)."""
    _, ExptData, RoutingData = _aiter()
    tile_e, r0, r1 = tiles[0], tiles[1], tiles[2]
    # bi_moe's tiles are (expert, tile_start, EXPERT_end): several tiles of one expert share r1, so the expert's row count is
    # max(r1) - min(r0) over its tiles (summing r1 - r0 double-counts multi-tile experts -> out-of-bounds rows).
    # Padding tiles (r0 == r1 == 0) cannot enlarge any expert; experts without tiles get 0.
    # bi_moe's graph padding tiles are (E-1 [clamped], 0, 0): empty tiles must not touch the reductions.
    idx = tile_e.long()
    r0i, r1i = r0.to(torch.int32), r1.to(torch.int32)
    nonempty = r1i > r0i
    rmax = torch.zeros(n_experts, dtype=torch.int32, device=device)
    rmax.scatter_reduce_(0, idx, torch.where(nonempty, r1i, torch.zeros_like(r1i)), reduce="amax", include_self=True)
    rmin = torch.full((n_experts,), n_rows, dtype=torch.int32, device=device)
    rmin.scatter_reduce_(0, idx, torch.where(nonempty, r0i, torch.full_like(r0i, n_rows)), reduce="amin",
                         include_self=True)
    hist = torch.clamp(rmax - rmin, min=0).to(torch.int32)          # integer ops: exact, order-free, no host sync
    # block_m: aiter's heuristic from the static shape (rows per expert), power of 2 in [16, 128]
    bm = max(16, min(_next_pow2(max(1, n_rows // max(1, n_experts))), 128))
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    token_offs_raw = torch.cat((zero, torch.cumsum(hist, 0, dtype=torch.int32)))
    n_tiles = (hist + bm - 1) // bm
    token_offs_pad = torch.cat((zero, torch.cumsum(n_tiles, 0, dtype=torch.int32)))
    # static tile bound (aiter's formula), so the grid and the map size do not depend on the data
    if n_rows <= n_experts:
        max_tiles = max(1, n_rows)
    else:
        max_tiles = n_experts - 1 - ((n_experts - n_rows - 1) // bm)
    t = torch.arange(max_tiles, dtype=torch.int32, device=device)
    e = torch.searchsorted(token_offs_pad[1:].contiguous(), t, right=True).to(torch.int32)
    e_cl = torch.clamp(e, max=n_experts - 1)
    b = t - token_offs_pad[e_cl.long()]
    valid = t < token_offs_pad[-1]
    block_pid_map = torch.where(valid, (b << 16) + e_cl, torch.full_like(t, -1)).to(torch.int32)
    gate_scal = torch.ones(max(1, n_rows), dtype=torch.float32, device=device)   # unused (gammas=None)
    rd = RoutingData(bm, gate_scal, hist, n_experts, 1,
                     ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map))
    return rd, token_offs_raw[-1]


def _zero_uncovered() -> bool:
    """bi_moe decides per forward whether rows not covered by any tile must be exactly 0 (torch fallback combine reads
    every row) or may hold garbage (fused combine16 / combine16_ep mask them at the leaf; the ops in between are row-local).
    Missing flag (older k31lora) or K31_BI_MOE_ZERO_UNCOVERED=1 -> zeros, the original behavior."""
    import os
    if os.environ.get("K31_BI_MOE_ZERO_UNCOVERED", "auto") == "1":
        return True
    try:
        from k31lora import bi_moe
        return bool(bi_moe._ZERO_INIT[0])
    except Exception:
        return True


def a16w4_grouped_gemm(a, a_rows, w, tiles, n_out, scale=None, add=None, shared_w=False, cfg=None):
    assert scale is not None and w.dtype == torch.uint8, "a16w4 engine: MXFP4 weights only"
    assert not shared_w, "a16w4 engine: shared (E=1) MXFP4 weights not supported"
    assert a.dtype == torch.bfloat16 and a.stride(-1) == 1
    moe_gemm_a16w4, _, _ = _aiter()
    R = a_rows.shape[0]
    E = w.shape[0]
    if R == 0:
        return torch.zeros((0, n_out), dtype=torch.float32, device=a.device)
    zu = _zero_uncovered()
    out = torch.zeros((R, n_out), dtype=torch.float32, device=a.device) if zu else None
    rd, n_real = (_routing_from_tiles_fused if _use_fused_route(E) else _routing_from_tiles)(tiles, R, E, a.device)
    # transposed VIEWS (no copy): kernel needs w.stride(-2) == 1 and takes all strides explicitly
    wk = w.transpose(-2, -1)
    sk = scale.transpose(-2, -1)
    y = moe_gemm_a16w4(a, wk, None, sk, None, None, None, rd, gather_indx=a_rows.to(torch.int32).contiguous(),
                       scatter_indx=None, gammas=None, apply_swiglu=False, swizzle_mx_scale=None,
                       out_dtype=torch.float32, backend="triton")
    if zu:
        rows = torch.arange(R, device=a.device)
        out = torch.where((rows < n_real)[:, None], y[:R], out)    # rows not covered by a tile -> 0 (graph padding)
    else:
        out = y[:R]                                                # covered rows exact; uncovered rows never read
    if add is not None:
        out = out + add
    return out
