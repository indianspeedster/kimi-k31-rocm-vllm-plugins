"""Batch-invariant DSA indexer for Kimi-K3.1 on ROCm (VLLM_BATCH_INVARIANT=1).

Covers the uncompressed FP8 index cache in the block-flat NORMAL layout, without DCP or candidate blocks. A row's
logits come from one fixed kernel configuration in prefill chunks and decode steps alike, and its top-k from one
deterministic kernel, so the selected indices (set and order) depend only on the row. Other indexer calls go to
vLLM's own implementation.
"""
import os
import sys

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as _ops
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    FP8_DTYPE,
    _ON_GFX950,
    cp_gather_indexer_k_quant_cache_triton,
    indexer_k_quant_and_cache_triton,
    mqa_logits_module,
)
from vllm.v1.worker.workspace import current_workspace_manager


def _use_bi_indexer(
    compress_ratio: int,
    candidate_blocks: torch.Tensor | None,
    cache_layout: str | None,
    dcp_world_size: int,
) -> bool:
    return (
        envs.VLLM_BATCH_INVARIANT
        and compress_ratio == 1
        and candidate_blocks is None
        and cache_layout == "NORMAL"
        and dcp_world_size == 1
    )


@triton.jit
def _ordered_u32(x):
    # fp32 bits -> uint32 with the order of the float values (negative: all
    # bits flipped; non-negative: sign bit set).
    bits = x.to(tl.int32, bitcast=True)
    u = bits.to(tl.uint32, bitcast=True)
    return tl.where(bits < 0, ~u, u | tl.full((), 0x80000000, tl.uint32))


@triton.jit
def _det_topk_digit(u, m, thr, shift, DBITS: tl.constexpr):
    # Largest digit d (DBITS bits at ``shift``) with #(u >= thr | d << shift)
    # >= sel; the counts fall as d grows, so d = #(digits passing) - 1.
    d = tl.arange(0, 1 << DBITS).to(tl.uint32)
    cand = thr | (d << shift)
    ge = m[:, None] & (u[:, None] >= cand[None, :])
    return tl.sum(ge.to(tl.int32), axis=0)


@triton.jit
def _dsv4_det_topk_kernel(
    logits_ptr,
    stride0,
    stride1,
    row_starts_ptr,
    row_ends_ptr,
    out_ptr,
    out_stride,
    n_cols,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    RES: tl.constexpr,
    DBITS: tl.constexpr,
):
    """Exact, fully specified top-k of row ``logits[row, row_start:row_end]``.

    Order: by score (fp32 total order of the bit patterns), ties to the larger
    column. Output: the selected row-local columns ascending, then -1 to
    width TOPK. One program per row; every decision depends on the row only.

    The cutoff key T (the sel-th largest key) is found by an MSB-first radix
    select, DBITS bits per round; rows of at most RES columns are loaded once
    and selected in registers, longer rows are streamed in BLOCK tiles each
    round. T is unique, so the selected set and its order do not depend on
    DBITS, RES or BLOCK.
    """
    row = tl.program_id(0)
    rs = tl.load(row_starts_ptr + row)
    start = tl.maximum(rs, 0)
    end = tl.minimum(tl.load(row_ends_ptr + row), n_cols)
    nv = tl.maximum(end - start, 0)
    sel = tl.minimum(nv, TOPK)
    row_ptr = logits_ptr + row.to(tl.int64) * stride0
    out_row = out_ptr + row.to(tl.int64) * out_stride

    if nv > TOPK:
        if nv <= RES:
            cols = start + tl.arange(0, RES)
            m = cols < end
            u = _ordered_u32(tl.load(row_ptr + cols * stride1, mask=m, other=0.0))
            thr = tl.zeros((), dtype=tl.uint32)
            for r in tl.static_range(32 // DBITS):
                shift = 32 - DBITS * (r + 1)
                cnt = _det_topk_digit(u, m, thr, shift, DBITS)
                nd = tl.sum((cnt >= sel).to(tl.int32), axis=0)
                thr = thr | ((nd - 1).to(tl.uint32) << shift)
            gt = m & (u > thr)
            eq = m & (u == thr)
            n_gt = tl.sum(gt.to(tl.int32), axis=0)
            n_eq = tl.sum(eq.to(tl.int32), axis=0)
            skip_eq = n_eq - (sel - n_gt)
            eq_rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
            take = gt | (eq & (eq_rank >= skip_eq))
            pos = tl.cumsum(take.to(tl.int32), axis=0) - 1
            tl.store(out_row + pos, cols - rs, mask=take)
        else:
            thr = tl.zeros((), dtype=tl.uint32)
            for r in tl.static_range(32 // DBITS):
                shift = 32 - DBITS * (r + 1)
                cnt = tl.zeros((1 << DBITS,), dtype=tl.int32)
                for c0 in range(start, end, BLOCK):
                    cols = c0 + tl.arange(0, BLOCK)
                    m = cols < end
                    u = _ordered_u32(
                        tl.load(row_ptr + cols * stride1, mask=m, other=0.0)
                    )
                    cnt += _det_topk_digit(u, m, thr, shift, DBITS)
                nd = tl.sum((cnt >= sel).to(tl.int32), axis=0)
                thr = thr | ((nd - 1).to(tl.uint32) << shift)
            n_gt = tl.zeros((), dtype=tl.int32)
            n_eq = tl.zeros((), dtype=tl.int32)
            for c0 in range(start, end, BLOCK):
                cols = c0 + tl.arange(0, BLOCK)
                m = cols < end
                u = _ordered_u32(tl.load(row_ptr + cols * stride1, mask=m, other=0.0))
                n_gt += tl.sum((m & (u > thr)).to(tl.int32), axis=0)
                n_eq += tl.sum((m & (u == thr)).to(tl.int32), axis=0)
            # Of the n_eq tied columns take the (sel - n_gt) largest.
            skip_eq = n_eq - (sel - n_gt)
            eq_seen = tl.zeros((), dtype=tl.int32)
            written = tl.zeros((), dtype=tl.int32)
            for c0 in range(start, end, BLOCK):
                cols = c0 + tl.arange(0, BLOCK)
                m = cols < end
                u = _ordered_u32(tl.load(row_ptr + cols * stride1, mask=m, other=0.0))
                eq = m & (u == thr)
                eq_rank = eq_seen + tl.cumsum(eq.to(tl.int32), axis=0) - 1
                take = (m & (u > thr)) | (eq & (eq_rank >= skip_eq))
                pos = written + tl.cumsum(take.to(tl.int32), axis=0) - 1
                tl.store(out_row + pos, cols - rs, mask=take)
                eq_seen += tl.sum(eq.to(tl.int32), axis=0)
                written += tl.sum(take.to(tl.int32), axis=0)
    else:
        for c0 in range(0, sel, BLOCK):
            j = c0 + tl.arange(0, BLOCK)
            tl.store(out_row + j, start + j - rs, mask=j < sel)
    for c0 in range(0, TOPK, BLOCK):
        j = c0 + tl.arange(0, BLOCK)
        tl.store(out_row + j, -1, mask=(j >= sel) & (j < TOPK))


# Launch configuration of _dsv4_det_topk_kernel. Performance only: the
# selected indices and their order do not depend on it.
_DET_TOPK_CFG = {"BLOCK": 2048, "RES": 4096, "DBITS": 4, "num_warps": 16}


def dsv4_det_topk_per_row(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    num_rows: int,
    topk_tokens: int,
) -> None:
    """Deterministic indexer top-k (VLLM_BATCH_INVARIANT=1), for prefill
    chunks and decode rows alike.

    Selects, per row, the ``topk_tokens`` columns of
    ``logits[row, row_starts[row]:row_ends[row]]`` with the largest scores,
    ties going to the larger column, and writes their row-local indices in
    ascending order followed by -1 padding. The result depends only on the
    row's own scores, so it does not change with the batch, the launch shape
    or the row's position.
    """
    if num_rows == 0:
        return
    assert topk_indices.shape[-1] == topk_tokens
    _dsv4_det_topk_kernel[(num_rows,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        row_starts,
        row_ends,
        topk_indices,
        topk_indices.stride(0),
        logits.shape[1],
        TOPK=topk_tokens,
        BLOCK=_DET_TOPK_CFG["BLOCK"],
        RES=_DET_TOPK_CFG["RES"],
        DBITS=_DET_TOPK_CFG["DBITS"],
        num_warps=_DET_TOPK_CFG["num_warps"],
    )


def _bi_fp8_mqa_logits_into(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    logits: torch.Tensor,
    stride_logits_s: int,
) -> bool:
    """AITER's gfx950 Gluon fp8_mqa_logits kernel at one fixed configuration.

    The fp8_mqa_logits wrapper picks BLOCK_M (one or two query rows per
    program), and with it NUM_CHAINS (the head-sum fold) and BLOCK_KV /
    num_warps, from the number of query rows, so a row's logits depend on its
    chunk. This launch always takes the wrapper's BLOCK_M = 1 configuration
    (2 warps, BLOCK_KV 64, its MFMA shape and head chunking, which follow only
    the model's head count and head size). One program owns one row, walks
    [ks, ke) in BLOCK_KV tiles aligned at ks and stores only inside it, so a
    logit depends on its row's q, weights and key only, not on the batch, the
    row's position or ``stride_logits_s`` (0 packs rows with disjoint ranges
    into one buffer). KV splits are tile aligned and only partition the walk.
    Returns False when that kernel is not available.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    if not (_ON_GFX950 and rocm_aiter_ops.is_enabled()):
        return False
    module = mqa_logits_module()
    if module is None or not hasattr(module, "fp8_mqa_logits"):
        return False
    m = sys.modules[module.fp8_mqa_logits.__module__]
    kernel = getattr(m, "_gluon_fp8_mqa_logits_kernel", None)
    if (
        kernel is None
        or not getattr(m, "TRITON_GE_36", False)
        or getattr(m, "arch", None) != "gfx950"
    ):
        return False
    seq_len, num_heads, head_size = q.shape
    if seq_len == 0:
        return True
    num_warps = 2
    waves_per_eu = 2 if getattr(m, "TRITON_GE_38", False) else 3
    kv_splits_fn = getattr(m, "_gfx950_kv_splits", None)
    num_kv_splits = (
        kv_splits_fn(seq_len, k.shape[0], 1, num_warps, waves_per_eu)
        if kv_splits_fn is not None
        else 1
    )
    mfma_nonk_dim = 32 if (head_size <= 64 or num_heads >= 32) else 16
    num_chains = (
        1 if getattr(m, "FOLDED_REDUCTED_SUPPORT", False) and num_heads > 16 else 0
    )
    m_chunk = (
        mfma_nonk_dim
        if num_chains >= 1 and num_heads > mfma_nonk_dim and mfma_nonk_dim == 32
        else 0
    )
    kernel[(seq_len, num_kv_splits)](
        Q_ptr=q,
        KV_ptr=k,
        kv_scales_ptr=k_scale,
        weights_ptr=weights,
        cu_start_ptr=ks,
        cu_end_ptr=ke,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=k.shape[0],
        num_kv_splits=num_kv_splits,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_kv_s=k.stride(0),
        stride_kv_d=k.stride(1),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_logits_s=stride_logits_s,
        stride_logits_k=1,
        BLOCK_KV=64,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=2,
        NUM_CHAINS=num_chains,
        USE_BUFFER_LOAD=True,
        USE_BUFFER_STORE=True,
        USE_PADDED_SHARED_LAYOUT=getattr(m, "ASYNC_COPY_SUPPORTS_DISTRIBUTED", False),
        BLOCK_M=1,
        MFMA_NONK_DIM=mfma_nonk_dim,
        M_CHUNK=m_chunk,
        UNROLL=2,
        RELAXED_STORE=0,
        HAS_KV_SPLIT=1 if num_kv_splits > 1 else 0,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        enable_fp_fusion=True,
    )
    return True


def _bi_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """rocm_fp8_mqa_logits for batch-invariant prefill chunks: the logits of
    _bi_fp8_mqa_logits_into, those the decode path computes. Positions outside
    a row's [ks, ke) are unspecified, as with clean_logits=False."""
    k_fp8, k_scale = kv
    n = k_fp8.shape[0]
    logits = torch.empty(
        (q.shape[0], triton.cdiv(n, 256) * 256), dtype=torch.float32, device=q.device
    )[:, :n]
    if _bi_fp8_mqa_logits_into(
        q, k_fp8, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke, logits, logits.stride(0)
    ):
        return logits
    raise RuntimeError(
        "VLLM_BATCH_INVARIANT=1: the batch-invariant DSA indexer needs AITER's gfx950 Gluon fp8_mqa_logits kernel"
    )


# Batch-invariant decode indexer workspace, in gathered index-cache tokens:
# requests are laid out at a fixed per-request capacity (max_model_len), so a
# CUDA graph capture and every replay take the same launches. The budget is
# max_num_seqs x capacity, capped (VLLM_K31_BI_DSA_DECODE_WS_TOKENS, default 4M
# tokens = ~570 MB) and never below the prefill gather workspace; rows beyond it
# run in further groups. Set once by the profiling pass (before any capture).
_BI_DECODE_WS_TOKENS: dict[tuple[int, int], int] = {}


def _bi_decode_ws_tokens(total_seq_lens: int, capacity: int) -> int:
    key = (total_seq_lens, capacity)
    if key not in _BI_DECODE_WS_TOKENS:
        try:
            from vllm.config import get_current_vllm_config

            max_seqs = int(get_current_vllm_config().scheduler_config.max_num_seqs)
        except Exception:
            max_seqs = 1
        cap = int(os.environ.get("VLLM_K31_BI_DSA_DECODE_WS_TOKENS", str(1 << 22)))
        _BI_DECODE_WS_TOKENS[key] = max(
            total_seq_lens, min(max_seqs * capacity, max(cap, capacity))
        )
    return _BI_DECODE_WS_TOKENS[key]


def _bi_decode_indexer_workspace(
    num_tokens: int, head_dim: int, num_reqs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Workspace views (K fp8, K scales, logits, row bounds) of the
    batch-invariant decode indexer: ``num_tokens`` gathered index-cache tokens
    and ``num_reqs`` request rows."""
    return current_workspace_manager().get_simultaneous(
        ((num_tokens, head_dim), FP8_DTYPE),
        ((num_tokens,), torch.float32),
        ((num_tokens,), torch.float32),
        ((3 * num_reqs,), torch.int32),
    )


@triton.jit
def _bi_decode_indexer_k_gather_kernel(
    k_cache_ptr,
    block_table_ptr,
    block_table_stride,
    seq_lens_ptr,
    k_out_ptr,
    scale_out_ptr,
    row_bounds_ptr,
    num_reqs,
    capacity,
    cache_block_stride,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Copies the first seq_len[b] tokens (fp8 bytes + fp32 scale bits) of
    # request b from the NORMAL-layout indexer cache to rows
    # [b * capacity, b * capacity + seq_len[b]), the bytes
    # _cp_gather_indexer_quant_cache_kernel copies for a prefill, and writes
    # row_bounds = [starts (b * capacity) | ends | zeros] for the logits and
    # top-k kernels.
    b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + b)
    if tl.program_id(1) == 0:
        tl.store(row_bounds_ptr + b, b * capacity)
        tl.store(row_bounds_ptr + num_reqs + b, b * capacity + seq_len)
        tl.store(row_bounds_ptr + 2 * num_reqs + b, 0)
    t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t < seq_len
    block = tl.load(
        block_table_ptr + b * block_table_stride + t // CACHE_BLOCK_SIZE,
        mask=mask,
        other=0,
    ).to(tl.int64)
    off = t % CACHE_BLOCK_SIZE
    src = k_cache_ptr + block * cache_block_stride
    d = tl.arange(0, HEAD_DIM)
    m2 = mask[:, None]
    vals = tl.load(src[:, None] + off[:, None] * HEAD_DIM + d[None, :], mask=m2)
    row = b.to(tl.int64) * capacity + t
    tl.store(k_out_ptr + row[:, None] * HEAD_DIM + d[None, :], vals, mask=m2)
    scale_src = (src + CACHE_BLOCK_SIZE * HEAD_DIM).to(tl.pointer_type(tl.int32))
    scale = tl.load(scale_src + off, mask=mask)
    tl.store(scale_out_ptr + row, scale, mask=mask)


def _bi_decode_indexer_topk(
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    max_model_len: int,
    topk_tokens: int,
    workspace_tokens: int,
) -> None:
    """Decode indexer on the prefill score + top-k kernels (DSv4 sparse decode).

    Request b's keys are gathered to rows ``[b * C, b * C + seq_len_b)`` of a
    request-major workspace, its logits are written to the same positions of
    a ``[B * C]`` buffer (_bi_fp8_mqa_logits_into with row stride 0, so no
    ``B x B * C`` matrix) and dsv4_det_topk_per_row selects on the ``[B, C]``
    view with row-local columns. The arithmetic is that of the same token
    scored in a prefill chunk, so the indices are too. Rows are processed in
    groups that fit the prefill gather workspace (``workspace_tokens``).

    Args:
        kv_cache: ``[num_blocks, block_size, head_dim + 4]`` uint8 index
            cache, block-flat NORMAL layout.
        q_fp8: ``[B, H, D]`` fp8 decode queries (one token per request).
        weights: ``[B, H]`` fp32 head weights.
        seq_lens: ``[B]`` (or ``[B, 1]``) context length of each decode token.
        block_table: ``[B, max_blocks]`` index-cache block table.
        topk_indices: ``[B, topk_tokens]`` int32 output (request-local, -1
            padded).
        max_model_len: fixed per-request capacity C.
        workspace_tokens: static workspace size (_bi_decode_ws_tokens).

    CUDA-graph safe (FULL decode graphs run this inside the capture): every
    shape, group and launch grid follows from ``B``, ``C`` and
    ``workspace_tokens`` only, nothing is read from the host per step, and
    eager steps take the same launches. Rows with ``seq_len == 0`` (vLLM's
    padding to the capture size) gather nothing, score nothing and get -1
    indices. A row's values do not depend on C or its group: the logits kernel
    walks only ``[ks, ke)`` and the top-k reads only its row.
    """
    num_reqs, _, head_dim = q_fp8.shape
    if num_reqs == 0:
        return
    seq_lens = seq_lens.reshape(-1)[:num_reqs].to(torch.int32)
    capacity = max_model_len
    assert capacity <= workspace_tokens
    group = max(1, min(num_reqs, workspace_tokens // capacity))
    block_t = 32
    for lo in range(0, num_reqs, group):
        hi = min(lo + group, num_reqs)
        n = hi - lo
        k_fp8, k_scale, logits, row_bounds = _bi_decode_indexer_workspace(
            n * capacity, head_dim, n
        )
        ks, ke, row_starts = row_bounds.view(3, n)
        _bi_decode_indexer_k_gather_kernel[(n, triton.cdiv(capacity, block_t))](
            kv_cache,
            block_table[lo:hi],
            block_table.stride(0),
            seq_lens[lo:hi],
            k_fp8.view(torch.uint8),
            k_scale.view(torch.int32),
            row_bounds,
            n,
            capacity,
            kv_cache.stride(0),
            CACHE_BLOCK_SIZE=kv_cache.shape[1],
            HEAD_DIM=head_dim,
            BLOCK_T=block_t,
        )
        if not _bi_fp8_mqa_logits_into(q_fp8[lo:hi], k_fp8, k_scale, weights[lo:hi], ks, ke, logits, 0):
            raise RuntimeError(
                "VLLM_BATCH_INVARIANT=1: the batch-invariant DSA indexer needs AITER's gfx950 Gluon fp8_mqa_logits kernel"
            )
        logits_2d = logits.view(n, capacity)
        dsv4_det_topk_per_row(
            logits_2d, row_starts, seq_lens[lo:hi], topk_indices[lo:hi], n, topk_tokens
        )


def bi_sparse_attn_indexer(
    hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights, quant_block_size, scale_fmt, topk_tokens, head_dim,
    max_model_len, total_seq_lens, topk_indices_buffer, skip_k_cache_insert=False, compress_ratio=1,
    candidate_blocks=None, candidate_block_size=0, candidate_write=False, cache_layout=None, dcp_rank=0,
    dcp_world_size=1, cp_interleave=1,
):
    """vllm::rocm_aiter_sparse_attn_indexer with the batch-invariant path for the case _use_bi_indexer covers."""
    args = (hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights, quant_block_size, scale_fmt, topk_tokens,
            head_dim, max_model_len, total_seq_lens, topk_indices_buffer)
    kwargs = dict(skip_k_cache_insert=skip_k_cache_insert, compress_ratio=compress_ratio,
                  candidate_blocks=candidate_blocks, candidate_block_size=candidate_block_size,
                  candidate_write=candidate_write, cache_layout=cache_layout, dcp_rank=dcp_rank,
                  dcp_world_size=dcp_world_size, cp_interleave=cp_interleave)
    resolved_layout = cache_layout
    if resolved_layout is None and _ops._indexer_k_is_c4a_block_flat(compress_ratio):
        resolved_layout = "NORMAL"
    if not _use_bi_indexer(compress_ratio, candidate_blocks, resolved_layout, dcp_world_size):
        return _ORIG_INDEXER(*args, **kwargs)

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        # Profiling pass: vLLM reserves its workspaces; also reserve the static decode workspace.
        out = _ORIG_INDEXER(*args, **kwargs)
        ws = _bi_decode_ws_tokens(total_seq_lens, max_model_len)
        _bi_decode_indexer_workspace(ws, head_dim, ws // max_model_len + 1)
        return out

    meta = attn_metadata[_resolve_layer_name(k_cache_prefix)]
    assert isinstance(meta, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None and scale_fmt is not None
    slot_mapping = meta.slot_mapping
    num_decode_tokens = meta.num_decode_tokens
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if k is not None:
        k = k[: slot_mapping.shape[0]]
    elif not skip_k_cache_insert:
        raise ValueError("k must be provided when skip_k_cache_insert is False")
    if not skip_k_cache_insert:
        indexer_k_quant_and_cache_triton(k, kv_cache, slot_mapping, quant_block_size, scale_fmt,
                                         cache_layout=cache_layout)

    if meta.num_prefills > 0:
        assert meta.prefill is not None
        k_fp8_full, k_scale_full = current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), FP8_DTYPE),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in meta.prefill.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            cp_gather_indexer_k_quant_cache_triton(kv_cache, k_fp8, k_scale, chunk.block_table, chunk.cu_seq_lens,
                                                   token_to_seq=chunk.token_to_seq, cache_layout=resolved_layout)
            logits = _bi_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            dsv4_det_topk_per_row(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices_buffer[chunk.token_start : chunk.token_end, :topk_tokens],
                logits.shape[0],
                topk_tokens,
            )

    if meta.num_decodes > 0:
        decode = meta.decode
        assert decode is not None
        if decode.requires_padding:
            raise NotImplementedError(
                "VLLM_BATCH_INVARIANT=1: the ROCm DSA decode indexer supports one token per decode request "
                "(no speculative decoding)"
            )
        # One row per decode token; under a CUDA graph the padded rows carry seq_len 0 (no-ops).
        assert decode.seq_lens.numel() >= num_decode_tokens
        _bi_decode_indexer_topk(
            kv_cache,
            q_fp8[:num_decode_tokens],
            weights[:num_decode_tokens],
            decode.seq_lens,
            decode.block_table,
            topk_indices_buffer[:num_decode_tokens, :topk_tokens],
            max_model_len,
            topk_tokens,
            _bi_decode_ws_tokens(total_seq_lens, max_model_len),
        )
    return topk_indices_buffer


_ORIG_INDEXER = None
_LIB = None


def install() -> None:
    """Route vllm::rocm_aiter_sparse_attn_indexer through bi_sparse_attn_indexer (same op, same schema)."""
    global _ORIG_INDEXER, _LIB
    if _LIB is not None:
        return
    import vllm._aiter_ops  # noqa: F401  registers the op
    from vllm.platforms import current_platform

    _ORIG_INDEXER = _ops.rocm_aiter_sparse_attn_indexer
    if not hasattr(torch.ops.vllm, "rocm_aiter_sparse_attn_indexer"):
        raise RuntimeError("K31 BI DSA: vllm::rocm_aiter_sparse_attn_indexer is not registered")
    _LIB = torch.library.Library("vllm", "IMPL")
    _LIB.impl("rocm_aiter_sparse_attn_indexer", bi_sparse_attn_indexer, current_platform.dispatch_key,
              allow_override=True)
