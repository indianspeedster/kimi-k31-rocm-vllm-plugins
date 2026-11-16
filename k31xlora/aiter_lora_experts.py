# SPDX-License-Identifier: Apache-2.0
"""Unfused AITER-Triton MXFP4 (W4A16) routed experts with LoRA hook points.

Kimi-K3.1 routed experts on gfx950: MXFP4 weights (compressed-tensors
mxfp4-pack, group 32, UE8M0 scales), SITU-GLU activation on concatenated
[gate | up].  The default ROCm path (AiterMxfp4Experts, CK/FlyDSL 2-stage)
fuses the activation, so FusedMoEWithLoRA / LoRAExpertsMixin cannot inject
the w13 / w2 deltas.  This module keeps every stage separate:

    ep_sort_routing(topk_ids, topk_weights, expert_map)   # aiter, no logits
    gemm1  = moe_gemm_a16w4(x, W13, gather_indx, apply_swiglu=False)
             -> [G, 2N] pre-activation, ragged (expert-sorted) rows
    permute ragged -> token-topk [M, topk, 2N]            (LoRA path only)
    apply_w13_lora (punica fused_moe_lora, token-topk order)
    situ_and_mul -> act_tt [G, N]
    permute token-topk -> ragged                           (LoRA path only)
    gemm2  = moe_gemm_a16w4(act, W2, gammas=router weights), no scatter
    permute ragged -> token-topk [M, topk, K]              (LoRA path only)
    apply_w2_lora (mul_routed_weight=True, token-topk order)
    reduce_grouped (fp32 accumulate, skips non-local / invalid slots) -> [M, K]

Template: UnfusedOAITritonExperts (gpt_oss_triton_kernels_moe.py), with the
token-topk ordering fixed for BOTH LoRA hooks (see NOTE below), and
TritonExperts (triton_moe.py) for the LoRA call conventions.

NOTE on LoRA ordering (verified in vllm/lora/ops/triton_ops/fused_moe_lora_op.py
and punica_gpu.add_lora_w13/add_lora_w2): the fused_moe_lora kernels address
rows by the flat token id ``t * topk + slot`` produced by
moe_lora_align_block_size from ``topk_ids``.
  * w13: x = hidden [M, K] (row = flat // topk), y viewed as [M, topk, 2N]
    -> y MUST be in token-topk order.
  * w2 : mul_routed_weight=True => token_mapping_factor=1, x row = flat id
    -> x (activation output) MUST be [M*topk, N] token-topk; y is [M, topk, K].
The triton_kernels "3.8" branch of UnfusedOAITritonExperts passes the RAGGED
``inter`` as w13 y and the RAGGED ``act_out`` as w2 x; both are mis-ordered
(rows land on the wrong (token, slot)).  Do not copy that part.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

__all__ = [
    "AiterTritonMxfp4LoRAExperts",
    "prepare_kimi_mxfp4_moe_weights_for_aiter_a16w4",
    "process_layer_weights_for_aiter_a16w4",
    "aiter_a16w4_operands",
    "build_aiter_routing_from_topk",
    "aiter_a16w4_unfused_moe_forward",
]


# ---------------------------------------------------------------------------
# aiter imports (lazy so the module can be imported on non-ROCm hosts)
# ---------------------------------------------------------------------------
def _aiter():
    try:
        from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
    except ImportError:  # older aiter layout
        from aiter.ops.triton.moe_op_gemm_a16w4 import moe_gemm_a16w4
    from aiter.ops.triton.moe.moe_routing.routing import (
        ExptData,
        RoutingData,
        _compute_expt_data_internal,
        ep_sort_routing,
    )
    from aiter.ops.triton.moe.reduce import reduce_grouped

    return (
        moe_gemm_a16w4,
        ExptData,
        RoutingData,
        _compute_expt_data_internal,
        ep_sort_routing,
        reduce_grouped,
    )


# ---------------------------------------------------------------------------
# Weight processing
# ---------------------------------------------------------------------------
def _as_u8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    if t.element_size() == 1:  # float4_e2m1fn_x2 / float8_e8m0fnu
        return t.view(torch.uint8)
    raise TypeError(f"expected a 1-byte MXFP4 / E8M0 tensor, got {t.dtype}")


def prepare_kimi_mxfp4_moe_weights_for_aiter_a16w4(
    w13_weight: torch.Tensor,  # [E, 2N, K/2] u8, rows = [gate(w1) | up(w3)]
    w13_weight_scale: torch.Tensor,  # [E, 2N, K/32] u8 (UE8M0)
    w2_weight: torch.Tensor,  # [E, K, N/2] u8
    w2_weight_scale: torch.Tensor,  # [E, K, N/32] u8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load-time conversion of raw compressed-tensors Kimi MXFP4 experts.

    Returns STORAGE tensors (keep these on the layer, in the same shapes vLLM
    loaded them); call ``aiter_a16w4_operands`` at run time to get the views
    moe_gemm_a16w4 wants.  What this does / does not do, and why:

    * No gate/up interleave.  vllm oracle/mxfp4.py (AITER_TRITON_MXFP4_BF16
      branch, ``interleave_gate_up``, ~line 1897) interleaves only because
      moe_gemm_a16w4(apply_swiglu=True) reads gate/up as even/odd columns
      (aiter ``swiglu_torch``: a[..., ::2] / a[..., 1::2]).  We run with
      apply_swiglu=False and apply ``situ_and_mul``, which expects the
      concatenated [gate | up] layout FusedMoE.weight_loader already produced.
    * No scale swizzle (swizzle_mx_scale=None).  moe_gemm_a16w4 accepts None
      on gfx950; it then forces the Triton backend (gluon needs CDNA4_SCALE),
      which vLLM forces on gfx950 anyway (aiter_mxfp4_w4a16_moe.py).
    * Storage stays row-major [E, N_out, K_in/2]; the kernel's "column-major"
      requirement (``w.stride(-2) == 1``) is met by a transposed VIEW.  This
      is exactly what vllm mxfp4_utils._swizzle_mxfp4 does on ROCm with
      StridedLayout (``quant_tensor.transpose(-2, -1)``,
      ``scale.transpose(-2, -1)``, no data movement) -- that helper itself
      cannot be called here because it requires triton_kernels.
    """
    E, two_n, k_half = w13_weight.shape
    assert w13_weight_scale.shape == (E, two_n, (2 * k_half) // 32), (
        w13_weight_scale.shape
    )
    assert w2_weight.shape[0] == E and w2_weight.shape[1] == 2 * k_half
    n_half = w2_weight.shape[2]
    assert 4 * n_half == two_n, "w2 in-dim must equal N (= w13 out-dim / 2)"
    assert w2_weight_scale.shape == (E, 2 * k_half, (2 * n_half) // 32)
    for t in (w13_weight, w2_weight):
        assert not getattr(t, "is_shuffled", False), (
            "weights were already shuffled for the CK/FlyDSL path; this "
            "backend needs the raw loaded layout"
        )
    return (
        _as_u8(w13_weight).contiguous(),
        _as_u8(w13_weight_scale).contiguous(),
        _as_u8(w2_weight).contiguous(),
        _as_u8(w2_weight_scale).contiguous(),
    )


def process_layer_weights_for_aiter_a16w4(layer: torch.nn.Module) -> None:
    """In-place variant for a FusedMoE layer loaded by
    CompressedTensorsW4A4Mxfp4MoEMethod (param names ``w13_weight_packed`` /
    ``w2_weight_packed``; falls back to ``w13_weight`` / ``w2_weight``)."""
    w13_name = "w13_weight_packed" if hasattr(layer, "w13_weight_packed") else (
        "w13_weight"
    )
    w2_name = "w2_weight_packed" if hasattr(layer, "w2_weight_packed") else (
        "w2_weight"
    )
    w13, s13, w2, s2 = prepare_kimi_mxfp4_moe_weights_for_aiter_a16w4(
        getattr(layer, w13_name).data,
        layer.w13_weight_scale.data,
        getattr(layer, w2_name).data,
        layer.w2_weight_scale.data,
    )
    for name, t in (
        (w13_name, w13),
        ("w13_weight_scale", s13),
        (w2_name, w2),
        ("w2_weight_scale", s2),
    ):
        setattr(layer, name, torch.nn.Parameter(t, requires_grad=False))


def aiter_a16w4_operands(
    w: torch.Tensor, w_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, Nout, Kin/2] u8, [E, Nout, Kin/32] u8 -> moe_gemm_a16w4 operands.

    moe_gemm_a16w4 takes ``w`` as (E, Kin/2, Nout) with stride(-2) == 1 and
    passes w_scales.stride(1) as the K stride and stride(2) as the N stride
    (aiter _moe_gemm_a16w4 args ``stride_w_mx_k, stride_w_mx_n``), i.e.
    scales are (E, Kin/32, Nout).  Both are zero-copy transposes.
    """
    return _as_u8(w).transpose(-2, -1), _as_u8(w_scale).transpose(-2, -1)


# ---------------------------------------------------------------------------
# Routing from precomputed topk ids / weights (no router logits)
# ---------------------------------------------------------------------------
class AiterRouting:
    __slots__ = (
        "routing_data",
        "gather_indx",
        "scatter_indx",
        "gate_valid",
        "n_live",
        "num_tokens",
        "topk",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def build_aiter_routing_from_topk(
    topk_ids: torch.Tensor,  # [M, topk] GLOBAL ids, -1 = invalid slot
    topk_weights: torch.Tensor,  # [M, topk]
    num_local_experts: int,
    expert_map: torch.Tensor | None,  # [E_global] global->local, -1 non-local
    global_num_experts: int,
    pad_m_pow2: bool = False,
) -> AiterRouting:
    """aiter ``ep_sort_routing`` (routing.py) sorts already-chosen top-k
    (id, weight) pairs by LOCAL expert, sending non-local gates to a sentinel
    bin that gets no GEMM tile.  It is the only aiter entry point that starts
    from topk ids instead of logits.  Outputs (all over G = M * topk gates):
      gather_indx[pos]  = flat gate (t*topk+slot) at sorted row pos (pos < n_live)
      scatter_indx[g]   = sorted row of gate g (0 for dead gates)
      gate_scal[pos]    = fp32 router weight in sorted order
      gate_valid[g]     = 1 if gate g is computed on this rank
    The weights are indexed by LOCAL id, so the GEMMs need no expert_map.
    """
    (_, ExptData, RoutingData, _compute_expt_data_internal, ep_sort_routing, _) = (
        _aiter()
    )
    if pad_m_pow2:
        # aiter's _ep_gate_prep_scan_kernel takes N_CTAS = cdiv(M*topk, 256) as tl.constexpr, so every new
        # 16-token bucket of M recompiles it (~1 s). Pad M to a power of two (>= 16) with invalid (-1) rows: they
        # land in the sentinel bin, are never live, and only ~log2(M) kernel variants exist.
        M0, tk = topk_ids.shape
        Mp = max(16, triton.next_power_of_2(M0))
        if Mp > M0:
            topk_ids = torch.cat(
                (topk_ids, torch.full((Mp - M0, tk), -1, dtype=topk_ids.dtype, device=topk_ids.device)))
            topk_weights = torch.cat(
                (topk_weights, torch.zeros((Mp - M0, tk), dtype=topk_weights.dtype, device=topk_weights.device)))
    M, topk = topk_ids.shape
    G = M * topk
    dev = topk_ids.device

    if expert_map is None:
        emap = torch.arange(num_local_experts, dtype=torch.int32, device=dev)
    else:
        emap = expert_map.to(torch.int32)
    n_map = emap.numel()
    # ep_sort_routing CLAMPS ids to [0, n_map-1] (it does not treat -1 as
    # invalid: a -1 slot would alias global expert 0).  Append a -1 entry and
    # route invalid slots to it.
    emap_ext = torch.cat(
        (emap, torch.full((1,), -1, dtype=torch.int32, device=dev))
    ).contiguous()
    ids = torch.where(topk_ids >= 0, topk_ids, n_map).to(torch.int32).contiguous()
    weights = topk_weights.to(torch.float32).contiguous()

    # aiter routing() block_m heuristic, with tokens/expert counted globally
    # (under EP ~G*E_local/E_global gates land on E_local experts).
    tokens_per_expt = max(1, G // max(1, global_num_experts))
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))

    bufs = _compute_expt_data_internal(num_local_experts, G, block_m, dev)
    hist, gather_indx, scatter_indx, gate_scal, gate_valid, _ = ep_sort_routing(
        weights,
        ids,
        emap_ext,
        num_local_experts,
        None,  # num_local_tokens: no row mask
        M,
        topk,
        G,
        bufs,
    )
    token_offs_raw, token_offs_pad, block_pid_map = bufs[0], bufs[1], bufs[2]
    hist = hist[:num_local_experts]
    routing_data = RoutingData(
        block_m,
        gate_scal,
        hist,
        num_local_experts,
        topk,
        ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map),
    )
    return AiterRouting(
        routing_data=routing_data,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_valid=gate_valid,
        # token_offs_raw[E_local] == number of live (sorted) rows, on device
        n_live=token_offs_raw[num_local_experts : num_local_experts + 1],
        num_tokens=M,
        topk=topk,
    )


# ---------------------------------------------------------------------------
# Kernel pipeline (shared by the experts class and the standalone test)
# ---------------------------------------------------------------------------
ActFn = Callable[[torch.Tensor, torch.Tensor], None]  # (out [G,N], in [G,2N])
W13Hook = Callable[[torch.Tensor], None]  # (y_tt [M, topk, 2N]) in-place add
W2Hook = Callable[[torch.Tensor, torch.Tensor], None]  # (y_tt [M,topk,K], x_tt [G,N])


def aiter_a16w4_unfused_moe_forward(
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13: torch.Tensor,  # [E_local, 2N, K/2] u8 (storage layout)
    w13_scale: torch.Tensor,  # [E_local, 2N, K/32] u8
    w2: torch.Tensor,  # [E_local, K, N/2] u8
    w2_scale: torch.Tensor,  # [E_local, K, N/32] u8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act_fn: ActFn,
    *,
    expert_map: torch.Tensor | None = None,
    global_num_experts: int = -1,
    apply_router_weight_on_input: bool = False,
    w13_hook: W13Hook | None = None,
    w2_hook: W2Hook | None = None,
    output: torch.Tensor | None = None,
    return_debug: bool = False,
):
    moe_gemm_a16w4, *_, reduce_grouped = _aiter()
    assert hidden_states.dtype == torch.bfloat16, (
        "moe_gemm_a16w4 hard-codes bf16 activations in tl.dot_scaled"
    )
    assert hidden_states.dim() == 2
    M, K = hidden_states.shape
    E_local, two_n, _ = w13.shape
    N = two_n // 2
    topk = topk_ids.shape[1]
    G = M * topk
    if global_num_experts <= 0:
        global_num_experts = E_local if expert_map is None else expert_map.numel()
    if output is None:
        output = torch.empty((M, K), dtype=hidden_states.dtype, device=w13.device)
    if M == 0:
        return output

    r = build_aiter_routing_from_topk(
        topk_ids, topk_weights, E_local, expert_map, global_num_experts
    )
    rd = r.routing_data
    gammas = rd.gate_scal  # fp32, sorted order
    w13_k, s13_k = aiter_a16w4_operands(w13, w13_scale)
    w2_k, s2_k = aiter_a16w4_operands(w2, w2_scale)
    common = dict(swizzle_mx_scale=None, out_dtype=hidden_states.dtype,
                  backend="triton")

    # GEMM1: gathered rows, NO activation -> [G, 2N] ragged (expert-sorted).
    # Rows >= n_live are never written (torch.empty) and never read below.
    inter = moe_gemm_a16w4(
        hidden_states, w13_k, None, s13_k, None, None, None, rd,
        gather_indx=r.gather_indx, scatter_indx=None,
        gammas=gammas if apply_router_weight_on_input else None,
        apply_swiglu=False, **common,
    )
    assert inter.shape == (G, two_n), inter.shape

    lora = w13_hook is not None or w2_hook is not None
    dbg = {}
    if not lora:
        act = torch.empty((G, N), dtype=inter.dtype, device=inter.device)
        act_fn(act, inter)
        down = moe_gemm_a16w4(
            act, w2_k, None, s2_k, None, None, None, rd,
            gammas=None if apply_router_weight_on_input else gammas,
            **common,
        )  # [G, K] ragged, router-weighted
        reduce_indx = r.scatter_indx.view(M, topk)
        down3 = down.view(1, G, K)
        if return_debug:
            dbg["inter_tt"] = inter.index_select(0, r.scatter_indx.long())
    else:
        sc = r.scatter_indx.long()  # dead gates -> 0 (a live row; masked later)
        inter_tt = inter.index_select(0, sc).view(M, topk, two_n)
        if w13_hook is not None:
            w13_hook(inter_tt)
        if return_debug:
            dbg["inter_tt"] = inter_tt.view(G, two_n).clone()
        act_tt = torch.empty((G, N), dtype=inter.dtype, device=inter.device)
        act_fn(act_tt, inter_tt.view(G, two_n))
        # back to ragged order for GEMM2; gather_indx is uninitialised past
        # n_live, so clamp those positions to 0 before indexing.
        pos = torch.arange(G, device=inter.device, dtype=torch.int32)
        gsafe = torch.where(pos < r.n_live, r.gather_indx, 0).long()
        act_sorted = act_tt.index_select(0, gsafe)
        down = moe_gemm_a16w4(
            act_sorted, w2_k, None, s2_k, None, None, None, rd,
            gammas=None if apply_router_weight_on_input else gammas,
            **common,
        )
        down_tt = down.index_select(0, sc).view(M, topk, K)
        if w2_hook is not None:
            w2_hook(down_tt, act_tt)
        reduce_indx = pos.view(M, topk)
        down3 = down_tt.view(1, G, K)

    # fp32-accumulating top-k combine that skips dead gates (NaN-safe).
    reduce_grouped(
        down3, reduce_indx, output,
        out_dtype=output.dtype,
        indx_valid=r.gate_valid.view(M, topk),
    )
    if return_debug:
        dbg["gate_valid"] = r.gate_valid.view(M, topk)
        return output, dbg
    return output


def aiter_a16w4_compact_lora_forward(
    experts,  # the LoRAExpertsMixin instance (for apply_w13_lora / apply_w2_lora)
    ctx,  # MoELoRAContext
    lora_x: torch.Tensor,  # [M, K] LoRA input (unquantized hidden states)
    hidden_states: torch.Tensor,  # [M, K] bf16
    w13: torch.Tensor, w13_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, act_fn: ActFn, *,
    expert_map: torch.Tensor | None, global_num_experts: int, output: torch.Tensor,
    w1_ref: torch.Tensor, w2_ref: torch.Tensor, map_override: torch.Tensor | None = None,
):
    """Large-M (prefill, eager) variant: only LIVE gates are materialised, in aiter's expert-sorted order, and each live
    gate is presented to the punica MoE-LoRA kernels as its own pseudo-token with top_k = 1 (they address rows as
    token*top_k + slot, so [n_live, 1, .] in sorted order is a valid layout). No [G, .] reorders, no dead EP rows.
    Costs one host sync (n_live). Must not run under CUDA-graph capture."""
    import dataclasses

    moe_gemm_a16w4, *_ = _aiter()
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    E_local, two_n, _ = w13.shape
    N = two_n // 2
    r = build_aiter_routing_from_topk(topk_ids, topk_weights, E_local, expert_map, global_num_experts,
                                      pad_m_pow2=os.environ.get("XLORA_PAD_M", "1") == "1")
    rd = r.routing_data
    n_live = int(r.n_live.item())
    if n_live == 0:
        output.zero_()
        return output
    gidx = r.gather_indx[:n_live].contiguous()          # flat gate t*topk+slot per sorted row
    gi = gidx.long()
    tok = gi // topk
    w13_k, s13_k = aiter_a16w4_operands(w13, w13_scale)
    w2_k, s2_k = aiter_a16w4_operands(w2, w2_scale)
    common = dict(swizzle_mx_scale=None, out_dtype=hidden_states.dtype, backend="triton")
    gam = rd.gate_scal[:n_live]

    inter = moe_gemm_a16w4(hidden_states, w13_k, None, s13_k, None, None, None, rd, gather_indx=gidx,
                           scatter_indx=None, gammas=None, apply_swiglu=False, **common)   # [n_live, 2N]
    ids_p = topk_ids.reshape(-1)[gi].view(n_live, 1).contiguous()
    w_p = topk_weights.reshape(-1)[gi].view(n_live, 1).contiguous()
    base_map = map_override if map_override is not None else ctx.local_token_lora_mapping
    if base_map is None:
        base_map = ctx.punica_wrapper.token_mapping_meta.meta_args(
            M, ctx.punica_wrapper.lora_config.specialize_active_lora)[0]
    ctx_p = dataclasses.replace(ctx, top_k=1, local_token_lora_mapping=base_map[:M][tok].contiguous())
    s_, e_, n_, t_ = experts.apply_w13_lora(
        ctx_p, y=inter.view(n_live, 1, two_n), x=lora_x[tok].contiguous(), topk_ids=ids_p, topk_weights=w_p,
        expert_map=expert_map, w1=w1_ref, w2=w2_ref, num_tokens=n_live, top_k_num=1)
    act = torch.empty((n_live, N), dtype=inter.dtype, device=inter.device)
    act_fn(act, inter)
    down = moe_gemm_a16w4(act, w2_k, None, s2_k, None, None, None, rd, gammas=gam, **common)   # [n_live, K]
    experts.apply_w2_lora(
        ctx_p, y=down.view(n_live, 1, K), x=act, topk_weights=w_p, sorted_token_ids_lora=s_, expert_ids_lora=e_,
        num_tokens_post_padded_lora=n_, token_lora_mapping=t_, num_tokens=n_live, w1=w1_ref, w2=w2_ref, top_k_num=1)
    acc = torch.zeros((M, K), dtype=torch.float32, device=hidden_states.device)
    acc.index_add_(0, tok, down.float())
    output.copy_(acc.to(output.dtype))
    return output


# ---------------------------------------------------------------------------
# Modular experts class
# ---------------------------------------------------------------------------
class AiterTritonMxfp4LoRAExperts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Standard-format MXFP4 W4A16 experts on aiter moe_gemm_a16w4 with LoRA.

    Expects layer weights in the raw loaded layout produced by
    ``prepare_kimi_mxfp4_moe_weights_for_aiter_a16w4`` (w1 = w13 [E,2N,K/2]
    u8, w2 [E,K,N/2] u8) and quant_config.w1_scale / w2_scale = the raw u8
    UE8M0 scale tensors.  Default moe_problem_size works on that layout.
    """

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return False
        try:
            _aiter()
        except ImportError:
            return False
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Any gated [gate | up] activation that apply_moe_activation handles.
        return activation in (MoEActivation.SITU, MoEActivation.SILU)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Router weights + top-k sum are done inside apply().
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # moe_gemm_a16w4 allocates its own outputs; intermediates are
        # allocated in apply() (output may alias workspace13, so never use it).
        return (0,), (0,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is None, "W4A16: activations must arrive unquantized"
        w1_scale, w2_scale = self.w1_scale, self.w2_scale
        assert w1_scale is not None and w2_scale is not None
        M = hidden_states.shape[0]
        topk = topk_ids.shape[1]

        def act_fn(out: torch.Tensor, inp: torch.Tensor) -> None:
            self.activation(activation, out, inp)

        w13_hook = w2_hook = None
        ctx = self._lora_context
        if ctx is not None:
            # Same LoRA-input choice as TritonExperts.apply.
            if (
                ctx.original_hidden_states is not None
                and ctx.original_hidden_states.shape[0] == M
            ):
                lora_x = ctx.original_hidden_states
            else:
                lora_x = hidden_states
            meta: dict[str, torch.Tensor | None] = {}

            def w13_hook(y_tt: torch.Tensor) -> None:
                (
                    meta["s"],
                    meta["e"],
                    meta["n"],
                    meta["t"],
                ) = self.apply_w13_lora(
                    ctx,
                    y=y_tt,  # [M, topk, 2N] token-topk order
                    x=lora_x,  # [M, K]
                    topk_ids=topk_ids,  # GLOBAL ids (kernel remaps w/ map)
                    topk_weights=topk_weights,
                    expert_map=expert_map,
                    w1=w1,
                    w2=w2,
                    num_tokens=M,
                    top_k_num=topk,
                )

            def w2_hook(y_tt: torch.Tensor, x_tt: torch.Tensor) -> None:
                self.apply_w2_lora(
                    ctx,
                    y=y_tt,  # [M, topk, K], already router-weighted
                    x=x_tt,  # [M*topk, N] token-topk order
                    topk_weights=topk_weights,  # mul_routed_weight=True
                    sorted_token_ids_lora=meta["s"],
                    expert_ids_lora=meta["e"],
                    num_tokens_post_padded_lora=meta["n"],
                    token_lora_mapping=meta["t"],
                    num_tokens=M,
                    w1=w1,
                    w2=w2,
                    top_k_num=topk,
                )

        import os as _os
        if (
            ctx is not None
            and _os.environ.get("XLORA_COMPACT", "1") == "1"   # default on; 0 = old path
            and not apply_router_weight_on_input
            and M * topk >= int(_os.environ.get("XLORA_COMPACT_MIN_G", "4096"))
            and not torch.cuda.is_current_stream_capturing()
        ):
            aiter_a16w4_compact_lora_forward(
                self, ctx, lora_x, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids, act_fn,
                expert_map=expert_map, global_num_experts=global_num_experts, output=output.view(M, -1),
                w1_ref=w1, w2_ref=w2,
            )
            return
        aiter_a16w4_unfused_moe_forward(
            hidden_states,
            w1,
            w1_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            act_fn,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            w13_hook=w13_hook,
            w2_hook=w2_hook,
            output=output.view(M, -1),
        )
