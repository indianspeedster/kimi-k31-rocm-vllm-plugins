"""Receive side of per-rank expert-LoRA slices in the colocated weight sync. A tensor adapter headed by an EP tag
[ep_rank, ep_size, num_experts] (expert block) or a TP tag [tp_rank, tp_size] (w1/w3 lora_B dim 1, w2 lora_A dim 2
narrowed) is checked against this worker's MoE LoRA layout and its tag stripped; pre-sliced TP shards are not sliced
again. Any mismatch raises. Untagged adapters load unchanged."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# the trainer exporter's tag names (mLite K31 checkpoint.py)
EP_TAG = "__k31_lora_expert_slice__"
TP_TAG = "__k31_lora_expert_tp_slice__"

# (name suffix, dim sharded over TP by vLLM's FusedMoEWithLoRA: _slice_w13_b / _slice_w2_a)
_TP_SHARDED = (
    (".experts.w1.lora_B.weight", 1),
    (".experts.w3.lora_B.weight", 1),
    (".experts.w2.lora_A.weight", 2),
)


def tp_sharded_dim(name):
    for suffix, dim in _TP_SHARDED:
        if name.endswith(suffix):
            return dim
    return None


def _moe_lora_modules(adapter_manager):
    return [m for m in getattr(adapter_manager, "modules", {}).values() if hasattr(m, "w13_lora_b_stacked")]


def localize(weights, adapter_manager):
    """Check and strip a slice tag (EP or TP) from an adapter's tensors; return them ready for vLLM's loader.

    ``adapter_manager`` is the worker's LoRA model manager (``WorkerLoRAManager._adapter_manager``). An adapter without
    a tag is returned unchanged (whole adapter: vLLM slices it itself).
    """
    has_ep, has_tp = EP_TAG in weights, TP_TAG in weights
    if not (has_ep or has_tp):
        return weights
    if has_ep and has_tp:
        raise RuntimeError("K31 adapter carries both an EP and a TP slice tag")
    moes = _moe_lora_modules(adapter_manager)
    if not moes:
        raise RuntimeError("expert-sliced K31 adapter reached a vLLM worker without MoE LoRA layers")
    if has_ep:
        moe = moes[0]
        placement = getattr(getattr(moe, "base_layer", None), "expert_placement_strategy", None)
        mpc = getattr(getattr(moe, "moe_config", None), "moe_parallel_config", None)
        return localize_ep(
            weights,
            getattr(adapter_manager, "moe_ep_load_spec", None),
            placement,
            bool(getattr(mpc, "enable_eplb", False)),
        )
    return localize_tp(weights, moes)


def localize_ep(weights, ep_spec, placement=None, eplb=False):
    """EP block ``[r, R, E]``: check it against this worker's expert placement and drop the tag."""
    rank, size, num_experts = (int(v) for v in weights[EP_TAG].detach().to("cpu").reshape(-1).tolist())
    if ep_spec is None:
        raise RuntimeError(
            f"expert-sliced K31 adapter (EP rank {rank}/{size}) reached a vLLM worker without "
            "expert parallelism for its LoRA MoE layers"
        )
    if placement not in (None, "linear") or eplb:
        raise RuntimeError(
            "expert-sliced K31 adapter needs vLLM's linear expert placement without EPLB, got "
            f"placement={placement!r} eplb={eplb}"
        )
    want = (
        ep_spec.ep_rank,
        ep_spec.global_num_experts // max(ep_spec.local_num_experts, 1),
        ep_spec.global_num_experts,
    )
    if (rank, size, num_experts) != want or num_experts % size:
        raise RuntimeError(
            f"expert-sliced K31 adapter for EP rank {rank}/{size} ({num_experts} experts) reached a "
            f"vLLM worker with EP rank {want[0]}/{want[1]} ({want[2]} experts): placement mismatch"
        )
    local = num_experts // size
    out = {}
    for name, t in weights.items():
        if name == EP_TAG:
            continue
        if ".experts." in name and t.dim() == 3 and t.shape[0] not in (1, local):
            raise RuntimeError(
                f"expert-sliced K31 adapter tensor {name} has expert dim {t.shape[0]}, expected 1 "
                f"(shared) or {local} (EP rank {rank}/{size})"
            )
        out[name] = t
    return out


def localize_tp(weights, moes):
    """TP shard ``[tp_rank, tp_size]``: check against every MoE LoRA layer, install the pass-through, drop the tag."""
    rank, size = (int(v) for v in weights[TP_TAG].detach().to("cpu").reshape(-1).tolist())
    got = {(int(m.tp_rank), int(m.tp_size), int(m._tp_shard_size), bool(m.use_ep)) for m in moes}
    if len(got) != 1:
        raise RuntimeError(f"TP-sliced K31 adapter: MoE LoRA layers disagree on their TP shard: {sorted(got)}")
    w_rank, w_size, shard, use_ep = got.pop()
    if use_ep or (rank, size) != (w_rank, w_size) or size < 2:
        raise RuntimeError(
            f"TP-sliced K31 adapter for TP rank {rank}/{size} reached a vLLM worker whose MoE LoRA "
            f"layers are TP rank {w_rank}/{w_size} (use_ep={use_ep}): trainer/rollout layout mismatch"
        )
    out = {}
    n_sliced = 0
    for name, t in weights.items():
        if name == TP_TAG:
            continue
        dim = tp_sharded_dim(name)
        if dim is not None:
            if t.dim() != 3 or t.shape[dim] != shard:
                raise RuntimeError(
                    f"TP-sliced K31 adapter tensor {name} {tuple(t.shape)}: dim {dim} should be this "
                    f"worker's expert shard {shard} (TP rank {rank}/{size})"
                )
            n_sliced += 1
        out[name] = t
    install_presliced_tp_passthrough()
    logger.info("TP-sliced K31 adapter for TP rank %d/%d (%d expert factors at shard %d)", rank, size, n_sliced, shard)
    return out


def install_presliced_tp_passthrough():
    """Make ``FusedMoEWithLoRA``'s TP slicing keep a factor that already has this rank's shard width.

    vLLM shards ``w13_lora_b`` (dim 1) and ``w2_lora_a`` (dim 2) by ``_tp_shard_size``; a whole factor has
    ``_tp_shard_size * tp_size`` there, a pre-sliced one ``_tp_shard_size``: with TP >= 2 the two never coincide, so
    whole adapters (incl. vLLM's warm-up dummy) keep vLLM's own path.  Idempotent.
    """
    from vllm.lora.layers import fused_moe as fm

    changed = False
    for cls_name in ("FusedMoEWithLoRA", "FusedMoE3DWithLoRA"):
        cls = getattr(fm, cls_name, None)
        if cls is None:
            continue
        for meth, dim in (("_slice_w13_b", 1), ("_slice_w2_a", 2)):
            orig = cls.__dict__.get(meth)
            if orig is None or getattr(orig, "_k31_presliced", False):
                continue

            def wrapped(self, t, _orig=orig, _dim=dim):
                if self.tp_size > 1 and t is not None and t.dim() == 3 and t.shape[_dim] == self._tp_shard_size:
                    return t
                return _orig(self, t)

            wrapped._k31_presliced = True
            setattr(cls, meth, wrapped)
            changed = True
    return changed


def _check_expert_scaling(request):
    """Under BI an expert-LoRA adapter needs a power-of-two alpha / r (vLLM keeps bf16(B * s); the trainer's BI MoE uses
    bf16(B) * s). Checked when the request carries its peft config; the trainer checks the same ratio."""
    from .bi_env import bi, is_pow2

    peft = getattr(request, "peft_config", None)
    if not bi() or not isinstance(peft, dict):
        return
    targets = peft.get("target_modules") or ()
    if not any("experts" in str(t) for t in ([targets] if isinstance(targets, str) else targets)):
        return
    alpha, r = peft.get("lora_alpha"), peft.get("r")
    if alpha is None or not r or not is_pow2(float(alpha) / float(r)):
        raise RuntimeError(f"K31 BI expert LoRA needs a power-of-two lora_alpha / r, got {alpha} / {r}")


def install():
    """Localize tagged tensor adapters in the worker LoRA managers' add_adapter (before vLLM loads them). Idempotent."""
    from vllm.lora import worker_manager as wm

    changed = False
    for cls_name in ("WorkerLoRAManager", "LRUCacheWorkerLoRAManager"):
        cls = getattr(wm, cls_name, None)
        orig = None if cls is None else cls.__dict__.get("add_adapter")
        if orig is None or getattr(orig, "_k31_lora_slice", False):
            continue

        def add_adapter(self, request, _orig=orig):
            _check_expert_scaling(request)
            tensors = getattr(request, "lora_tensors", None)
            if isinstance(tensors, dict) and (EP_TAG in tensors or TP_TAG in tensors):
                request.lora_tensors = localize(tensors, self._adapter_manager)
            return _orig(self, request)

        add_adapter._k31_lora_slice = True
        setattr(cls, "add_adapter", add_adapter)
        changed = True
    return changed
