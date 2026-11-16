"""k31kdabi: batch-invariant KDA for the ROCm Kimi-K3.1 vLLM model (vLLM general plugin, active only under
VLLM_BATCH_INVARIANT=1). Every request runs the trainer's expressions: FLA chunk_kda over 64-token chunks aligned to the
sequence start (autotuners pinned), FP32 causal conv + SiLU, FP32 gated RMSNorm. Spec decode and prefix caching are refused."""

from __future__ import annotations

import importlib
import os
import sys

_CHUNK = 64
_PINNED = False


def bi_enabled() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


def register():
    if not bi_enabled():
        return
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    mod = importlib.import_module("vllm.models.kimi_k3.amd.kda")
    cls = mod.KimiK3DeltaAttention
    if getattr(cls, "_k31kdabi_patched", False):
        return
    cls._k31kdabi_orig_forward = cls._forward
    # Keep vLLM's cudagraph break around the KDA core op (host-side work must not be captured).
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
        cls._forward = eager_break_during_capture(_bi_forward)
    except ImportError:  # vLLM without breakable cudagraphs: BI runs must then be eager
        cls._forward = _bi_forward
    orig_init = cls.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        _init_bi_state(self)

    cls.__init__ = __init__
    cls._k31kdabi_patched = True
    # With the forward above installed, the KDA backend is batch invariant (vLLM checks this under BI).
    backend = cls.get_attn_backend(None) if callable(getattr(cls, "get_attn_backend", None)) else None
    if backend is None:
        backend = getattr(mod, "KimiK3ROCmKDABackend")
    backend.supports_batch_invariance = classmethod(lambda c: True)


# ----------------------------------------------------------------------------------------------- shared math
def pin_fla_autotune():
    """Same as megatron primitive/kda.py `pin_fla_autotune`: every Triton autotuner on the chunk_kda path -> configs[0]."""

    global _PINNED
    if _PINNED:
        return
    from triton.runtime.autotuner import Autotuner

    importlib.import_module("fla.ops.kda")
    importlib.import_module("fla.ops.kda.chunk_bwd")
    pkgs = ("fla.ops.kda", "fla.ops.common", "fla.ops.gla", "fla.ops.utils", "fla.modules.l2norm")
    for name, m in list(sys.modules.items()):
        if m is None or not name.startswith(pkgs):
            continue
        for _attr, obj in list(vars(m).items()):
            t = obj
            for _ in range(8):
                if t is None or isinstance(t, Autotuner):
                    break
                t = getattr(t, "fn", None)
            if isinstance(t, Autotuner) and not getattr(t, "_k31_pinned", False):
                t.configs = [t.configs[0]]
                t._k31_pinned = True
    apply_bi_fla_fix()
    _PINNED = True


# sha256 prefix of the FLA 0.5.2 `fla/ops/kda/chunk_intra.py` that apply_bi_fla_fix patches.
_FLA_INTRA_SHA = "b59a1f2987d4577b"
_FIXED = None


def apply_bi_fla_fix():
    """Swap FLA's forward intra sub-chunk kernel for one whose SAFE_GATE reference row is the sub-chunk start (row bits
    independent of the sequence length). Rollout and trainer must both call this."""

    global _FIXED
    if _FIXED is not None:
        return _FIXED
    import hashlib
    import importlib.util
    import tempfile

    src_mod = importlib.import_module("fla.ops.kda.chunk_intra")
    path = src_mod.__file__
    text = open(path).read()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if not sha.startswith(_FLA_INTRA_SHA):
        raise RuntimeError(f"K31 BI KDA: unexpected FLA chunk_intra.py (sha256 {sha[:16]}), re-derive the fix for this FLA version")
    start = text.index("def chunk_kda_fwd_kernel_intra_sub_chunk(")
    end = text.index("@dispatch('kda')", start)
    body = text[start:end]
    old_g = "b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)"
    old_p = "p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + tl.arange(0, BK)"
    if body.count(old_g) != 1 or body.count(old_p) != 1:
        raise RuntimeError("K31 BI KDA: FLA sub-chunk kernel text not as expected")
    body = body.replace(old_g, "b_gn = gather(b_g, tl.full([1, BK], 0, dtype=tl.int16), axis=0)")
    body = body.replace(old_p, "p_gn = g + i_ti * HV*K + tl.arange(0, BK)")
    fixed = text[:start] + body + text[end:]
    cache = os.path.join(os.environ.get("K31_KDA_BI_CACHE", tempfile.gettempdir()), f"k31kdabi_fla_intra_{sha[:12]}.py")
    if not os.path.exists(cache) or open(cache).read() != fixed:
        tmp = cache + f".{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(fixed)
        os.replace(tmp, cache)
    spec = importlib.util.spec_from_file_location("k31kdabi_fla_intra_fixed", cache)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["k31kdabi_fla_intra_fixed"] = mod
    spec.loader.exec_module(mod)
    kern = mod.chunk_kda_fwd_kernel_intra_sub_chunk
    t = kern
    from triton.runtime.autotuner import Autotuner

    for _ in range(8):
        if t is None or isinstance(t, Autotuner):
            break
        t = getattr(t, "fn", None)
    if isinstance(t, Autotuner):
        t.configs = [t.configs[0]]
        t._k31_pinned = True
    src_mod.chunk_kda_fwd_kernel_intra_sub_chunk = kern
    _FIXED = cache
    return cache


def conv_silu_rows(ext, weight):
    """Trainer's causal conv expression on [hist(3) | new] rows; returns SiLU(conv) FP32 for every row (first 3 discarded by caller).

    ext [3 + L, C] bf16, weight [C, W] fp32. Same op order as megatron `causal_conv_silu`:
    acc = x * w[:, W-1]; for s in 1..W-1: acc = acc + x[t-s] * w[:, W-1-s]."""
    import torch
    import torch.nn.functional as F

    W = weight.shape[1]
    xf = ext.float()
    wf = weight.float()
    acc = xf * wf[:, W - 1]
    for shift in range(1, W):
        shifted = F.pad(xf[: xf.shape[0] - shift], (0, 0, shift, 0))
        acc = acc + shifted * wf[:, W - 1 - shift]
    return F.silu(acc)


def conv_silu_last(ext, weight):
    """conv_silu_rows for the LAST row of each [R, W, C] window only (decode): the same per-element op sequence
    (x[t] * w[W-1], then + x[t-s] * w[W-1-s] for s = 1..W-1, separate mul / add kernels, then SiLU) -> identical bits."""
    import torch.nn.functional as F

    W = weight.shape[1]
    wf = weight.float()
    acc = ext[:, W - 1].float() * wf[:, W - 1]
    for shift in range(1, W):
        acc = acc + ext[:, W - 1 - shift].float() * wf[:, W - 1 - shift]
    return F.silu(acc)


def gated_rms_norm(x, gate, weight, eps):
    """Trainer's expression (megatron primitive/kda.py `gated_rms_norm`)."""
    import torch

    xf = x.float()
    y = xf * torch.rsqrt(xf.square().sum(dim=-1, keepdim=True) / xf.shape[-1] + eps)
    return y * weight.float() * torch.sigmoid(gate.float())


def kda_chunk(q, k, v, g, beta_raw, A_log, dt_bias, lower, cu, init=None, final=False, cu_cpu=None):
    """The contract call: exactly the trainer's FLA chunk_kda arguments (megatron primitive/kda.py `fla_functional`)."""
    from fla.ops.kda import chunk_kda

    return chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta_raw.float().sigmoid(), A_log=A_log, dt_bias=dt_bias,
        initial_state=init, output_final_state=final, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        safe_gate=True, lower_bound=lower, cu_seqlens=cu,
        **({"cu_seqlens_cpu": cu_cpu} if cu_cpu is not None else {}),  # same indices, no device->host sync in FLA
    )


# ----------------------------------------------------------------------------------------------- BI step (pure)
def _to_dev(lists, dev):
    """Several Python int lists -> device int64 tensors with ONE pinned host->device copy."""
    import torch

    sizes = [len(x) for x in lists]
    flat = torch.empty(sum(sizes), dtype=torch.int64, pin_memory=True)
    o = 0
    for x, n in zip(lists, sizes):
        if n:
            flat[o:o + n] = torch.as_tensor(x, dtype=torch.int64)
        o += n
    return list(flat.to(dev, non_blocking=True).split(sizes))


def _seg_arange(starts, lens, total, dev):
    """concat_i arange(starts[i], starts[i] + lens[i]) on the GPU; `total` = sum(lens) known on the host (no sync)."""
    import torch

    n = lens.numel()
    offs = torch.cumsum(lens, 0) - lens
    rep = torch.repeat_interleave(torch.arange(n, device=dev), lens, output_size=total)
    return starts.index_select(0, rep) + torch.arange(total, device=dev) - offs.index_select(0, rep)


def _n_slots():
    """Per-layer BI slot count: K31_KDA_BI_SLOTS, else 2 * max_num_seqs + 8 (read at layer construction)."""
    env = os.environ.get("K31_KDA_BI_SLOTS")
    if env:
        return int(env)
    from vllm.config import get_current_vllm_config

    try:
        max_num_seqs = get_current_vllm_config().scheduler_config.max_num_seqs
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("K31 BI KDA: cannot read max_num_seqs from the vLLM config; set K31_KDA_BI_SLOTS") from exc
    if not max_num_seqs:
        raise RuntimeError("K31 BI KDA: vLLM max_num_seqs unset; set K31_KDA_BI_SLOTS")
    return 2 * int(max_num_seqs) + 8


def _init_bi_state(layer):
    """Allocate the per-slot BI buffers at layer construction, so vLLM's memory profiling counts them."""
    import torch

    layer._k31_bi_slots = _n_slots()
    w = getattr(getattr(layer, "conv1d", None), "weight", None)
    if w is None or w.device.type != "cuda":
        return
    P, D, H = layer.local_projection_size, layer.head_dim, layer.local_num_heads
    layer._bi_state = BIKDAState(layer._k31_bi_slots, P, H, D, w.shape[-1], w.device)
    lower = getattr(layer, "gate_lower_bound", None)
    if lower is not None:
        layer._bi_state.ensure_rows(w.device, lower)



def _all_padding_batch():
    """True iff every token of this forward is padding (vLLM cudagraph warmup/capture batches)."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return False
        ip = getattr(get_forward_context(), "is_padding", None)
        import torch
        return isinstance(ip, torch.Tensor) and ip.numel() > 0 and bool(ip.all())
    except Exception:  # noqa: BLE001
        return False

class BIKDAState:
    """Per-layer BI state per compact slot: open-chunk tail rows (<= 63), pos, null flag, conv history (W-1 raw rows).
    The boundary state lives in vLLM's `recurrent_state` (FLA [H, K, V] fp32)."""

    def __init__(self, n_slots, P, H, D, W, device):
        import torch

        self.P, self.H, self.D, self.W = P, H, D, W
        self.tail = torch.zeros((n_slots, _CHUNK - 1, 4 * P + H), dtype=torch.bfloat16, device=device)
        self.hist = torch.zeros((n_slots, W - 1, 3 * P), dtype=torch.bfloat16, device=device)
        self.pos = [0] * n_slots
        self.null = [True] * n_slots
        self.real = [-1] * n_slots  # compact id -> vLLM state index (the caller's recurrent_state row)
        self._cid = {}  # vLLM state index -> compact id
        self._lru = []  # compact ids, least recently used first
        self._cid_changes = []  # (vLLM state index, compact id | None) since the last device push (FULL mode)
        self.real[n_slots - 1] = -2  # the last id is the dummy slot for padded rows; never handed out

    def compact(self, state_idx, has_init):
        """vLLM state indices -> compact ids. A new request takes a free or least-recently-used id (never one owned by this
        batch); a continuing request must still own its id (fail closed). State index 0 (vLLM's null block) -> dummy id."""
        out = []
        protect = {self._cid[s] for s in state_idx if s in self._cid}
        dm = self.dummy_id()
        for s, hi in zip(state_idx, has_init):
            if s == 0:
                out.append(dm)
                continue
            c = self._cid.get(s)
            if c is None:
                if hi and not _all_padding_batch():
                    raise RuntimeError(f"K31 BI KDA: state {s} continues but its open-chunk rows were evicted; raise K31_KDA_BI_SLOTS")
                # an all-padding batch may "continue" a state it never prefilled: fresh id, output discarded
                if -1 in self.real:
                    c = self.real.index(-1)
                else:
                    c = self._evict(out, protect)
                    old = self.real[c]
                    del self._cid[old]
                    self._cid_changes.append((old, None))
                self.real[c] = s
                self._cid[s] = c
                self._cid_changes.append((s, c))
            if c in self._lru:
                self._lru.remove(c)
            self._lru.append(c)
            out.append(c)
        return out

    def _evict(self, taken, protect=()):
        """Victim id for a new request: least recently used (FULL mode: by the device last-use table), skipping `protect`
        unless nothing else is left."""
        cand = [x for x in self._lru if x not in taken and x not in protect] or [x for x in self._lru if x not in taken]
        rows = getattr(self, "rows", None)
        if _full_mode() and rows is not None and hasattr(rows, "lastuse_dev"):
            lu = rows.lastuse_dev.tolist()
            return min(cand, key=lambda x: lu[x])
        return cand[0]

    # ------------------------------------------------------------------------------------- row engine (decode)
    def ensure_rows(self, device, lower):
        if hasattr(self, "rows"):
            return self.rows
        import torch

        from .graphdec import StaticRowDecoder

        self.rows = StaticRowDecoder(self.tail.shape[0], self.H, self.D, device, lower, self.D ** -0.5, self._rs)
        self.rows.lastuse_dev = torch.zeros(self.tail.shape[0], dtype=torch.int64, device=device)
        self.rows.tick_dev = torch.zeros(1, dtype=torch.int64, device=device)
        return self.rows

    def dummy_id(self):
        return len(self.real) - 1

    def ensure_cid(self, recurrent_state):
        """cid_dev[vLLM state index] -> compact id (int32), default the dummy id. Rebuilt if the state cache moves."""
        import torch

        key = (recurrent_state.data_ptr(), recurrent_state.shape[0])
        if getattr(self, "_cid_key", None) == key:
            return self.cid_dev
        self.cid_dev = torch.full((recurrent_state.shape[0],), self.dummy_id(), dtype=torch.int32, device=recurrent_state.device)
        self._cid_key = key
        self._cid_changes = [(s_, c) for s_, c in self._cid.items() if s_ != 0]   # cid_dev[0] stays the dummy id
        return self.cid_dev

    def full_pre(self, recurrent_state, slots):
        """FULL mode, eager step: refresh host pos of the touched slots from the device (graph replays advanced it)."""
        rows = self.rows
        vals = rows.pos_dev.index_select(0, _to_dev([slots], recurrent_state.device)[0]).tolist()
        for s_, p_ in zip(slots, vals):
            self.pos[s_] = int(p_)
            rows.pos[s_] = int(p_)

    def full_post(self, recurrent_state, slots, A_log, dt_bias):
        """FULL mode, eager step: warm every touched slot, then push pos, last-use and the cid_dev changes to the device."""
        import torch

        rows = self.rows
        cold = [s_ for s_ in dict.fromkeys(slots) if not rows.warm[s_]]
        if cold:
            for s_ in cold:
                if self.null[s_]:
                    recurrent_state[self._rs([s_])[0]].zero_()  # boundary state not materialized yet: zero == None for FLA
                    self.null[s_] = False
            rows.warm_up(recurrent_state, cold, [self.tail[s_, :self.pos[s_]] for s_ in cold], A_log, dt_bias)
        uniq = list(dict.fromkeys(slots))
        d, p = _to_dev([uniq, [rows.pos[s_] for s_ in uniq]], recurrent_state.device)
        rows.pos_dev.index_copy_(0, d, p.to(torch.int32))
        rows.tick_dev += 1
        rows.lastuse_dev.index_copy_(0, d, rows.tick_dev.expand(d.numel()))
        if self._cid_changes:
            cid = self.ensure_cid(recurrent_state)
            # dedupe (last mapping wins, as the host map): index_copy_ with duplicate indices is nondeterministic
            ch = list(dict(self._cid_changes).items())
            self._cid_changes = []
            ch = [(a, None if a == 0 else b) for a, b in ch]   # cid_dev[0] stays the dummy id
            si, ci = _to_dev([[a for a, _ in ch], [self.dummy_id() if b is None else b for _, b in ch]], recurrent_state.device)
            cid.index_copy_(0, si, ci.to(torch.int32))

    def full_decode(self, recurrent_state, state_idx, raw, g1f, b1, g2, conv_w, A_log, dt_bias, norm_w, eps):
        """Decode-only batch, device-side only (no host reads, static shapes: capturable by vLLM FULL graphs). Padded rows
        (null block 0) run on the dummy slot, whose pos is parked at 0 so its chunk never completes."""
        import torch
        from .graphdec import decode_step_static

        rows = self.rows
        cid = self.ensure_cid(recurrent_state)
        dm = self.dummy_id()
        SL = cid.index_select(0, state_idx).to(torch_int64())
        SL = torch.where(state_idx == 0, torch.full_like(SL, dm), SL)
        RSL = state_idx.to(torch_int64())
        rows.pos_dev.narrow(0, dm, 1).zero_()
        out = decode_step_static(self, rows, recurrent_state, SL, RSL, raw, g1f, b1, g2, conv_w, A_log, dt_bias, norm_w, eps)
        rows.tick_dev += 1
        rows.lastuse_dev.index_copy_(0, SL, rows.tick_dev.expand(SL.numel()))
        return out

    def _rs(self, ids):
        # unmapped ids (direct step() callers) address state rows 1:1; the dummy id addresses the null block row 0
        return [self.real[c] if self.real[c] >= 0 else (0 if self.real[c] == -2 else c) for c in ids]

    def step(self, recurrent_state, slots, has_init, qsl, raw, g1f, b1, g2, conv_w, A_log, dt_bias, lower, norm_w, eps):
        """Eager step: decode requests (one token, existing state) through the row engine, the rest through the window path."""
        import torch

        n_req = len(qsl) - 1
        self.ensure_rows(raw.device, lower)
        rd = [i for i in range(n_req) if has_init[i] and qsl[i + 1] - qsl[i] == 1]
        for i in range(n_req):
            if not has_init[i] or qsl[i + 1] - qsl[i] != 1:
                self.rows.warm[slots[i]] = False  # phase 1 handles this slot now: row caches are rebuilt on its next decode
        if not rd:
            return self._step_p1(recurrent_state, slots, has_init, qsl, raw, g1f, b1, g2, conv_w, A_log, dt_bias, lower,
                                 norm_w, eps)
        p1 = [i for i in range(n_req) if i not in set(rd)]
        out = torch.empty((raw.shape[0], self.H, self.D), dtype=torch.bfloat16, device=raw.device)
        if p1:
            sel = torch.cat([torch.arange(qsl[i], qsl[i + 1], device=raw.device) for i in p1])
            q2 = [0]
            for i in p1:
                q2.append(q2[-1] + qsl[i + 1] - qsl[i])
            o1 = self._step_p1(recurrent_state, [slots[i] for i in p1], [has_init[i] for i in p1], q2, raw.index_select(0, sel),
                               g1f.index_select(0, sel), b1.index_select(0, sel), g2.index_select(0, sel), conv_w, A_log,
                               dt_bias, lower, norm_w, eps)
            out.index_copy_(0, sel, o1)
        # ---- row-engine decode for rd
        rs = [slots[i] for i in rd]
        tok, st = _to_dev([[qsl[i] for i in rd], rs], raw.device)
        Wm = self.W - 1
        with torch.no_grad():
            ext = torch.cat([self.hist.index_select(0, st), raw.index_select(0, tok)[:, None]], 1)  # [R, W, 3P]
            qkv = conv_silu_last(ext, conv_w).to(raw.dtype)
            self.hist.index_copy_(0, st, ext[:, 1:])
            cold = [j for j, s_ in enumerate(rs) if not self.rows.warm[s_]]
            if cold:
                cs = [rs[j] for j in cold]
                for s_ in cs:
                    if self.null[s_]:
                        recurrent_state[self._rs([s_])[0]].zero_()  # no boundary state yet: zero == None for FLA
                        self.null[s_] = False
                self.rows.warm_up(recurrent_state, cs, [self.tail[s_, :self.pos[s_]] for s_ in cs], A_log, dt_bias)
            g1r, b1r = g1f.index_select(0, tok), b1.index_select(0, tok)
            core = self.rows.decode(recurrent_state, rs, qkv, g1r, b1r, A_log, dt_bias)
            # keep the window path's view of these slots (tail rows, pos) current
            new_rows = torch.cat([qkv, g1r, b1r], 1)
            keep = [j for j, s_ in enumerate(rs) if self.pos[s_] < self.tail.shape[1]]
            if keep:
                kj, ks, kp = _to_dev([keep, [rs[j] for j in keep], [self.pos[rs[j]] for j in keep]], raw.device)
                self.tail[ks, kp] = new_rows.index_select(0, kj).to(self.tail.dtype)
            for s_ in rs:
                self.pos[s_] = (self.pos[s_] + 1) % _CHUNK
                if self.pos[s_] == 0:
                    self.null[s_] = False
            normed = gated_rms_norm(core, g2.index_select(0, tok), norm_w, eps).to(torch.bfloat16)
            out.index_copy_(0, tok, normed)
        return out

    def _step_p1(self, recurrent_state, slots, has_init, qsl, raw, g1f, b1, g2, conv_w, A_log, dt_bias, lower, norm_w, eps):
        """Window path. raw [T, 3P] (pre-conv q|k|v), g1f [T, P], b1 [T, H] raw b, g2 [T, H, D]; requests in token order
        with boundaries `qsl`. Returns normed core [T, H, D] bf16 and updates the per-slot state."""
        import torch

        P, H, D, W = self.P, self.H, self.D, self.W
        dev = raw.device
        n_req = len(qsl) - 1
        Wm = W - 1
        RW = 4 * P + H
        with torch.no_grad():
            for i in range(n_req):
                if not has_init[i]:
                    slot = slots[i]
                    self.pos[slot] = 0
                    self.null[slot] = True
            fresh = [i for i in range(n_req) if not has_init[i]]
            if fresh:
                self.hist.index_fill_(0, torch.tensor([slots[i] for i in fresh], device=dev), 0)
            T = raw.shape[0]
            C = _CHUNK - 1
            Ls = [qsl[i + 1] - qsl[i] for i in range(n_req)]
            ps = [self.pos[slots[i]] for i in range(n_req)]
            win_len = [p + L for p, L in zip(ps, Ls)]
            keep = Ls
            # per-request small vectors -> device in one pinned copy; every row index is built on the GPU from them
            slot_t, L_t, p_t = _to_dev([slots, Ls, ps], dev)
            blk_t = L_t + Wm
            blk_off = torch.cumsum(blk_t, 0) - blk_t
            win_t = p_t + L_t
            win_off = torch.cumsum(win_t, 0) - win_t
            Wm_t = torch.full_like(L_t, Wm)
            n_ext = n_req * Wm + T
            hist_dst_t = _seg_arange(blk_off, Wm_t, n_req * Wm, dev)
            new_dst_t = _seg_arange(blk_off + Wm, L_t, T, dev)
            hist_last_t = _seg_arange(blk_off + L_t, Wm_t, n_req * Wm, dev)
            ext = raw.new_empty((n_ext, 3 * P))
            ext.index_copy_(0, hist_dst_t, self.hist.index_select(0, slot_t).reshape(n_req * Wm, 3 * P))
            ext.index_copy_(0, new_dst_t, raw)
            # conv over all blocks at once: shifts <= W-1 never leave a request's own block for the rows kept
            qkv = conv_silu_rows(ext, conv_w).index_select(0, new_dst_t).to(raw.dtype)
            self.hist.index_copy_(0, slot_t, ext.index_select(0, hist_last_t).reshape(n_req, Wm, 3 * P))
            new_rows = torch.cat([qkv, g1f, b1], 1)  # [T, RW]
            # windows: [tail_i[:pos_i] | new rows_i]
            tail_flat = self.tail.view(-1, RW)
            n_tail = sum(ps)
            allwin = new_rows.new_empty((T + n_tail, RW))
            if n_tail:
                tsrc_t = _seg_arange(slot_t * C, p_t, n_tail, dev)
                tdst_t = _seg_arange(win_off, p_t, n_tail, dev)
                allwin.index_copy_(0, tdst_t, tail_flat.index_select(0, tsrc_t))
            allwin.index_copy_(0, _seg_arange(win_off + p_t, L_t, T, dev), new_rows)
            windows = list(allwin.split(win_len, 0))

            def split(rows):
                n = rows.shape[0]
                q, k, v = (rows[:, j * P:(j + 1) * P].reshape(1, n, H, D) for j in range(3))
                return q, k, v, rows[:, 3 * P:4 * P].reshape(1, n, H, D), rows[:, 4 * P:].reshape(1, n, H)

            def init_for(idx):
                # boundary state per request; zeros (== None for FLA) while the first chunk is still open
                sl, nulls = _to_dev([self._rs([slots[i] for i in idx]), [j for j, i in enumerate(idx) if self.null[slots[i]]]], dev)
                init = recurrent_state.index_select(0, sl)
                if nulls.numel():
                    init.index_fill_(0, nulls, 0)
                return init

            def run(idx, lengths, rows, final):
                if not idx:
                    return None, None
                cu_cpu = torch.zeros(len(lengths) + 1, dtype=torch.long)
                cu_cpu[1:] = torch.tensor(lengths, dtype=torch.long).cumsum(0)
                cu = cu_cpu.pin_memory().to(dev, non_blocking=True)
                q, k, v, g, b = split(rows)
                return kda_chunk(q, k, v, g, b, A_log, dt_bias, lower, cu, init=init_for(idx), final=final, cu_cpu=cu_cpu)

            new_states = {}
            # (1) outputs: ONE call over every request's whole window
            o_all, _ = run(list(range(n_req)), win_len, allwin, False)
            outs = o_all[0].split(win_len, 0)
            out_rows = [outs[i][-keep[i]:] for i in range(n_req)]
            # (2) boundary states for windows that complete >= 1 chunk: one call over their aligned prefixes
            cross = [i for i in range(n_req) if win_len[i] >= _CHUNK]
            if cross:
                alen = [(win_len[i] // _CHUNK) * _CHUNK for i in cross]
                rows = torch.cat([windows[i][:a] for i, a in zip(cross, alen)], 0)
                _, st = run(cross, alen, rows, True)
                for j, i in enumerate(cross):
                    new_states[i] = st[j]
            st_slots, st_vals = [], []
            d0, s0, nr = [], [], []
            woff = 0
            for i in range(n_req):
                slot, n = slots[i], win_len[i]
                a = (n // _CHUNK) * _CHUNK
                if i in new_states:
                    st_slots.append(slot)
                    st_vals.append(new_states[i])
                    self.null[slot] = False
                rest = n - a
                first = a if a > 0 else ps[i]  # rows [pos, n) are new; after a boundary the tail is window[a:], all new rows
                d0.append(slot * C + (first - a))
                s0.append(woff + first)
                nr.append(n - first)
                self.pos[slot] = rest
                woff += n
            if st_slots:
                recurrent_state.index_copy_(0, torch.tensor(self._rs(st_slots), dtype=torch.long).to(dev, non_blocking=True),
                                            torch.stack(st_vals))
            n_new = sum(nr)
            if n_new:
                d0_t, s0_t, nr_t = _to_dev([d0, s0, nr], dev)
                tail_flat.index_copy_(0, _seg_arange(d0_t, nr_t, n_new, dev),
                                      allwin.index_select(0, _seg_arange(s0_t, nr_t, n_new, dev)))
            o = torch.cat(out_rows, 0)
            return gated_rms_norm(o, g2, norm_w, eps).to(torch.bfloat16)


# ----------------------------------------------------------------------------------------------- vLLM adapter
def _full_mode() -> bool:
    """K31_KDA_BI_FULL=1: decode-only batches run device-side (BIKDAState.full_decode), capturable by vLLM FULL graphs."""
    return os.environ.get("K31_KDA_BI_FULL", "0") == "1"


def torch_int64():
    import torch

    return torch.int64


_META = {"key": None, "val": None}


def _host_meta(m):
    """Host copies of (query_start_loc, state indices, has_initial_state), read once per step for all KDA layers."""
    from vllm.forward_context import get_forward_context

    t_q, t_s, t_h = m.non_spec_query_start_loc, m.non_spec_state_indices_tensor, m.has_initial_state
    key = (get_forward_context(), m, t_q, t_s, t_h)
    hit = _META["key"]
    if hit is not None and all(a is b for a, b in zip(hit, key)):
        q, s, h = _META["val"]
        return q, list(s), list(h)
    qsl = t_q.tolist()
    n_req = len(qsl) - 1
    slots = t_s[:n_req].tolist()
    has_init = [True] * n_req if t_h is None else [bool(x) for x in t_h[:n_req].tolist()]
    _META["key"], _META["val"] = key, (qsl, slots, has_init)
    return qsl, list(slots), list(has_init)


def _bi_forward(self, mixed_qkv, g1, g2, beta, core_attn_out):
    import torch
    from vllm.forward_context import get_forward_context

    md = get_forward_context().attn_metadata
    if md is None:
        return
    m = md.get(self.prefix) if isinstance(md, dict) else None
    if m is None:
        return
    if m.spec_sequence_masks is not None or m.num_spec_decodes:
        raise NotImplementedError("K31 BI KDA: speculative decoding is not supported under VLLM_BATCH_INVARIANT")
    if getattr(m, "checkpoint", None) is not None:
        raise NotImplementedError("K31 BI KDA: prefix-cache checkpoints are not supported under VLLM_BATCH_INVARIANT")
    pin_fla_autotune()
    P, D, H = self.local_projection_size, self.head_dim, self.local_num_heads
    n_tok = m.num_actual_tokens
    _conv_state, recurrent_state = self.kv_cache
    if recurrent_state.dtype != torch.float32 or tuple(recurrent_state.shape[-3:]) != (H, D, D):
        raise RuntimeError(f"K31 BI KDA needs an FP32 [H, {D}, {D}] state, got {recurrent_state.dtype} {tuple(recurrent_state.shape)}")
    W = self.conv1d.weight.shape[-1]
    st = getattr(self, "_bi_state", None)
    if st is None:
        n = getattr(self, "_k31_bi_slots", None) or _n_slots()
        st = self._bi_state = BIKDAState(n, P, H, D, W, mixed_qkv.device)
    full = _full_mode()
    if full:
        st.ensure_rows(mixed_qkv.device, self.gate_lower_bound)
        st.ensure_cid(recurrent_state)
    if full and m.num_prefills == 0:
        # FULL decode: R = padded batch rows, no host reads
        sidx = m.non_spec_state_indices_tensor
        R = sidx.shape[0]
        out = st.full_decode(recurrent_state, sidx, mixed_qkv[:R], g1[0, :R].reshape(R, P), beta[0, :R].reshape(R, H),
                             g2[:R], self.conv1d.weight.view(self.conv1d.weight.size(0), W), self.A_log, self.dt_bias,
                             self.o_norm.weight.to(torch.bfloat16), self.o_norm.eps)
        core_attn_out[0, :R] = out.to(core_attn_out.dtype)
        return
    qsl, slots, has_init = _host_meta(m)
    slots = st.compact(slots, has_init)
    args = (mixed_qkv[:n_tok], g1[0, :n_tok].reshape(n_tok, P), beta[0, :n_tok].reshape(n_tok, H), g2[:n_tok],
            self.conv1d.weight.view(self.conv1d.weight.size(0), W), self.A_log, self.dt_bias, self.gate_lower_bound,
            self.o_norm.weight.to(torch.bfloat16), self.o_norm.eps)
    if full:
        st.full_pre(recurrent_state, slots)
    out = st.step(recurrent_state, slots, has_init, qsl, *args)
    if full:
        st.full_post(recurrent_state, slots, self.A_log, self.dt_bias)
    core_attn_out[0, :n_tok] = out.to(core_attn_out.dtype)
