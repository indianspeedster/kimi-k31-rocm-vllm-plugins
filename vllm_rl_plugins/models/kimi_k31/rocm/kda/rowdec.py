"""BI KDA eager decode (row engine): one new token per request, bitwise equal to the FLA chunk contract. Per-slot caches
hold the open chunk's rows; a completed 64-token chunk's boundary state comes from FLA chunk_kda_fwd."""
from __future__ import annotations

import torch
import triton

from . import rowkern as RK


def _to_dev(lists, dev):
    from . import _to_dev as f

    return f(lists, dev)


def _gate_cfg():
    """FLA's pinned (configs[0]) BS / num_warps of kda_gate_chunk_cumsum_vector_kernel: tl.cumsum's scan layout must match."""
    from triton.runtime.autotuner import Autotuner
    import fla.ops.kda.gate as G

    t = G.kda_gate_chunk_cumsum_vector_kernel
    for _ in range(8):
        if isinstance(t, Autotuner):
            break
        t = getattr(t, "fn", None)
    if not getattr(t, "_k31_pinned", False):
        raise RuntimeError("K31 BI KDA row engine: FLA gate cumsum autotuner is not pinned (call pin_fla_autotune first)")
    c = t.configs[0]
    return c.kwargs["BS"], c.num_warps

BT = 64
BC = 16


class RowDecoder:
    def __init__(self, n_slots, H, D, device, lower, scale, rmap=lambda ids: list(ids)):
        f32, bf16 = torch.float32, torch.bfloat16
        self.H, self.D, self.lower, self.scale = H, D, float(lower), float(scale)
        z = lambda *shape, dt: torch.zeros(*shape, dtype=dt, device=device)  # noqa: E731
        self.Q = z(n_slots, H, BT, D, dt=bf16)
        self.KC = z(n_slots, H, BT, D, dt=bf16)
        self.VC = z(n_slots, H, BT, D, dt=bf16)
        self.GR = z(n_slots, H, BT, D, dt=bf16)
        self.G = z(n_slots, H, BT, D, dt=f32)
        self.BETA = z(n_slots, H, BT, dt=f32)
        self.AD = z(n_slots, H, BT, BC, dt=f32)
        self.AP = z(n_slots, H, BT, BC, dt=f32)
        self.AF = z(n_slots, H, BT, BT, dt=f32)
        self.VN = z(n_slots, H, BT, D, dt=bf16)
        self.pos = [0] * n_slots
        self.warm = [False] * n_slots
        self.rmap = rmap  # compact ids -> rows of the caller's recurrent_state

    # ------------------------------------------------------------------------------------------------ one row
    def _rows(self, recurrent_state, slots, q, k, v, graw, braw, A_log, dt_bias):
        """Append ONE row (q, k, v, raw gate [R, H*D] / raw beta [R, H]) to each slot's open chunk; returns core [R, H, D] BF16."""
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.utils.constant import RCP_LN2

        H, D = self.H, self.D
        R = len(slots)
        dev = q.device
        ts = [self.pos[s] for s in slots]
        sl, tt, rsl_ = _to_dev([slots, ts, self.rmap(slots)], dev)
        q, k, v, graw, braw = (x.contiguous() for x in (q, k, v, graw, braw))
        qn = l2norm_fwd(q.reshape(1, R, H, D))
        qn = qn[0] if isinstance(qn, tuple) else qn
        kn = l2norm_fwd(k.reshape(1, R, H, D))
        kn = kn[0] if isinstance(kn, tuple) else kn
        hh = torch.arange(H, device=dev)
        idx = (sl[:, None], hh[None, :], tt[:, None])
        self.Q[idx] = qn.reshape(R, H, D)
        self.KC[idx] = kn.reshape(R, H, D)
        self.VC[idx] = v.reshape(R, H, D)
        self.GR[idx] = graw.reshape(R, H, D)
        self.BETA[idx] = braw.reshape(R, H).float().sigmoid()
        # g_cum row t of each open chunk: FLA's gate-cumsum kernel expression on the slot's 64-row block with T = t + 1 (GPU, no host loop)
        SLp, POp = sl.to(torch.int32), tt.to(torch.int32)
        BS, nw = _gate_cfg()
        RK.b_gate[(R, H, triton.cdiv(D, BS))](SLp, POp, self.GR, self.G, A_log, dt_bias, self.lower, RCP_LN2, H, S=D, BT=BT, BS=BS,
                                               num_warps=nw)
        # scratch per request
        AQK = torch.zeros(R, H, BT, dtype=torch.bfloat16, device=dev)
        AKK = torch.zeros(R, H, BT, dtype=torch.bfloat16, device=dev)
        W = torch.empty(R, H, D, dtype=torch.bfloat16, device=dev)
        U = torch.empty(R, H, D, dtype=torch.bfloat16, device=dev)
        O = torch.empty(R, H, D, dtype=torch.bfloat16, device=dev)
        SL = sl.to(torch.int32)
        RSL = rsl_.to(torch.int32)
        PO = tt.to(torch.int32)
        grid = (R, H)
        RK.b_diag[grid](SL, PO, self.Q, self.KC, self.G, self.BETA, AQK, self.AD, self.AP, self.scale, H, K=D, BT=BT, BC=BC,
                        num_warps=1)
        RK.b_inter[grid](SL, PO, self.Q, self.KC, self.G, self.BETA, AQK, self.AD, self.AF, AKK, self.scale, H, K=D, BT=BT,
                         BC=BC, num_warps=4)
        RK.b_wu[grid](SL, PO, self.KC, self.VC, self.G, self.BETA, AKK, W, U, H, K=D, V=D, BT=BT, num_warps=2)
        if recurrent_state.stride(3) != 1 or recurrent_state.stride(2) != D:
            raise RuntimeError(f"K31 BI KDA row engine: per-head state must be a contiguous [K, V] block, strides {recurrent_state.stride()}")
        RK.b_stage_a[grid](SL, RSL, PO, recurrent_state.stride(0), recurrent_state.stride(1), self.Q, self.G, U, W, AQK, recurrent_state, self.VN, O, self.scale, H, K=D, V=D, BT=BT,
                           R=16, num_warps=4)
        # chunk completion -> next boundary state (FLA's own chunk forward on the cached chunk), open a new chunk
        done = [i for i, t in enumerate(ts) if t + 1 == BT]
        if done:
            self._complete(recurrent_state, [slots[i] for i in done], A_log, dt_bias)
            self._clear([slots[i] for i in done])
        for s, t in zip(slots, ts):
            self.pos[s] = (t + 1) % BT
        return O

    def _complete(self, recurrent_state, slots, A_log, dt_bias):
        from fla.ops.kda.chunk_fwd import chunk_kda_fwd

        H, D = self.H, self.D
        n = len(slots)
        sl, rsl = _to_dev([slots, self.rmap(slots)], self.Q.device)
        tq = lambda X: X.index_select(0, sl).transpose(1, 2).reshape(1, n * BT, H, D).contiguous()  # noqa: E731
        beta = self.BETA.index_select(0, sl).transpose(1, 2).reshape(1, n * BT, H).contiguous()
        cu = torch.arange(0, (n + 1) * BT, BT, device=self.Q.device, dtype=torch.long)
        h0 = recurrent_state.index_select(0, rsl)
        out = chunk_kda_fwd(q=tq(self.Q), k=tq(self.KC), v=tq(self.VC), g=tq(self.GR), beta=beta, scale=self.scale,
                            initial_state=h0, output_final_state=True, cu_seqlens=cu, safe_gate=True, lower_bound=self.lower,
                            use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias)
        recurrent_state.index_copy_(0, rsl, out[1])

    def _clear(self, slots):
        """Open a fresh chunk: zero the slot's row caches (rows above t must read as 0 in the row kernels, as in FLA's masked tiles)."""
        (sl,) = _to_dev([slots], self.Q.device)
        for X in (self.Q, self.KC, self.VC, self.GR, self.G, self.BETA, self.AD, self.AP, self.AF, self.VN):
            X.index_fill_(0, sl, 0)

    # ------------------------------------------------------------------------------------------------ public
    def warm_up(self, recurrent_state, slots, tails, A_log, dt_bias):
        """First decode of a slot after phase-1 prefill: replay its open-chunk rows (phase-1 tail rows: post-conv q|k|v, raw g, raw b)."""
        P = self.H * self.D
        for s in slots:
            self.pos[s] = 0
        if slots:
            self._clear(slots)
        maxp = max((t.shape[0] for t in tails), default=0)
        for j in range(maxp):
            sub = [(s, t[j]) for s, t in zip(slots, tails) if j < t.shape[0]]
            if not sub:
                continue
            rows = torch.stack([r for _, r in sub])
            self._rows(recurrent_state, [s for s, _ in sub], rows[:, :P], rows[:, P:2 * P], rows[:, 2 * P:3 * P],
                       rows[:, 3 * P:4 * P], rows[:, 4 * P:], A_log, dt_bias)
        for s in slots:
            self.warm[s] = True

    def decode(self, recurrent_state, slots, qkv, graw, braw, A_log, dt_bias):
        P = self.H * self.D
        return self._rows(recurrent_state, slots, qkv[:, :P], qkv[:, P:2 * P], qkv[:, 2 * P:], graw, braw, A_log, dt_bias)
