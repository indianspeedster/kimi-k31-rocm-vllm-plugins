"""k31xlora: expert-LoRA plugin for the ROCm Kimi-K3.1 image (vLLM f5926fb).
Put tracks/lora/plugin AND tracks/xlora/plugin on PYTHONPATH; set K31LORA_EXPERTS=1 so k31lora keeps
FusedMoEConfig.is_lora_enabled (experts get wrapped by FusedMoEWithLoRA).

XLORA_CAPTURE_DIR=<dir>: while <dir>/ON exists, every MoE experts apply() (any FusedMoEExpertsModular subclass we patch)
dumps {x, topk_ids, topk_weights, expert_map, out, lora_map} to <dir>/<pid>_<n>.pt. Use enforce_eager=True.
"""
import os


def _patch_capture(cls):
    if getattr(cls, "_xlora_cap", False):
        return
    orig = cls.apply
    d = os.environ["XLORA_CAPTURE_DIR"]
    st = {"n": 0}

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **k):
        r = orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *a, **k)
        if os.path.exists(os.path.join(d, "ON")):
            import torch
            em = k.get("expert_map", a[2] if len(a) > 2 else None)
            ctx = getattr(self, "_lora_context", None)
            lm = getattr(ctx, "local_token_lora_mapping", None) if ctx is not None else None
            rec = dict(cls=type(self).__name__, x=hidden_states.detach().cpu(), topk_ids=topk_ids.detach().cpu(),
                       topk_weights=topk_weights.detach().cpu(), out=output.detach().cpu(),
                       expert_map=None if em is None else em.detach().cpu(),
                       lora_map=None if lm is None else lm.detach().cpu(),
                       w1_shape=tuple(w1.shape), w2_shape=tuple(w2.shape))
            try:
                rec["tp_rank"] = int(os.environ.get("RANK", "-1"))
                import torch.distributed as dist
                if dist.is_initialized():
                    rec["rank"] = dist.get_rank()
            except Exception:
                pass
            torch.save(rec, os.path.join(d, f"{os.getpid()}_{st['n']:05d}.pt"))
            st["n"] += 1
        return r

    cls.apply = apply
    cls._xlora_cap = True


def _patch_prof(cls):
    """XLORA_PROF=1: time every experts apply() with M >= XLORA_PROF_MINM (default 1024) and log, per 32 such calls,
    the median/max ms and the allocator's num_alloc_retries delta (large-M prefill diagnosis; adds a sync per call)."""
    import torch
    orig = cls.apply
    minm = int(os.environ.get("XLORA_PROF_MINM", "1024"))
    st = {"t": [], "r0": None, "M": []}

    def apply(self, output, hidden_states, *a, **k):
        M = hidden_states.shape[0]
        if M < minm or torch.cuda.is_current_stream_capturing():
            return orig(self, output, hidden_states, *a, **k)
        if st["r0"] is None:
            st["r0"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        r = orig(self, output, hidden_states, *a, **k)
        e1.record(); torch.cuda.synchronize()
        st["t"].append(e0.elapsed_time(e1)); st["M"].append(M)
        if len(st["t"]) >= 32:
            t = sorted(st["t"])
            retries = torch.cuda.memory_stats().get("num_alloc_retries", 0) - st["r0"]
            print(f"XLORA_PROF rank? M={sorted(set(st['M']))} n={len(t)} med_ms={t[len(t)//2]:.2f} "
                  f"max_ms={t[-1]:.2f} alloc_retries+={retries} "
                  f"reserved_GB={torch.cuda.memory_reserved()/2**30:.1f}", flush=True)
            st.update(t=[], M=[], r0=None)
        return r

    cls.apply = apply


def register():
    from vllm.platforms import current_platform
    if not current_platform.is_rocm():
        return
    os.environ.setdefault("K31LORA_EXPERTS", "1")
    import k31lora
    k31lora.register()
    if os.environ.get("XLORA_A16", "1") == "1":
        # EMULATION's quant config is ocp_mx "mxfp4" = W4A4 (activations fake-quantized to MXFP4 before both GEMMs).
        # Production Kimi on gfx950 is AITER A16W4, so drop the activation fake-quant: weight-only MXFP4, BF16 acts.
        from vllm.model_executor.layers.fused_moe.experts import ocp_mx_emulation_moe as _em
        C = _em.OCP_MXQuantizationEmulationTritonExperts
        if not getattr(C, "_xlora_a16", False):
            _init = C.__init__

            def __init__(self, *a, **k):
                _init(self, *a, **k)
                self._quant_dtype = None

            C.__init__ = __init__
            C._xlora_a16 = True
    if os.environ.get("XLORA_FAST", "0") == "1":
        # Fast path: keep EMULATION's backend selection + raw (unshuffled) weight layout, but run the experts on the
        # unfused AITER Triton moe_gemm_a16w4 class (tracks/xlora/fast/aiter_lora_experts.py) instead of
        # dequant-everything + TritonExperts.
        import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as _orc
        import vllm.model_executor.layers.quantization.mxfp4 as _q
        from k31xlora.aiter_lora_experts import AiterTritonMxfp4LoRAExperts

        class KimiAiterA16W4LoRAExperts(AiterTritonMxfp4LoRAExperts):
            @property
            def quant_dtype(self):
                return None   # A16: never fake-quantize activations (EMULATION's quant config says mxfp4)

        if not getattr(_orc, "_xlora_fast", False):
            _b2k = _orc.backend_to_kernel_cls

            def backend_to_kernel_cls(backend, *a, **k):
                if backend == _orc.Mxfp4MoeBackend.EMULATION:
                    return [KimiAiterA16W4LoRAExperts]
                return _b2k(backend, *a, **k)

            _orc.backend_to_kernel_cls = backend_to_kernel_cls
            _q.backend_to_kernel_cls = backend_to_kernel_cls
            _orc._xlora_fast = True
            _patch_capture(KimiAiterA16W4LoRAExperts) if os.environ.get("XLORA_CAPTURE_DIR") else None
            if os.environ.get("XLORA_PROF", "0") == "1":
                _patch_prof(KimiAiterA16W4LoRAExperts)
    if os.environ.get("XLORA_CAPTURE_DIR"):
        from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
            OCP_MXQuantizationEmulationTritonExperts)
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import AiterMxfp4Experts
        _patch_capture(OCP_MXQuantizationEmulationTritonExperts)
        _patch_capture(AiterMxfp4Experts)
