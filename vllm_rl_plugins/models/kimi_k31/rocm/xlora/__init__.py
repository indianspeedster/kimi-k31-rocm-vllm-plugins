"""W4A16 expert LoRA for the ROCm Kimi-K3.1 model, batch-variant arm only (VLLM_BATCH_INVARIANT=0).

Needs K31LORA_EXPERTS=1 (the default in this arm) so that lora.py keeps LoRA on the MoE layers. XLORA_FAST=1 runs
the experts on AITER's moe_gemm_a16w4 (AiterTritonMxfp4LoRAExperts); otherwise vLLM's MXFP4 emulation experts with
BF16 activations.
"""
import os


def install():
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1":
        return
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
        # unfused AITER Triton moe_gemm_a16w4 class (aiter_lora_experts.py) instead of
        # dequant-everything + TritonExperts.
        import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as _orc
        import vllm.model_executor.layers.quantization.mxfp4 as _q
        from .aiter_lora_experts import AiterTritonMxfp4LoRAExperts

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
