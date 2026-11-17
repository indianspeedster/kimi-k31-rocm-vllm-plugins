"""ROCm fixes for vLLM's batch-invariant matmul (VLLM_BATCH_INVARIANT=1). Opt-in per model (they change bits)."""
import os


def bi_small_m_tiles():
    """BI on ROCm: small-M tiles for vLLM's BI persistent BF16 matmul (M <= 64). BLOCK_K stays 64, so each element's
    K-reduction order (and its bits) is unchanged."""
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "1":
        return "off"
    import torch
    from vllm.platforms import current_platform
    if not current_platform.is_rocm():
        return "off"
    from vllm.model_executor.determinism import batch_invariant as BI
    if getattr(BI, "_k31_small_m", False):
        return "on"
    orig = BI._get_matmul_config

    def _get_matmul_config(M, N, K, dtype, default):
        if dtype == torch.bfloat16 and M <= 64 and default.get("BLOCK_SIZE_K") == 64:
            bn = 128 if N >= 16384 else 64
            bm = 16 if M <= 32 else 64
            if N >= 16384:
                bm = 32 if M <= 16 else 64
            return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8,
                    "num_stages": 3, "num_warps": 4}
        return orig(M, N, K, dtype, default)
    BI._get_matmul_config = _get_matmul_config
    BI._k31_small_m = True
    return "on"
