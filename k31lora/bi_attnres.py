"""BI AttnRes: the image's `_attn_res_kernel` with its launch tiling pinned to (BLOCK_L, num_warps) = (1, 4) for every
batch (the image picks it by batch size, which changes a row's FP32 rounding). Rebinds `attn_res` in the AMD ops and
linear modules; the trainer calls the same pinned op."""
import os


def _pinned_attn_res(orig_mod):
    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
    from vllm.triton_utils import triton

    kernel = orig_mod._attn_res_kernel
    bl, nw = 1, 4

    def attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx, eps,
                 output_norm_eps, *, quant_dtype=None):
        num_tokens, hidden_size = prefix.shape
        assert prefix.stride(-1) == 1
        assert delta is None or delta.stride(-1) == 1
        assert blocks.stride(-1) == 1
        assert norm_weight.stride(-1) == 1
        assert qk_weight.stride(-1) == 1
        assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
        if quant_dtype is not None:
            assert quant_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        output = torch.empty_like(prefix, dtype=quant_dtype or prefix.dtype, memory_format=torch.contiguous_format)
        scale = (torch.empty((num_tokens, 1), device=prefix.device, dtype=torch.float32)
                 if quant_dtype is not None else None)
        if num_tokens == 0:
            return output if scale is None else (output, scale)
        kernel[(num_tokens,)](
            prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output, scale,
            prefix.stride(0), 0 if delta is None else delta.stride(0), blocks.stride(0), blocks.stride(1),
            output.stride(0), num_blocks, hidden_size, block_write_idx, eps, output_norm_eps,
            HAS_DELTA=delta is not None, WRITE_BLOCK=block_write_idx >= 0,
            APPLY_OUTPUT_NORM=output_norm_weight is not None,
            QUANT_MAX=0.0 if quant_dtype is None else get_fp8_min_max()[1],
            BLOCK_L=bl,                                   # PINNED (default: 1 if num_tokens >= 256 else 4)
            BLOCK_D=triton.next_power_of_2(hidden_size),
            num_warps=nw,                                 # PINNED (default: 4 if num_tokens >= 256 else 8)
            num_stages=2,
        )
        return output if scale is None else (output, scale)

    attn_res._k31_bi_pinned = (bl, nw)
    return attn_res


def install():
    import importlib
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "1":
        return False
    ops = importlib.import_module("vllm.models.kimi_k3.amd.ops.attn_res")
    if getattr(ops.attn_res, "_k31_bi_pinned", None):
        return True
    fn = _pinned_attn_res(ops)
    ops.attn_res = fn
    lin = importlib.import_module("vllm.models.kimi_k3.amd.linear")
    lin.attn_res = fn
    return True
