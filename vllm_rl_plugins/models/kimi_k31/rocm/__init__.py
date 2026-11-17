"""Kimi-K3.1 on ROCm. Arms: VLLM_BATCH_INVARIANT=1 (BI MoE, AttnRes, KDA, DSA, LoRA BI fixes) and =0 (xlora)."""
import os

MODEL_CLS = "vllm_rl_plugins.models.kimi_k31.rocm.model:KimiK3LoRAForConditionalGeneration"


def early(ctx):
    if not ctx.bi:
        return
    from . import dsa, kda

    kda.register()
    dsa.early()


def activate(ctx):
    from .. import env

    env.check("rollout")
    if not ctx.bi:
        os.environ.setdefault("K31LORA_EXPERTS", "1")
    from vllm_rl_plugins.common import bi_lora, lora_slice
    from vllm_rl_plugins.platforms.rocm import bi_matmul

    from . import attnres, bi_moe, lora

    lora.install()
    lora_slice.install()
    bi_lora.install()
    bi_lora.bi_inplace_addmm()
    bi_matmul.bi_small_m_tiles()
    bi_moe.install()
    attnres.install()
    if ctx.bi:
        from . import dsa

        dsa.activate()
    else:
        from . import xlora

        xlora.install()
