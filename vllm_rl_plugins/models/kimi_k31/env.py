"""Kimi-K3.1 switch tables for the batch-invariance contract (common.bi_env), shared by the rollout and the trainer.
Removed switches are accepted only at the value that names today's fixed behavior."""

import os

from vllm_rl_plugins.common.bi_env import bi, is_pow2  # noqa: F401  (old k31lora.bi_env API)
from vllm_rl_plugins.common import bi_env as _bi_env

# Under BI: switch -> the only accepted value (unset = that value).
PINNED_BI = {
    "K31_BI_LORA_F32ADD": "1",
    "K31_BI_ATTNRES": "1",
    "K31_BI_MOE": "1",
}

# Removed switches -> accepted values (unset is always fine).
RETIRED = {
    "K31_BI_ATTNRES_CFG": ("1,4",),
    "K31_BI_ALLOW_NONBI_ATTN": ("0",),
    "K31_BI_AR": ("0",),
    "K31_BI_ADDMM_": ("1",),
    "K31_BI_MM_SMALLM": ("1",),
    "K31_PERSIST_SLEEP": ("0",),
    "K31_KDA_BI": ("0", "1"),
    "K31_KDA_BI_FLA_SHA": (),
    "K31_KDA_BI_ROW": ("1",),
    "K31_KDA_BI_GRAPH": ("1",),
    "K31_KDA_BI_LEAN2": ("1",),
    "K31_KDG_COMPLETE": ("row",),
    "K31_KDA_BI_CID_DEDUPE": ("1",),
    "K31_KDA_BI_EVICT_PROTECT": ("1",),
    "K31_KDA_BI_NULL_DUMMY": ("1",),
    "K31_KDA_BI_T3": ("0",),
    "K31_KDA_BI_DUMP": (),
    "K31_KDA_BI_DUMP_LAYER": (),
    "K31_BI_MOE_TUNED": ("1",),
    "K31_BI_MOE_ENGINE": ("dequant",),
    "K31_BI_MOE_ENGINE_FN": (),
    "K31_BI_MOE_GRAPH": ("1",),
    "K31_BI_MOE_FUSED_GLUE": ("1",),
    "K31_BI_MOE_SCALED": ("1",),
    "K31_BI_MOE_SCALED_CFG": (),
    "K31_BI_MOE_FUSED_COMBINE": ("1",),
    "K31_BI_MOE_FUSED_COMBINE_EP": ("1",),
    "K31_BI_MOE_FUSED_ROUTE": ("1",),
    "K31_BI_MOE_ROUTE_PAR": ("1",),
    "K31_BI_MOE_LORA_VIEW": ("1",),
    "K31_BI_MOE_LORA_TOKEN_SHRINK": ("1",),
    "K31_BI_MOE_LORA_INKERNEL_SCALE": ("1",),
    "K31_BI_MOE_ZERO_UNCOVERED": ("auto",),
    "K31_BI_MOE_ROUTE_MAX_SLOTS": ("1024",),
    "K31_BI_MOE_ROUTE_MAX_E": ("128",),
    "K31_BI_MOE_SCALED_W2_MAX_ROWS": ("2048",),
    "K31_BI_MOE_FUSED_W2": ("auto",),
    "K31_BI_MOE_FUSED_W2_MAX_ROWS": ("128",),
    "K31_BI_MOE_ENGINE_AUTO_MIN_T": ("64",),
    "K31_BI_MOE_ENGINE_AUTO_MIN_T_W2": ("256",),
}


def check(where: str) -> None:
    """Raise if a removed switch is set to anything but its fixed value, or (under BI) a pinned switch is changed."""
    extra = ["K31_KDA_BI='1' without VLLM_BATCH_INVARIANT=1"] if os.environ.get("K31_KDA_BI", "0") == "1" and not bi() else []
    _bi_env.check("K31", where, PINNED_BI, RETIRED, extra)
