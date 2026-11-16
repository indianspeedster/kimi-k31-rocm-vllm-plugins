"""Environment contract shared by the rollout plugins and the trainer: under VLLM_BATCH_INVARIANT=1 every switch that
changes bits accepts only its one supported value, so both sides always run the same arithmetic. Removed switches are
accepted only at the value that names today's fixed behavior."""

import os

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


def is_pow2(x: float) -> bool:
    """True for a positive power of two: bf16(B * s) == fp32(bf16(B) * s) for every B exactly then."""
    import math

    return x > 0 and math.frexp(x)[0] == 0.5


def bi() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


def check(where: str) -> None:
    """Raise if a removed switch is set to anything but its fixed value, or (under BI) a pinned switch is changed."""
    bad = [f"{k}={v!r}" for k, ok in RETIRED.items() if (v := os.environ.get(k, "")) != "" and v not in ok]
    if os.environ.get("K31_KDA_BI", "0") == "1" and not bi():
        bad.append("K31_KDA_BI='1' without VLLM_BATCH_INVARIANT=1")
    if bi():
        bad += [f"{k}={v!r}" for k, want in PINNED_BI.items() if (v := os.environ.get(k, "")) not in ("", want)]
    if bad:
        raise RuntimeError(f"K31 {where}: unsupported switch values (behavior is fixed): {', '.join(bad)}; unset them")
