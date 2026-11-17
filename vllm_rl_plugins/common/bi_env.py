"""Batch-invariance switch contract shared by the rollout plugins and the trainer: under VLLM_BATCH_INVARIANT=1 every
switch that changes bits accepts only its one supported value, so both sides always run the same arithmetic. Each
model supplies its own tables (pinned under BI, retired)."""

import os


def is_pow2(x: float) -> bool:
    """True for a positive power of two: bf16(B * s) == fp32(bf16(B) * s) for every B exactly then."""
    import math

    return x > 0 and math.frexp(x)[0] == 0.5


def bi() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


def check(prefix: str, where: str, pinned_bi: dict, retired: dict, extra=()) -> None:
    """Raise if a retired switch is set to anything but its fixed value, if `extra` (the model's own rules) found a
    problem, or (under BI) a pinned switch is changed."""
    bad = [f"{k}={v!r}" for k, ok in retired.items() if (v := os.environ.get(k, "")) != "" and v not in ok]
    bad += list(extra)
    if bi():
        bad += [f"{k}={v!r}" for k, want in pinned_bi.items() if (v := os.environ.get(k, "")) not in ("", want)]
    if bad:
        raise RuntimeError(f"{prefix} {where}: unsupported switch values (behavior is fixed): {', '.join(bad)}; unset them")
