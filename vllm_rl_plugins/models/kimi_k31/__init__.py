"""Kimi-K3 / Kimi-K3.1. Platform code: rocm/ (vllm.models.kimi_k3.amd, MI355X)."""
from vllm_rl_plugins.registry import ModelSpec

SPEC = ModelSpec(
    name="kimi_k31",
    archs=("KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration"),
    platforms={"rocm": "vllm_rl_plugins.models.kimi_k31.rocm"},
)
