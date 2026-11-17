"""Old name (kimi-k31-rocm-vllm-plugins 0.1), kept for the trainer.
This module is vllm_rl_plugins.common.bi_lora (the same module object)."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("vllm_rl_plugins.common.bi_lora")
