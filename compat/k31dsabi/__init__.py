"""Old name (kimi-k31-rocm-vllm-plugins 0.1), kept for the trainer: this package is vllm_rl_plugins.models.kimi_k31.rocm.dsa, and k31dsabi.indexer
is its indexer (the same module objects)."""
import importlib
import sys

sys.modules[__name__ + ".indexer"] = importlib.import_module("vllm_rl_plugins.models.kimi_k31.rocm.dsa.indexer")
sys.modules[__name__] = importlib.import_module("vllm_rl_plugins.models.kimi_k31.rocm.dsa")
