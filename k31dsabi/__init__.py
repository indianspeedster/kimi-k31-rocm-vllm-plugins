"""k31dsabi: batch-invariant DSA for the ROCm Kimi-K3.1 model (vLLM general plugin, VLLM_BATCH_INVARIANT=1 only).

- Kimi-K3.1 DSA layers force sparse MQA and disable ShardQ, so prefill and decode rows take the same kernel.
- The indexer computes decode logits like prefill (row-bounded kernels) and selects top-k deterministically
  (indexer.py); MXFP4 scoring and DCP are refused.
- Sparse MLA runs on the Triton kernel with one KV split (no FlyDSL).
- The DSA indexer and sparse MLA backends, and the unused AITER FA prefill backend, accept batch-invariant mode
  for this model only.
"""
import functools
import importlib
import logging
import os

log = logging.getLogger("vllm")


def _bi() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


def _is_dsa_model(cfg) -> bool:
    return cfg is not None and cfg.model_config is not None and getattr(
        cfg.model_config.hf_text_config, "dsa_config", None) is not None


def _patch_config() -> None:
    from vllm.model_executor.models import config as C

    cls = C.KimiK3ForConditionalGenerationConfig
    orig = cls.verify_and_update_config.__func__

    @classmethod
    def verify_and_update_config(c, vllm_config):
        orig(c, vllm_config)
        att = vllm_config.attention_config
        att.sparse_mla_force_mqa = True
        if getattr(att, "sparse_mla_shard_q", False):
            att.sparse_mla_shard_q = False
        log.info("VLLM_BATCH_INVARIANT=1: Kimi K3 DSA uses sparse MQA without ShardQ")

    cls.verify_and_update_config = verify_and_update_config


def _patch_indexer_layer() -> None:
    from vllm.models.kimi_k3.amd import dsa

    cls = dsa.KimiK3DSAIndexer
    orig = cls.__init__

    @functools.wraps(orig)
    def __init__(self, *a, **k):
        orig(self, *a, **k)
        from vllm.config import get_current_vllm_config

        if self.use_fp4 or get_current_vllm_config().parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("VLLM_BATCH_INVARIANT=1: the ROCm Kimi K3.1 DSA indexer supports the FP8 "
                                      "index cache without decode context parallelism only")

    cls.__init__ = __init__


def _patch_backends() -> None:
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.attention.backends.mla import indexer, rocm_aiter_mla_sparse
    from vllm.v1.attention.backends.mla.prefill import aiter_flash_attn

    def indexer_bi(cls) -> bool:
        cfg = get_current_vllm_config_or_none()
        # KV-cache grouping on the worker may run outside the config context; this plugin targets K3.1 only.
        return True if cfg is None or cfg.model_config is None else _is_dsa_model(cfg)

    indexer.DeepseekV32IndexerBackend.supports_batch_invariance = classmethod(indexer_bi)
    rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend.supports_batch_invariance = classmethod(lambda cls: True)

    # The AITER FA prefill backend may be selected only when it never runs: every MLA layer is DSA with forced MQA.
    pre = aiter_flash_attn.AiterFlashAttnPrefillBackend
    orig_validate = pre.validate_configuration.__func__

    @classmethod
    def validate_configuration(cls, device_capability, selector_config):
        reasons = orig_validate(cls, device_capability, selector_config)
        cfg = get_current_vllm_config_or_none()
        if _is_dsa_model(cfg) and getattr(cfg.attention_config, "sparse_mla_force_mqa", False):
            reasons = [r for r in reasons if r != "batch invariance not supported"]
        return reasons

    pre.validate_configuration = validate_configuration


def _patch_sparse_mla() -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    # One kernel at every token count: FlyDSL only takes batches above a token threshold.
    rocm_aiter_ops._flydsl_sparse_mla_fwd = classmethod(lambda cls, *a, **k: (False, None))
    # No split-K: the default split count follows the number of query rows.
    sm = importlib.import_module("aiter.ops.triton.attention.sparse_mla")
    orig = sm.sparse_mla_fwd

    @functools.wraps(orig)
    def sparse_mla_fwd(*a, **k):
        k.setdefault("kv_splits", 1)
        return orig(*a, **k)

    sm.sparse_mla_fwd = sparse_mla_fwd


def register():
    if not _bi():
        return
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    import vllm._aiter_ops as A

    if getattr(A, "_k31dsabi", False):
        return
    _patch_config()
    _patch_indexer_layer()
    _patch_backends()
    _patch_sparse_mla()
    from . import indexer

    indexer.install()
    A._k31dsabi = True
    log.warning("K31 BI DSA installed: sparse MQA, deterministic indexer, one-split sparse MLA")
