"""LoRA support for the ROCm Kimi-K3.1 model: MoE LoRA only with K31LORA_EXPERTS=1, the replicated KDA f_a shard and
the `.base_layer.` weight-name alias for reload."""
import os


def install():
    if os.environ.get("K31LORA_EXPERTS", "0") != "1":
        from vllm.model_executor.layers.fused_moe import config as moe_cfg
        cls = moe_cfg.FusedMoEConfig
        if not getattr(cls, "_k31lora_patched", False):
            orig = cls.__init__

            def __init__(self, *a, **k):
                k["is_lora_enabled"] = False
                orig(self, *a, **k)

            cls.__init__ = __init__
            cls._k31lora_patched = True
    _patch_replicated_shard()
    _patch_base_layer_alias()


def _patch_base_layer_alias():
    from vllm.models.kimi_k3.amd.linear import KimiLinearModel as KM
    if getattr(KM, "_k31lora_alias", False):
        return
    orig = KM.load_weights

    def load_weights(self, weights):
        if not any(".base_layer." in n for n, _ in self.named_parameters()):
            return orig(self, weights)
        orig_np = self.named_parameters

        # present only unwrapped names (checkpoint-name lookups and the plain expert mapping)
        def named_parameters(*a, **k):
            for n, p in orig_np(*a, **k):
                yield n.replace(".base_layer.", "."), p

        def strip(ws):
            for w in ws:
                yield (w[0].replace(".base_layer.", "."),) + tuple(w[1:])

        self.named_parameters = named_parameters
        try:
            return orig(self, strip(weights))
        finally:
            del self.named_parameters

    KM.load_weights, KM._k31lora_alias = load_weights, True


def _patch_replicated_shard():
    from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA as L
    if getattr(L, "_k31lora_patched", False):
        return
    orig_init, orig_set = L.__init__, L.set_lora

    def __init__(self, base_layer, *a, **k):
        orig_init(self, base_layer, *a, **k)
        shard = getattr(self.base_layer, "replicated_shard_id", None)
        rep = (shard,) if shard is not None else ((0, 1) if type(self.base_layer).__name__ == "KimiK3MergedQKVGateLinear" else ())
        if rep:
            ids = list(self.output_ids)
            for s_ in rep:
                if s_ < self.n_slices:
                    ids[s_] = 0
            self.output_ids = tuple(ids)

    def set_lora(self, index, lora_a, lora_b, *a, **k):
        if (getattr(self.base_layer, "replicated_shard_id", None) is not None and isinstance(lora_b, list)
                and len(lora_b) < self.n_slices):
            miss = self.n_slices - len(lora_b)
            lora_a = list(lora_a) + [None] * miss
            lora_b = list(lora_b) + [None] * miss
        return orig_set(self, index, lora_a, lora_b, *a, **k)

    L.__init__, L.set_lora, L._k31lora_patched = __init__, set_lora, True
