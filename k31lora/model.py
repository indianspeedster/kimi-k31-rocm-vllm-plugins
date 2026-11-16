from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration as _Base


class KimiK3LoRAForConditionalGeneration(_Base, SupportsLoRA):
    # fused vLLM module -> adapter module names, in the fused layer's output-slice order
    packed_modules_mapping = {
        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules: dict = {}
    is_3d_moe_weight = True
