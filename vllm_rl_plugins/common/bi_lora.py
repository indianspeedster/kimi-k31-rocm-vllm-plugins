"""BI fixes for vLLM LoRA under VLLM_BATCH_INVARIANT=1: the LoRA expand epilogue with one rounding and the in-place
addmm_. Opt-in per model (they change bits)."""
import inspect
import os
import re


def install():
    """BI: punica expand epilogue bf16(fp32(delta) + fp32(base)) (one rounding), recompiled from vLLM's own source."""
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "1":
        return "off"
    from vllm.lora.ops.triton_ops import kernel_utils as KU
    from vllm.lora.ops.triton_ops import lora_expand_op as LE
    if getattr(LE, "_k31_f32add", False):   # register() may run twice
        return "f32add"
    src = inspect.getsource(KU)
    old = re.compile(r"(\n(\s*)if ADD_INPUTS:\n\s*tiled_out = tl\.load\(c_ptr, mask=c_mask\)\n)\s*tiled_c \+= tiled_out\n")
    m = old.search(src)
    if m is None:
        raise RuntimeError("k31lora bi_lora: expand epilogue pattern not found; vLLM changed")
    ind = m.group(2) + "    "
    new = m.group(1) + f"{ind}tiled_c = (accumulator + tiled_out.to(tl.float32)).to(cur_lora_ptr.dtype.element_ty)\n"
    src = src[: m.start()] + new + src[m.end():]
    import hashlib
    import importlib.util
    import tempfile
    tag = hashlib.sha256(src.encode()).hexdigest()[:12]
    path = os.path.join(os.environ.get("K31_BI_TMP", tempfile.gettempdir()), f"k31_kernel_utils_f32add_{tag}.py")
    if not os.path.exists(path) or open(path).read() != src:   # triton needs a real source file; one per source version
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(src)
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(f"k31_kernel_utils_f32add_{tag}", path)
    m2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2)
    LE.do_expand_kernel = m2.do_expand_kernel
    LE._k31_f32add = True
    return "f32add"


_ADDMM_LIB = None


def bi_inplace_addmm():
    """BI: route the in-place aten::addmm_ through vLLM's batch-invariant addmm (vLLM overrides only addmm/addmm.out)."""
    global _ADDMM_LIB
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "1":
        return "off"
    if _ADDMM_LIB is not None:
        return "on"
    import torch
    from vllm.model_executor.determinism.batch_invariant import _addmm_impl
    from vllm.platforms import current_platform

    def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
        return self.copy_(_addmm_impl(self, mat1, mat2, beta, alpha, None))

    _ADDMM_LIB = torch.library.Library("aten", "IMPL")
    _ADDMM_LIB.impl("aten::addmm_", addmm_, current_platform.dispatch_key)
    return "on"
