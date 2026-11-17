"""The registry on CPU with a fake vLLM: which model code is registered and activated per platform and arm."""
import sys
import types

import pytest

from vllm_rl_plugins import registry
from vllm_rl_plugins.models import MODELS
from vllm_rl_plugins.models.kimi_k31 import SPEC as KIMI

ROCM = "vllm_rl_plugins.models.kimi_k31.rocm"


class _Registered(list):
    platform = None


@pytest.fixture
def fake_vllm(monkeypatch):
    """vllm.platforms.current_platform and vllm.ModelRegistry; returns the registered (arch, cls) list."""
    reg = _Registered()
    plat = types.SimpleNamespace(kind="rocm")
    plat.is_rocm = lambda: plat.kind == "rocm"
    plat.is_cuda = lambda: plat.kind == "cuda"
    vllm = types.ModuleType("vllm")
    vllm.ModelRegistry = types.SimpleNamespace(register_model=lambda arch, cls: reg.append((arch, cls)))
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = plat
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setattr(registry, "_EARLY", set())
    monkeypatch.setattr(registry, "_ACTIVE", set())
    reg.platform = plat
    return reg


@pytest.fixture
def calls(monkeypatch):
    """Replace the Kimi ROCm stage functions and installs with recorders."""
    import importlib

    rocm = importlib.import_module(ROCM)
    log = []
    monkeypatch.setattr(rocm, "early", lambda ctx: log.append(("early", ctx.bi)))
    monkeypatch.setattr(rocm, "activate", lambda ctx: log.append(("activate", ctx.bi)))
    return log


def test_models():
    assert MODELS == (KIMI,)
    assert KIMI.archs == ("KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration")
    assert KIMI.platforms == {"rocm": ROCM}


@pytest.mark.parametrize("bi", ["0", "1"])
def test_rocm_registers_kimi_and_runs_early_once(fake_vllm, calls, monkeypatch, bi):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", bi)
    registry.register()
    registry.register()
    cls = "vllm_rl_plugins.models.kimi_k31.rocm.model:KimiK3LoRAForConditionalGeneration"
    assert fake_vllm == [(a, cls) for a in KIMI.archs] * 2
    assert calls == [("early", bi == "1")]


@pytest.mark.parametrize("kind", ["cuda", "other"])
def test_other_platforms_get_nothing(fake_vllm, calls, kind):
    fake_vllm.platform.kind = kind
    registry.register()
    registry.activate(KIMI)
    assert fake_vllm == [] and calls == []


def test_activate_once(fake_vllm, calls, monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    registry.activate(KIMI)
    registry.activate(KIMI)
    assert calls == [("activate", True)]


def test_model_without_code_for_this_platform_is_skipped(fake_vllm, calls):
    other = registry.ModelSpec(name="x", archs=("XForCausalLM",), platforms={"cuda": "nowhere"})
    registry.register(models=[other])
    registry.activate(other)
    assert fake_vllm == [] and calls == []


@pytest.fixture
def kimi_installs(monkeypatch):
    """The Kimi ROCm activate() with every install replaced by a recorder (no vLLM, no GPU)."""
    import importlib

    rocm = importlib.import_module(ROCM)
    log = []

    def rec(name):
        return lambda *a, **k: log.append((name, __import__("os").environ.get("K31LORA_EXPERTS")))

    fakes = {
        "attnres": {"install": rec("attnres")},
        "bi_moe": {"install": rec("bi_moe")},
        "lora": {"install": rec("lora")},
        "dsa": {"activate": rec("dsa"), "early": rec("dsa.early")},
        "kda": {"register": rec("kda")},
        "xlora": {"install": rec("xlora")},
    }
    for name, fns in fakes.items():
        mod = types.SimpleNamespace(**fns)
        monkeypatch.setitem(sys.modules, f"{ROCM}.{name}", mod)
        monkeypatch.setattr(rocm, name, mod, raising=False)
    common = {
        "vllm_rl_plugins.common.bi_lora": {"install": rec("bi_lora"), "bi_inplace_addmm": rec("addmm_")},
        "vllm_rl_plugins.common.lora_slice": {"install": rec("lora_slice")},
        "vllm_rl_plugins.platforms.rocm.bi_matmul": {"bi_small_m_tiles": rec("small_m")},
    }
    for full, fns in common.items():
        pkg, name = full.rsplit(".", 1)
        mod = types.SimpleNamespace(**fns)
        monkeypatch.setitem(sys.modules, full, mod)
        monkeypatch.setattr(importlib.import_module(pkg), name, mod, raising=False)
    monkeypatch.delenv("K31LORA_EXPERTS", raising=False)
    return rocm, log


def test_kimi_rocm_bi_arm(kimi_installs, monkeypatch):
    rocm, log = kimi_installs
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    ctx = registry.Context(platform="rocm", bi=True)
    rocm.early(ctx)
    rocm.activate(ctx)
    assert [n for n, _ in log] == ["kda", "dsa.early", "lora", "lora_slice", "bi_lora", "addmm_", "small_m", "bi_moe",
                                   "attnres", "dsa"]


def test_kimi_rocm_variant_arm_keeps_expert_lora_for_xlora(kimi_installs, monkeypatch):
    rocm, log = kimi_installs
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    ctx = registry.Context(platform="rocm", bi=False)
    rocm.early(ctx)
    rocm.activate(ctx)
    assert [n for n, _ in log] == ["lora", "lora_slice", "bi_lora", "addmm_", "small_m", "bi_moe", "attnres", "xlora"]
    assert all(v == "1" for _, v in log)  # K31LORA_EXPERTS defaults to 1 before lora.install


def test_kimi_rocm_variant_arm_respects_an_explicit_expert_lora_off(kimi_installs, monkeypatch):
    rocm, log = kimi_installs
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("K31LORA_EXPERTS", "0")
    rocm.activate(registry.Context(platform="rocm", bi=False))
    assert log[0] == ("lora", "0")
