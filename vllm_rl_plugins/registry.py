"""Model registry. Two stages, because vLLM calls the plugin before the model architecture is known:

- register() (plugin load, every vLLM process): for each model with code for this platform, register its vLLM model
  class lazily for its architectures and install its early hooks (engine-side, scoped to the model's own classes).
- activate(spec) (vLLM imports the model class, i.e. resolves one of its architectures): install everything else."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    archs: tuple[str, ...]
    # platform name -> module with MODEL_CLS ("module:Class"), early(ctx) and activate(ctx)
    platforms: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    platform: str
    bi: bool


_EARLY: set[str] = set()
_ACTIVE: set[str] = set()


def context() -> Context:
    from .platforms import current

    return Context(platform=current(), bi=os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1")


def platform_module(spec: ModelSpec, ctx: Context):
    name = spec.platforms.get(ctx.platform)
    return None if name is None else importlib.import_module(name)


def register(models=None) -> None:
    if models is None:
        from .models import MODELS as models
    ctx = context()
    for spec in models:
        mod = platform_module(spec, ctx)
        if mod is None:
            continue
        from vllm import ModelRegistry

        for arch in spec.archs:
            ModelRegistry.register_model(arch, mod.MODEL_CLS)
        if spec.name not in _EARLY:
            mod.early(ctx)
            _EARLY.add(spec.name)


def activate(spec: ModelSpec) -> None:
    ctx = context()
    mod = platform_module(spec, ctx)
    if mod is None or spec.name in _ACTIVE:
        return
    mod.activate(ctx)
    _ACTIVE.add(spec.name)
