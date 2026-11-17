"""vLLM general plugins for RL rollouts: LoRA weight sync and batch-invariant inference, per model and platform.

One vLLM entry point (register). Each model under models/ is activated only when vLLM loads one of its architectures
on a platform it supports; other models see no patches. See registry.py."""


def register():
    from .registry import register as _register

    _register()
