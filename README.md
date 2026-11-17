# vllm-rl-plugins

vLLM general plugins for RL rollouts: LoRA weight sync and batch-invariant inference, so that rollout and trainer
log-probs can match bitwise. Code is organised per model and platform; a model's plugins are active only when vLLM
loads that model on a platform they support.

They are used by verl's `examples/grpo_trainer/run_kimi_k3_1_megatron_lite.sh` and its ROCm Dockerfile, which
installs this repo at a pinned commit.

```sh
pip install --no-deps git+https://github.com/indianspeedster/vllm-rl-plugins@<commit>
```

## Layout

| path | what |
|---|---|
| `vllm_rl_plugins/__init__.py` | `register()`, the one `vllm.general_plugins` entry point |
| `vllm_rl_plugins/registry.py` | activates a model's code from the architecture vLLM loads, the platform and `VLLM_BATCH_INVARIANT` |
| `vllm_rl_plugins/common/` | model-independent: the batch-invariance switch contract, LoRA BI fixes, the receive side of per-rank expert-LoRA slices |
| `vllm_rl_plugins/platforms/<platform>/` | platform-specific, model-independent (ROCm: small-M tiles for vLLM's BI matmul) |
| `vllm_rl_plugins/models/<model>/` | one folder per model, with its platform code in `<platform>/` |
| `compat/` | old import names (`k31lora`, `k31kdabi`, `k31dsabi`, `k31xlora`) for the trainer; aliases, not plugins |

## Models

### Kimi-K3 / Kimi-K3.1 (`models/kimi_k31`)

Architectures `KimiK3ForConditionalGeneration` and `KimiK31ForConditionalGeneration`. Platforms: ROCm (MI355X,
gfx950; MXFP4 routed experts).

| arm | active |
|---|---|
| both | LoRA model support (dense modules and routed experts with `K31LORA_EXPERTS=1`), the receive side of per-rank expert-LoRA slices, the switch contract (`models/kimi_k31/env.py` pins every switch that changes bits) |
| `VLLM_BATCH_INVARIANT=1` | the batch-invariant MXFP4 MoE with expert LoRA (also used by the trainer), the one-rounding LoRA epilogue and small-M tiles, pinned AttnRes tiling, BI KDA (the trainer's FLA expressions, a row engine for decode under FULL CUDA graphs), BI DSA (sparse MQA for every row, a deterministic indexer, one-split sparse MLA) |
| `VLLM_BATCH_INVARIANT=0` | W4A16 expert LoRA on AITER (`xlora`; `K31LORA_EXPERTS` defaults to 1) |

## Adding a model or a platform

- A model: add `models/<model>/__init__.py` with a `SPEC` (`registry.ModelSpec`: name, archs, platform modules) and
  list it in `models/__init__.py`. Each platform module defines `MODEL_CLS`, `early(ctx)` (hooks vLLM needs in the
  engine process, scoped to the model's own classes) and `activate(ctx)` (everything else, run when vLLM imports the
  model class).
- A platform for an existing model: add `models/<model>/<platform>/` and an entry in its `SPEC.platforms`; code that
  is the same for every model on that platform goes in `platforms/<platform>/`.
- Fixes that change bits (`common/bi_lora`, `platforms/rocm/bi_matmul`) are opt-in: a model's `activate` calls them.

## Tests

```sh
python -m pytest tests            # CPU: switch contract, slice checks, registry, packaging; GPU tests are skipped
python -m pytest tests            # in the ROCm vLLM image on one MI355X: also tests/models/kimi_k31/rocm/gpu (about 30 s)
```

The Kimi-K3.1 ROCm GPU tests check the DSA indexer top-k against a reference and the MQA logits against an FP64
reference, that a row's results do not depend on the batch, and that KDA gives a token the same bits however its
sequence is cut or packed. `golden.json` holds output hashes; `K31_GOLDEN_UPDATE=1` rewrites them.
