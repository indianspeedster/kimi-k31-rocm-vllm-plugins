# Kimi-K3.1 ROCm vLLM plugins

vLLM general plugins for LoRA RL on Kimi-K3.1 (MXFP4 routed experts) with a ROCm vLLM build on MI355X (gfx950).
They are used by verl's `examples/grpo_trainer/run_kimi_k3_1_megatron_lite.sh` and its ROCm Dockerfile, which
installs this repo at a pinned commit.

| Plugin | Active when | What it does |
|---|---|---|
| `k31lora` | always | LoRA for the ROCm Kimi-K3.1 model (dense modules and routed experts) and the receive side of per-rank expert-LoRA slices. Under `VLLM_BATCH_INVARIANT=1`: the batch-invariant MXFP4 MoE with expert LoRA (also used by the trainer), the one-rounding LoRA epilogue, and pinned AttnRes tiling. `k31lora.bi_env` pins every switch that changes bits. |
| `k31kdabi` | `VLLM_BATCH_INVARIANT=1` | Batch-invariant KDA: the trainer's FLA expressions, with a row engine for decode under FULL CUDA graphs. |
| `k31xlora` | `VLLM_BATCH_INVARIANT=0` | W4A16 expert LoRA on AITER for the batch-variant baseline. |

Install all of them; each plugin checks `VLLM_BATCH_INVARIANT` and stays inactive outside its arm.

```sh
pip install --no-deps git+https://github.com/<owner>/kimi-k31-rocm-vllm-plugins@<commit>
```
